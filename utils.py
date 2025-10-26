#!/usr/bin/env python3

import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language._semantic
import triton.experimental.gluon.language.nvidia.ampere
import typing

# This enables annoying constructions that force the control flow to be
# structured, which lets things get inlined. This is more annoying to write and
# probably slightly slower but better for profiling.
INLINE = gl.constexpr(False)


@tl.core.builtin
def is_gluon(_semantic=None):
    return isinstance(_semantic, triton.experimental.gluon.language._semantic.GluonSemantic)


def setup_tma():
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))


# Like device_assert but always active
@tl.core.builtin
def precondition(cond, msg="", mask=None, _semantic=None):
    msg = tl.core._unwrap_if_constexpr(msg)
    mask = tl.core._unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)

    cond = _semantic.to_tensor(cond)
    if mask is not None:
        cond = _semantic.or_(cond, _semantic.not_(mask))
    return _semantic.tensor(_semantic.builder.create_assert(cond.handle, msg), tl.void)


@triton.jit
def sequence(length, offset=0, layout: tl.constexpr = None):
    if layout is not None:
        return gl.arange(0, length, layout) + offset
    else:
        return tl.arange(0, length) + offset


@triton.constexpr_function
def can_tma_block(element_size, shape):
    return shape[-1] * element_size >= 16


def can_tma(element_size, base, shape, strides, is_triton=False):
    valid = can_tma_block(element_size, shape) and base % 16 == 0 and strides[-1] == 1
    for i in (gl.static_range if is_triton else range)(len(strides) - 1):
        valid = valid and strides[i] * element_size % 16 == 0
    return valid


can_tma_triton = triton.jit(can_tma)


class ConstexprWrapper(typing.NamedTuple):
    value: tl.constexpr


# constexpr_function strips out constexprs, so this adds them back
@triton.constexpr_function
def constexprify(x, _semantic=None):
    if isinstance(x, tl.tuple):
        return ConstexprWrapper(tl.tuple(constexprify(i).value for i in x))
    elif x is None or (
        not isinstance(x, tl.tensor)
        and not isinstance(x, tl.tensor_descriptor)
        and not isinstance(x, triton.experimental.gluon.language.nvidia.hopper.tma.tensor_descriptor)
    ):
        return ConstexprWrapper(tl.constexpr(x))
    else:
        return ConstexprWrapper(x)


class Tensor(typing.NamedTuple):
    data: torch.Tensor
    size: tuple
    stride: tuple
    can_tma: tl.constexpr
    block_shape: tuple | None
    layout: gl._layouts.DistributedLayout | gl._layouts.SharedLayout | None

    @staticmethod
    def _can_tma(tensor: torch.Tensor, block_shape):
        return can_tma(tensor.element_size(), tensor.data_ptr(), block_shape, tensor.stride())

    @classmethod
    def wrap(cls, tensor: torch.Tensor, block_shape: tuple | None = None, layout=None):
        return cls(
            tensor,
            tensor.size(),
            tensor.stride(),
            tl.constexpr(block_shape is not None and cls._can_tma(tensor, block_shape)),
            tuple(map(tl.constexpr, block_shape)) if block_shape is not None else None,
            layout,
        )


@tl.core._aggregate
class Block:
    block_shape: tl.tuple
    shape: tl.tuple | tl.constexpr
    strides: tl.tuple | tl.constexpr
    layout: tl.constexpr

    @triton.constexpr_function
    def __init__(
        self,
        block_shape,
        shape=tl.constexpr(None),
        strides=tl.constexpr(None),
        layout=tl.constexpr(None),
    ):
        self.block_shape = constexprify(block_shape).value
        self.shape = constexprify(shape).value
        self.strides = constexprify(strides).value
        self.layout = constexprify(layout).value

    @staticmethod
    @gluon.constexpr_function
    def unslice_layout(layout, dimension):
        dimensions = layout.rank

        # It is important that this goes in reverse order, because we will
        # expand in the forward direction and the dimensions need to match
        for i in range(dimensions)[::-1]:
            if i < dimension:
                layout = gl.SliceLayout(0, layout)
            elif i > dimension:
                layout = gl.SliceLayout(layout.rank - 1, layout)

        return layout

    @triton.jit
    def _supplement(self, offsets, dimension):
        return (
            offsets[dimension]
            if len(gl.to_tensor(offsets[dimension]).shape)
            else sequence(
                self.block_shape[dimension],
                offsets[dimension],
                Block.unslice_layout(self.layout, dimension) if self.layout is not None else None,
            )
        )

    @triton.jit
    def offsets(self, offsets) -> tl.tensor:
        if self.layout is not None:
            _offsets = gl.zeros(self.block_shape, gl.int32, self.layout)
        else:
            _offsets = tl.zeros((), tl.int32)

        for dimension in gl.static_range(len(self.block_shape)):
            supplement = self._supplement(offsets, dimension) * self.strides[dimension]

            for i in gl.static_range(len(self.block_shape)):
                if i < dimension:
                    supplement = supplement.expand_dims(0)
                elif i > dimension:
                    supplement = supplement.expand_dims(-1)
            _offsets += supplement

        return _offsets

    @triton.jit
    def mask(self, offsets) -> tl.tensor:
        if self.layout is not None:
            mask = gl.full(self.block_shape, True, gl.int1, self.layout)
        else:
            mask = tl.full((), True, tl.int1)

        for dimension in gl.static_range(len(self.block_shape)):
            supplement = self._supplement(offsets, dimension) < self.shape[dimension]

            for i in gl.static_range(len(self.block_shape)):
                if i < dimension:
                    supplement = supplement.expand_dims(0)
                elif i > dimension:
                    supplement = supplement.expand_dims(-1)
            mask &= supplement

        return mask


@tl.core._aggregate
class Descriptor:
    data: tl.tensor
    shape: tuple | tl.tuple
    strides: tuple | tl.tuple
    block_shape: tuple | tl.tuple
    descriptor: tl.tensor_descriptor | gl.nvidia.hopper.tma.tensor_descriptor | tl.tensor
    can_tma: tl.tensor | tl.constexpr
    layout: tl.constexpr

    @triton.constexpr_function
    def __init__(self, data, shape, strides, block_shape, descriptor, can_tma, layout):
        self.data = data
        self.shape = constexprify(shape).value
        self.strides = constexprify(strides).value
        self.block_shape = constexprify(block_shape).value
        self.descriptor = constexprify(descriptor).value
        self.can_tma = constexprify(can_tma).value
        self.layout = constexprify(layout).value

    @staticmethod
    @gluon.jit
    def tensor_descriptor(base, shape, strides, block_shape, dtype):
        if not can_tma_block(dtype.itemsize, block_shape):
            return base
        elif is_gluon():
            return gl.nvidia.hopper.tma.make_tensor_descriptor(
                base, shape, strides, block_shape, Layout.smem(block_shape, dtype).layout
            )
        else:
            return tl.make_tensor_descriptor(base, shape, strides, block_shape)

    @staticmethod
    @triton.jit
    def make(
        tensor: Tensor | triton.language.tensor, size=None, stride=None, block_shape=None, layout: gl.constexpr = None
    ):
        if not is_gluon() or layout is not None:
            _layout: gl.constexpr = layout
        else:
            _layout: gl.constexpr = Layout.gmem(tensor.block_shape, tensor.data.dtype.element_ty).layout

        if tensor.__class__ is tl.tuple:
            return Descriptor(
                tensor.data,
                tensor.size if size is None else size,
                tensor.stride if stride is None else stride,
                tensor.block_shape if block_shape is None else block_shape,
                Descriptor.tensor_descriptor(
                    tensor.data, tensor.size, tensor.stride, tensor.block_shape, tensor.data.dtype.element_ty
                ),
                tensor.can_tma,
                _layout,
            )
        else:
            return Descriptor(
                tensor,
                size,
                stride,
                block_shape,
                Descriptor.tensor_descriptor(tensor, size, stride, block_shape, tensor.dtype.element_ty),
                can_tma_triton(
                    tensor.dtype.element_ty.itemsize, tensor.cast(tl.uint64, bitcast=True), block_shape, stride, True
                ),
                _layout,
            )

    # TODO: fix returning dtypes
    @triton.jit
    def ty(self) -> tl.tensor:
        if self.layout is not None:
            return gl.zeros((), self.data.dtype.element_ty, gl.AutoLayout())
        else:
            return tl.zeros((), self.data.dtype.element_ty)

    @triton.jit
    def block(self):
        return Block(self.block_shape, self.shape, self.strides, self.layout)

    @triton.jit
    def offsets(self, offsets):
        return self.block().offsets(offsets)

    @triton.jit
    def mask(self, offsets):
        return self.block().mask(offsets)

    @triton.jit
    def load(self, offsets, other=None) -> tl.tensor:
        if not self.can_tma or is_gluon():
            return tl.load(self.data + self.offsets(offsets), mask=self.mask(offsets), other=other)
        elif other is None or other == 0:
            return self.descriptor.load(offsets)
        else:
            return tl.where(self.mask(offsets), self.descriptor.load(offsets), other)

    @triton.jit
    def store(self, offsets, value):
        if not self.can_tma or is_gluon():
            tl.store(self.data + self.offsets(offsets), value, mask=self.mask(offsets))
        else:
            self.descriptor.store(offsets, value)


@gluon.constexpr_function
def coalesced_block(size, dtype):
    # Max load size is 128 bits
    ideal_width = 128 // dtype.primitive_bitwidth
    width = min(size, ideal_width)
    # Ideal coalescing is 128 bytes
    ideal_coalesce = 128 // dtype.itemsize
    coalesce = min(size, ideal_coalesce)
    return gl.tuple((width, coalesce))


@gluon.constexpr_function
def gmem_layout(block_shape, dtype, preferred_axis, num_warps, num_threads):
    # Flipping things makes the logic easier
    block_shape = block_shape[::-1]
    if preferred_axis is not None:
        preferred_axis = -(preferred_axis + 1)

    (width, coalesce) = coalesced_block(block_shape[0], dtype)

    reg_bases = []
    lane_bases = []
    warp_bases = []

    def size(dimension):
        return sum([basis[dimension] for basis in (reg_bases + lane_bases + warp_bases)]) + 1

    def has_space(bases, count):
        return 2 ** len(bases) < count

    def add_basis(bases, dimension):
        basis = [0] * len(block_shape)
        basis[dimension] = size(dimension)
        bases.append(basis)

    count = math.prod(block_shape)

    # Number of elements at each level
    lane_count = num_threads // num_warps
    reg_count = count // num_threads

    # Distribute the load's elements into registers
    while size(0) < width and has_space(reg_bases, reg_count):
        add_basis(reg_bases, 0)

    # Distribute the coalescing layout across threads in the warp
    while size(0) < coalesce and has_space(lane_bases, lane_count):
        add_basis(lane_bases, 0)

    if preferred_axis is not None:
        while size(preferred_axis) < block_shape[preferred_axis]:
            # We'd like to have this dimension in registers as much as possible
            if has_space(reg_bases, reg_count):
                add_basis(reg_bases, preferred_axis)
            # If we can't do registers anymore, other threads in the warp are
            # fine (since we can warp shuffle to get them)
            elif has_space(lane_bases, lane_count):
                add_basis(lane_bases, preferred_axis)
            # Otherwise there is no benefit, everything's equally accessible in
            # the block
            # TODO: except maybe for warpgroup?
            else:
                break

    # Distribute the remaining elements as evenly as possible to not unfairly
    # pessimize reductions in that dimension
    def distribute_remainder(bases, count):
        while has_space(bases, count):
            remaining = [block_shape[dimension] // size(dimension) for dimension in range(len(block_shape))]
            min_dimension = remaining.index(max(remaining))
            add_basis(bases, min_dimension)

    distribute_remainder(reg_bases, reg_count)
    distribute_remainder(lane_bases, lane_count)

    # Now we just fill whatever is left
    for dimension in range(len(block_shape)):
        while size(dimension) < block_shape[dimension]:
            add_basis(warp_bases, dimension)

    def flip_bases(bases):
        return [basis[::-1] for basis in bases]

    return gl.DistributedLinearLayout(
        reg_bases=flip_bases(reg_bases),
        lane_bases=flip_bases(lane_bases),
        warp_bases=flip_bases(warp_bases),
        block_bases=[],
        shape=block_shape[::-1],
    )


@tl.core._aggregate
class Layout:
    layout: gl.constexpr

    @triton.constexpr_function
    def __init__(self, layout):
        self.layout = constexprify(layout).value

    # Note: we only support TMA-style shapes (last dimension contiguous)
    # TODO: take a full order parameter?
    @staticmethod
    @triton.jit
    def gmem(block_shape, dtype, preferred_axis=None):
        return Layout(
            gmem_layout(block_shape, dtype, preferred_axis, gl.num_warps(), triton.language.extra.cuda.num_threads())
        )

    @staticmethod
    @triton.jit
    def smem(block_shape, dtype):
        return Layout(gl.NVMMASharedLayout.get_default_for(block_shape, dtype))


@tl.core._aggregate
class DotAccumulator:
    version: gl.constexpr
    position: gl.tensor
    accumulator: gl.tensor | gl.nvidia.hopper.warpgroup_mma_accumulator

    @gluon.constexpr_function
    def __init__(self, version, position, accumulator):
        self.version = constexprify(version).value
        self.position = position
        self.accumulator = accumulator


@tl.core._aggregate
class DotPipeline:
    current: gl.tensor

    @gluon.constexpr_function
    def __init__(self, current):
        self.current = current

    @staticmethod
    @gluon.jit
    def make():
        return DotPipeline(gl.to_tensor(0))

    @staticmethod
    @gluon.jit
    def _wait(count, deps, _explicit_count):
        # Works around a bug in ptxas where it can't determine register
        # dependencies for the accumulator in the pipeline.
        if _explicit_count is not None:
            return gl.nvidia.hopper.warpgroup_mma_wait(_explicit_count, deps=deps)

        for i in gl.static_range(2):
            if count <= i:
                return gl.nvidia.hopper.warpgroup_mma_wait(i, deps=deps)

        precondition(False, "Pipeline too deep")

    @tl.core.must_use_result
    @gluon.jit
    def dot(self, dot: "Dot", a, b, c: DotAccumulator, _explicit_count=None):
        if dot.version == 3:
            result = self.wait(c, _explicit_count)
            accumulator = gl.nvidia.hopper.warpgroup_mma(a, b, result, is_async=True)
        elif dot.version == 2:
            accumulator = triton.experimental.gluon.language.nvidia.ampere.mma_v2(a, b, c.accumulator)
        elif dot.version == 1:
            accumulator = gl.dot_fma(a, b, c.accumulator)
        return (
            DotPipeline(self.current + 1),
            DotAccumulator(c.version, self.current, accumulator),
        )

    @tl.core.must_use_result
    @gluon.jit
    def dot_sync(self, dot: "Dot", a, b, c=None):
        if dot.version == 3:
            return gl.nvidia.hopper.warpgroup_mma(a, b, c, is_async=False)
        else:
            (pipeline, accumulator) = self.dot(dot, a, b, dot.make_accumulator(c))
            result = pipeline.wait(accumulator)
            return (pipeline, result.accumulator)

    @gluon.jit
    def wait(self, accumulator: DotAccumulator, _explicit_count=None):
        if accumulator.version == 3:
            return DotPipeline._wait(
                self.current - 1 - accumulator.position, (accumulator.accumulator,), _explicit_count
            )
        else:
            return accumulator.accumulator


@tl.core._aggregate
class Dot:
    version: gl.constexpr
    a_shape: gl.tuple
    b_shape: gl.tuple
    c_shape: gl.tuple
    in_dtype: gl.constexpr
    out_dtype: gl.constexpr
    a_layout: gl.constexpr
    b_layout: gl.constexpr
    c_layout: gl.constexpr
    a_alternate_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self, version, a_shape, b_shape, c_shape, in_dtype, out_dtype, a_layout, b_layout, c_layout, a_alternate_layout
    ):
        self.version = constexprify(version).value
        self.a_shape = constexprify(a_shape).value
        self.b_shape = constexprify(b_shape).value
        self.c_shape = constexprify(c_shape).value
        self.in_dtype = constexprify(in_dtype).value
        self.out_dtype = constexprify(out_dtype).value
        self.a_layout = constexprify(a_layout).value
        self.b_layout = constexprify(b_layout).value
        self.c_layout = constexprify(c_layout).value
        self.a_alternate_layout = constexprify(a_alternate_layout).value

    @staticmethod
    @gluon.constexpr_function
    def _version(a_shape, b_shape, in_dtype, out_dtype):
        if (
            (in_dtype.itemsize == 2 and out_dtype.itemsize == 2)
            or (in_dtype.itemsize == 2 and out_dtype.itemsize == 4)
            or (in_dtype.itemsize == 4 and out_dtype.itemsize == 4)
        ):
            if a_shape[0] >= 64 and b_shape[0] >= 8 and a_shape[1] >= 256 // in_dtype.primitive_bitwidth:
                return 3
            else:
                return 2
        elif in_dtype == out_dtype:
            return 1
        else:
            gl.static_assert(False, "Unsupported dot")

    @staticmethod
    @gluon.jit
    def make(a_shape, b_shape, in_dtype, out_dtype, _version=None):
        version: gl.constexpr = Dot._version(a_shape, b_shape, in_dtype, out_dtype) if _version is None else _version

        if version == 3:
            max_n: gl.constexpr = a_shape[0] * b_shape[1] // (gl.num_warps() * 16)
            c_layout: gl.constexpr = gl.NVMMADistributedLayout(
                [version, 0],
                warps_per_cta=[gl.num_warps(), 1],
                instr_shape=[
                    16,
                    max_n if max_n < b_shape[1] else b_shape[1],
                    256 // in_dtype.primitive_bitwidth,
                ],
            )
            a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(a_shape, in_dtype)
            b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(b_shape, in_dtype)
            a_alternate_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, 32 // in_dtype.primitive_bitwidth)
        elif version == 2:
            c_layout: gl.constexpr = gl.NVMMADistributedLayout(
                [version, 0], warps_per_cta=[4, gl.num_warps() // 4], instr_shape=[16, 8]
            )
            a_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, 32 // in_dtype.primitive_bitwidth)
            b_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, 32 // in_dtype.primitive_bitwidth)
            a_alternate_layout: gl.constexpr = None
        elif version == 1:
            (width, coalesce) = coalesced_block(b_shape[1], out_dtype)
            c_layout: gl.constexpr = gl.BlockedLayout(
                size_per_thread=[1, width],
                threads_per_warp=[
                    triton.language.extra.cuda.num_threads() // gl.num_warps() // (coalesce // width),
                    coalesce // width,
                ],
                warps_per_cta=[4, gl.num_warps() // 4],
                order=[1, 0],
            )
            a_layout: gl.constexpr = gl.DotOperandLayout(0, c_layout, 0)
            b_layout: gl.constexpr = gl.DotOperandLayout(1, c_layout, 0)
            a_alternate_layout: gl.constexpr = None
        else:
            tl.static_assert(False, "Invalid version")

        return Dot(
            version,
            a_shape,
            b_shape,
            (a_shape[0], b_shape[1]),
            in_dtype,
            out_dtype,
            a_layout,
            b_layout,
            c_layout,
            a_alternate_layout,
        )

    @staticmethod
    @gluon.jit
    def make_tensor(a, b, out_dtype, _version=None):
        return Dot.make(a.block_shape, b.block_shape, a.data.dtype.element_ty, out_dtype, _version)

    @gluon.jit
    def make_accumulator(self, value=None):
        result = gl.zeros(self.c_shape, self.out_dtype, self.c_layout) if value is None else value
        return DotAccumulator(
            self.version,
            gl.to_tensor(-1),
            gl.nvidia.hopper.warpgroup_mma_init(result) if self.version == 3 else result,
        )


@tl.core._aggregate
class CopyToken:
    tma: gl.tensor
    position: gl.tensor
    barrier: gl.shared_memory_descriptor
    phase: gl.tensor

    @gluon.constexpr_function
    def __init__(self, tma, position, barrier, phase):
        self.tma = tma
        self.position = position
        self.barrier = barrier
        self.phase = phase


@tl.core._aggregate
class CopyTokens:
    tokens: gl.tuple

    @gluon.constexpr_function
    def __init__(self, tokens):
        self.tokens = tokens

    @gluon.jit
    def __getitem__(self, index) -> "CopyTokensSlice":
        return CopyTokensSlice(self.tokens, index)


@tl.core._aggregate
class CopyTokensSlice:
    tokens: gl.tuple
    index: gl.tensor

    @gluon.constexpr_function
    def __init__(self, tokens, index):
        self.tokens = tokens
        self.index = index

    @gluon.jit
    def get(self):
        if INLINE:
            token = self.tokens[0]
            for i in gl.static_range(len(self.tokens)):
                if i == self.index:
                    token = self.tokens[i]
            return token
        else:
            for i in gl.static_range(len(self.tokens)):
                if i == self.index:
                    return self.tokens[i]

    @gluon.jit
    def update(self, token):
        if INLINE:
            result = CopyTokens(self.tokens)
            for i in gl.static_range(len(self.tokens)):
                if i == self.index:
                    result = CopyTokens(self.tokens[:i] + (token,) + self.tokens[i + 1 :])
            return result
        else:
            for i in gl.static_range(len(self.tokens)):
                if i == self.index:
                    return CopyTokens(self.tokens[:i] + (token,) + self.tokens[i + 1 :])


@tl.core._aggregate
class CopyPipeline:
    current: gl.tensor

    @gluon.constexpr_function
    def __init__(self, current):
        self.current = current

    @staticmethod
    @gluon.jit
    def make():
        return CopyPipeline(gl.to_tensor(0))

    @gluon.jit
    def make_token(self):
        barrier = gl.allocate_shared_memory(gl.int64, [1], gl.nvidia.hopper.mbarrier.MBarrierLayout())
        gl.nvidia.hopper.mbarrier.init(barrier, 1)
        return CopyToken(gl.to_tensor(False), gl.to_tensor(-1), barrier, gl.to_tensor(1))

    @gluon.jit
    def make_tokens(self, count):
        barrier = gl.allocate_shared_memory(gl.int64, [count, 1], gl.nvidia.hopper.mbarrier.MBarrierLayout())
        for i in gl.static_range(count):
            gl.nvidia.hopper.mbarrier.init(barrier.index(i), 1)
        tokens = ()
        for i in gl.static_range(count):
            tokens += (CopyToken(gl.to_tensor(False), gl.to_tensor(-1), barrier.index(i), gl.to_tensor(1)),)
        return CopyTokens(tokens)

    @gluon.jit
    def destroy_token(self, token):
        gl.nvidia.hopper.mbarrier.invalidate(token.barrier)

    @gluon.jit
    def destroy_tokens(self, tokens: CopyTokens):
        for i in gl.static_range(len(tokens.tokens)):
            gl.nvidia.hopper.mbarrier.invalidate(tokens.tokens[i].barrier)

    @typing.overload
    def load(self, token: CopyToken, destination, source, mask=None) -> tuple["CopyPipeline", CopyToken]: ...

    @typing.overload
    def load(self, token: CopyTokensSlice, destination, source, mask=None) -> tuple["CopyPipeline", CopyTokens]: ...

    @tl.core.must_use_result
    @gluon.jit
    def load(self, token: CopyToken | CopyTokensSlice, destination, source, mask=None):
        if token.__class__ is CopyTokensSlice:
            (pipeline, _token) = self.load(token.get(), destination, source, mask)
            return (pipeline, token.update(_token))
        else:
            triton.experimental.gluon.language.nvidia.ampere.async_copy.async_copy_global_to_shared(
                destination, source, mask
            )
            triton.experimental.gluon.language.nvidia.ampere.async_copy.commit_group()
            return (
                CopyPipeline(self.current + 1),
                CopyToken(gl.to_tensor(False), self.current, token.barrier, token.phase),
            )

    @typing.overload
    def tma_load(
        self, token: CopyToken, destination, source: Descriptor, offsets
    ) -> tuple["CopyPipeline", CopyToken]: ...

    @typing.overload
    def tma_load(
        self, token: CopyTokensSlice, destination, source: Descriptor, offsets
    ) -> tuple["CopyPipeline", CopyTokens]: ...

    @tl.core.must_use_result
    @gluon.jit
    def tma_load(self, token: CopyToken | CopyTokensSlice, destination, source: Descriptor, offsets):
        result = (self, token)
        if token.__class__ is CopyTokensSlice:
            (pipeline, _token) = self.tma_load(token.get(), destination, source, offsets)
            result = (pipeline, token.update(_token))
        elif source.can_tma:
            gl.nvidia.hopper.mbarrier.expect(token.barrier, source.descriptor.block_type.nbytes)
            gl.nvidia.hopper.tma.async_copy_global_to_shared(source.descriptor, offsets, token.barrier, destination)
            result = (self, CopyToken(gl.to_tensor(True), token.position, token.barrier, token.phase ^ 1))
        else:
            result = self.load(token, destination, source.data + source.offsets(offsets), source.mask(offsets))
        return result

    @gluon.jit
    def _wait(self, token):
        count = self.current - 1 - token.position
        # Note that this check has to be at the top-level, otherwise it doesn't
        # actually take effect
        if INLINE:
            waited = False
            for i in gl.static_range(2):
                if count <= i and not waited:
                    triton.experimental.gluon.language.nvidia.ampere.async_copy.wait_group(i)
                    waited = True
            
            precondition(waited, "Pipeline too deep")
        else:
            for i in gl.static_range(2):
                if count <= i:
                    triton.experimental.gluon.language.nvidia.ampere.async_copy.wait_group(i)
                    return

            precondition(False, "Pipeline too deep")

    @gluon.jit
    def wait(self, token: CopyToken | CopyTokensSlice):
        if token.__class__ is CopyTokensSlice:
            self.wait(token.get())
        elif token.tma:
            gl.nvidia.hopper.mbarrier.wait(token.barrier, token.phase)
        else:
            self._wait(token)

    @gluon.jit
    def shared_fence(self):
        gl.nvidia.hopper.fence_async_shared()


@tl.core._aggregate
class PipelineRange:
    start: gl.tensor | gl.constexpr
    end: gl.tensor | gl.constexpr
    step: gl.tensor | gl.constexpr

    @gluon.constexpr_function
    def __init__(self, start, end, step):
        self.start = constexprify(start).value
        self.end = constexprify(end).value
        self.step = constexprify(step).value


@tl.core._aggregate
class PipelineStage:
    base_index: gl.tensor
    offset: gl.constexpr
    range: PipelineRange

    @gluon.constexpr_function
    def __init__(self, base_index, offset, range):
        self.base_index = base_index
        self.offset = constexprify(offset).value
        self.range = range

    @gluon.jit
    def raw_index(self):
        return self.base_index + self.offset

    @gluon.jit
    def buffer_index(self, buffers):
        return self.raw_index() % buffers

    @gluon.jit
    def index(self):
        return self.raw_index() * self.range.step

    @gluon.jit
    def token(self, tokens: CopyTokens):
        return tokens[self.buffer_index(len(tokens.tokens))]

    @gluon.jit
    def smem_slice(self, smem: gl.shared_memory_descriptor):
        return smem.index(self.buffer_index(smem.shape[0]))

    @gluon.jit
    def active(self):
        return self.range.start <= self.index() and self.index() < self.range.end


# Exists solely to smuggle out constexprs
@tl.core._aggregate
class PipelineDistance:
    distance: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, distance):
        self.distance = constexprify(distance).value


@tl.core._aggregate
class PipelineSchedule:
    stages: gl.constexpr
    range: PipelineRange | gl.constexpr
    index: gl.tensor

    @gluon.constexpr_function
    def __init__(self, stages, range, index):
        self.stages = gl.constexpr(stages)
        self.range = range if range is not None else gl.constexpr(range)
        self.index = index

    @staticmethod
    @gluon.jit
    def make(stages, start=None, end=None, step=None):
        if start is None:
            range = None
        elif end is None:
            range = PipelineRange(0, start, 1)
        elif step is None:
            range = PipelineRange(start, end, 1)
        else:
            range = PipelineRange(start, end, step)
        return PipelineSchedule(gl.constexpr(stages), range, gl.to_tensor(0))

    @gluon.jit
    def __getitem__(self, stage) -> PipelineStage:
        for i in gl.static_range(len(self.stages)):
            if stage == self.stages[i]:
                return PipelineStage(self.index, -i, self.range)

    @gluon.jit
    def distance(self, stage1, stage2):
        distance: gl.constexpr = self[stage1].offset - self[stage2].offset
        return PipelineDistance(distance)

    @tl.core.must_use_result
    @gluon.jit
    def advance(self):
        return PipelineSchedule(self.stages, self.range, self.index + 1)

    @gluon.jit
    def active(self):
        return self.range.start + (self.index - len(self.stages)) * self.range.step < self.range.end
