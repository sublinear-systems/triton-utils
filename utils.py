#!/usr/bin/env python3

import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.ampere
import triton.experimental.gluon.language.nvidia.hopper
import typing


def setup_tma():
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))


@triton.jit
def sequence(length, offset=0, layout: tl.constexpr = None):
    if layout is not None:
        return gl.arange(0, length, layout) + offset
    else:
        return tl.arange(0, length) + offset


class Tensor(typing.NamedTuple):
    data: torch.Tensor
    size: tuple
    stride: tuple
    can_tma: tl.constexpr
    block_shape: tuple | None
    layout: gl._layouts.DistributedLayout | gl._layouts.SharedLayout | None

    @staticmethod
    def _can_tma(tensor: torch.Tensor, block_shape):
        return (
            block_shape[-1] * tensor.element_size() >= 16
            and tensor.data_ptr() % 16 == 0
            and tensor.stride(-1) == 1
            and all([(stride * tensor.element_size()) % 16 == 0 for stride in tensor.stride()[:-1]])
        )

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

    def __init__(
        self,
        block_shape,
        shape=tl.constexpr(None),
        strides=tl.constexpr(None),
        layout=tl.constexpr(None),
    ):
        self.block_shape = block_shape
        self.shape = shape
        self.strides = strides
        self.layout = layout

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
    descriptor: tl.tensor_descriptor | tl.constexpr
    layout: tl.constexpr

    def __init__(self, data, shape, strides, block_shape, descriptor, layout):
        self.data = data
        self.shape = shape
        self.strides = strides
        self.block_shape = block_shape
        self.descriptor = descriptor
        self.layout = layout

    @staticmethod
    @triton.jit
    def make(tensor: Tensor, block_shape=None, layout=None):
        return Descriptor(
            tensor.data,
            tensor.size,
            tensor.stride,
            tensor.block_shape if tensor.block_shape is not None else block_shape,
            tl.make_tensor_descriptor(tensor.data, tensor.size, tensor.stride, tensor.block_shape)
            if tensor.can_tma and layout is None
            else None,
            layout
            if layout is None
            else (
                gl.constexpr(Layout.gmem(tensor.block_shape, tensor.data.dtype.element_ty).layout)
                if layout == "gmem"
                else layout
            ),
        )

    # TODO clean up this API
    @staticmethod
    @triton.jit
    def make_inline(data: triton.language.tensor, size, stride, block_shape, can_tma=True, layout: tl.constexpr = None):
        return Descriptor(
            data,
            size,
            stride,
            block_shape,
            tl.make_tensor_descriptor(data, size, stride, block_shape) if can_tma and layout is None else None,
            layout,
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
        if self.descriptor is None or self.layout is not None:
            if self.layout is not None:
                return gl.load(self.data + self.offsets(offsets), mask=self.mask(offsets), other=other)
            else:
                return tl.load(self.data + self.offsets(offsets), mask=self.mask(offsets), other=other)
        elif other is None or other == 0:
            return self.descriptor.load(offsets)
        else:
            return tl.where(self.mask(offsets), self.descriptor.load(offsets), other)

    @triton.jit
    def store(self, offsets, value):
        if self.descriptor is not None:
            self.descriptor.store(offsets, value)
        else:
            tl.store(self.data + self.offsets(offsets), value, mask=self.mask(offsets))


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

    def __init__(self, layout):
        self.layout = layout

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
    result: gl.tensor
    accumulator: triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma_accumulator

    def __init__(self, version, position, result, accumulator):
        self.version = version
        self.position = position
        self.result = result
        self.accumulator = accumulator

    @gluon.jit
    def update_dot(self, position, accumulator):
        if self.version == 3:
            return DotAccumulator(self.version, position, self.result, accumulator)
        else:
            return DotAccumulator(self.version, position, accumulator, self.accumulator)

    @gluon.jit
    def update_wait(self, result):
        return DotAccumulator(self.version, self.position, result, self.accumulator)


@tl.core._aggregate
class DotPipeline:
    current: gl.tensor

    def __init__(self, current):
        self.current = current

    @staticmethod
    @gluon.jit
    def make():
        return DotPipeline(gl.to_tensor(0))

    @gluon.jit
    def shared_fence(self):
        triton.experimental.gluon.language.nvidia.hopper.fence_async_shared()

    @tl.core.must_use_result
    @gluon.jit
    def dot(self, dot: "Dot", a, b, c: DotAccumulator):
        if dot.version == 3:
            accumulator_result = triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma(
                a, b, c.result, is_async=True
            )
        elif dot.version == 2:
            accumulator_result = triton.experimental.gluon.language.nvidia.ampere.mma_v2(a, b, c.result)
        elif dot.version == 1:
            accumulator_result = gl.dot_fma(a, b, c.result)
        return (
            DotPipeline(self.current + 1),
            c.update_dot(self.current, accumulator_result),
        )

    @tl.core.must_use_result
    @gluon.jit
    def dot_sync(self, dot: "Dot", a, b, c=None):
        (pipeline, accumulator) = self.dot(dot, a, b, dot.make_accumulator(c))
        result = pipeline.wait(accumulator)
        return (pipeline, result.result)

    @staticmethod
    @gluon.jit
    def _wait(count, deps):
        for i in gl.static_range(4):
            if count <= i:
                return triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma_wait(i, deps=deps)

        gl.device_assert(False, "Pipeline too deep")

    @tl.core.must_use_result
    @gluon.jit
    def wait(self, accumulator: DotAccumulator):
        if accumulator.version == 3:
            return accumulator.update_wait(
                DotPipeline._wait(self.current - 1 - accumulator.position, (accumulator.accumulator,))
            )
        else:
            return accumulator


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

    def __init__(
        self, version, a_shape, b_shape, c_shape, in_dtype, out_dtype, a_layout, b_layout, c_layout, a_alternate_layout
    ):
        self.version = version
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.c_shape = c_shape
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.a_layout = a_layout
        self.b_layout = b_layout
        self.c_layout = c_layout
        self.a_alternate_layout = a_alternate_layout

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
            max_n: gl.constexpr = b_shape[1] * a_shape[0] // (gl.num_warps() * 16)
            c_layout: gl.constexpr = gl.NVMMADistributedLayout(
                [version, 0],
                warps_per_cta=[4, gl.num_warps() // 4],
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
            a_alternate_layout = None
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
            a_alternate_layout = None
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
            result,
            triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma_init(result),
        )


@tl.core._aggregate
class CopyToken:
    position: gl.tensor

    def __init__(self, position):
        self.position = position


@tl.core._aggregate
class CopyPipeline:
    current: gl.tensor

    def __init__(self, current):
        self.current = current

    @staticmethod
    @gluon.jit
    def make():
        return CopyPipeline(gl.to_tensor(0))

    @gluon.jit
    def make_token(self):
        return CopyToken(gl.to_tensor(-1))

    @tl.core.must_use_result
    @gluon.jit
    def load(self, destination, source, mask=None):
        triton.experimental.gluon.language.nvidia.ampere.async_copy.async_copy_global_to_shared(
            destination, source, mask
        )
        triton.experimental.gluon.language.nvidia.ampere.async_copy.commit_group()
        return (CopyPipeline(self.current + 1), CopyToken(self.current))

    @gluon.jit
    def wait(self, token: CopyToken):
        count = self.current - 1 - token.position
        for i in gl.static_range(4):
            if count <= i:
                triton.experimental.gluon.language.nvidia.ampere.async_copy.wait_group(i)
                return

        # TODO: we need to come up with a stronger assert for this that can't
        # be compiled out
        gl.device_assert(False, "Pipeline too deep")


class PipelineRange(typing.NamedTuple):
    start: gl.tensor
    end: gl.tensor
    step: gl.tensor


@tl.core._aggregate
class PipelineStage:
    base_index: gl.tensor
    offset: gl.constexpr
    range: gl.tuple

    def __init__(self, base_index, offset, range):
        self.base_index = base_index
        self.offset = offset
        self.range = range

    @gluon.jit
    def raw_index(self):
        return self.base_index + self.offset

    @gluon.jit
    def index(self):
        return self.raw_index() * self.range.step

    @gluon.jit
    def active(self):
        return self.range.start <= self.index() and self.index() < self.range.end


@tl.core._aggregate
class PipelineSchedule:
    stages: gl.constexpr
    range: gl.tuple | gl.constexpr
    index: gl.tensor

    def __init__(self, stages, range, index):
        self.stages = stages
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
        return self[stage1].offset - self[stage2].offset

    @tl.core.must_use_result
    @gluon.jit
    def advance(self):
        return PipelineSchedule(self.stages, self.range, self.index + 1)

    @gluon.jit
    def active(self):
        return self.range.start + (self.index - len(self.stages)) * self.range.step < self.range.end


@tl.core._aggregate
class BufferedValue:
    data: gl.tuple
    index: gl.tensor

    def __init__(self, data, index):
        self.data = data
        self.index = index

    @staticmethod
    @gluon.jit
    def _create(data, index):
        gl.device_assert(index >= 0 and index <= len(data), "Your buffer is not balanced")
        return BufferedValue(data, index)

    @staticmethod
    @gluon.jit
    def make(initializer, count=None):
        if count is None:
            return BufferedValue._create(initializer, gl.to_tensor(len(initializer)))
        else:
            data = none_tuple(count)
            for i in gl.static_range(count):
                data = data[:i] + (initializer,) + data[i + 1 :]
            return BufferedValue._create(data, gl.to_tensor(len(data)))

    @tl.core.must_use_result
    @gluon.jit
    def push(self, value):
        return BufferedValue._create(self.data[1:] + (value,), self.index - 1)

    @gluon.jit
    def pull(self):
        for i in gl.static_range(len(self.data) - 1, -1, -1):
            if i == self.index:
                return (BufferedValue._create(self.data, self.index + 1), self.data[i])
        gl.device_assert(False, "Buffer is full")

    @gluon.jit
    def peek(self):
        for i in gl.static_range(len(self.data)):
            if i == self.index:
                return self.data[i]
        gl.device_assert(False, "Buffer is empty")

    @gluon.jit
    def get(self, index):
        return self.data[index]

    @tl.core.must_use_result
    @gluon.jit
    def update(self, index, value):
        return BufferedValue._create(self.data[:index] + (value,) + self.data[index + 1 :], self.index)
