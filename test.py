#!/usr/bin/env python3

import triton
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import utils


@gluon.jit
def test(a, b, c, stages: gl.constexpr):
    pipeline = utils.DotPipeline.make()
    copy = utils.CopyPipeline.make()

    dot = utils.Dot.make_tensor(a, b, c.data.dtype.element_ty)
    accumulator = dot.make_accumulator()

    a = utils.Descriptor.make(a)
    b = utils.Descriptor.make(b)
    c = utils.Descriptor.make(c, layout=dot.c_layout)

    schedule = utils.PipelineSchedule.make(stages, 0, a.shape[1], a.block_shape[1])

    buffers: gl.constexpr = len(schedule.stages) + 1
    a_smem = gl.allocate_shared_memory(a.ty().type, [buffers] + a.block_shape, dot.a_layout)
    b_smem = gl.allocate_shared_memory(b.ty().type, [buffers] + b.block_shape, dot.b_layout)

    a_tokens = copy.make_tokens(buffers)
    b_tokens = copy.make_tokens(buffers)

    m = gl.program_id(0) * c.block_shape[0]
    n = gl.program_id(1) * c.block_shape[1]

    while schedule.active():
        if schedule["load"].active():
            stage = schedule["load"]
            
            (copy, a_tokens) = copy.tma_load(stage.token(a_tokens), stage.smem_slice(a_smem), a, (m, stage.index()))
            (copy, b_tokens) = copy.tma_load(stage.token(b_tokens), stage.smem_slice(b_smem), b, (stage.index(), n))

        if schedule["compute"].active():
            stage = schedule["compute"]

            copy.wait(stage.token(a_tokens))
            copy.wait(stage.token(b_tokens))

            (pipeline, accumulator) = pipeline.dot(
                dot, stage.smem_slice(a_smem), stage.smem_slice(b_smem), accumulator, _explicit_count=0
            )

        schedule = schedule.advance()

    gl.store(c.data + c.offsets((m, n)), pipeline.wait(accumulator, _explicit_count=0))

    copy.destroy_tokens(a_tokens)
    copy.destroy_tokens(b_tokens)


if __name__ == "__main__":
    utils.setup_tma()

    torch.manual_seed(0)

    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    M_BLOCK = 128
    N_BLOCK = 256
    K_BLOCK = 64

    test[(triton.cdiv(M, M_BLOCK), triton.cdiv(N, N_BLOCK))](
        utils.Tensor.wrap(a, (M_BLOCK, K_BLOCK)),
        utils.Tensor.wrap(b, (K_BLOCK, N_BLOCK)),
        utils.Tensor.wrap(c, (M_BLOCK, N_BLOCK)),
        stages=("load", "1", "compute"),
        num_warps=8,
    )
