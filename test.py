#!/usr/bin/env python3

import triton
import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.language.extra.cuda
import utils


@gluon.jit
def test(a, b, c):
    pipeline = utils.DotPipeline.make()
    copy = utils.CopyPipeline.make()

    dot = utils.Dot.make_tensor(a, b, c.data.dtype.element_ty)
    accumulator = dot.make_accumulator()

    a = utils.Descriptor.make(a, layout="gmem")
    b = utils.Descriptor.make(b, layout="gmem")
    c = utils.Descriptor.make(c, layout="gmem")

    schedule = utils.PipelineSchedule.make(("load", "compute"), 0, a.shape[1], a.block_shape[1])

    a_smem = gl.allocate_shared_memory(a.ty().type, [len(schedule.stages)] + a.block_shape, dot.a_layout)
    b_smem = gl.allocate_shared_memory(b.ty().type, [len(schedule.stages)] + b.block_shape, dot.b_layout)

    a_token = utils.BufferedValue.make((copy.make_token(), copy.make_token()))
    b_token = utils.BufferedValue.make((copy.make_token(), copy.make_token()))

    m = gl.program_id(0) * c.block_shape[0]
    n = gl.program_id(1) * c.block_shape[1]

    while schedule.active():
        if schedule["load"].active():
            index = schedule["load"].index()
            buffer_index = schedule["load"].raw_index() % len(schedule.stages)

            (copy, token) = copy.load(a_smem.index(buffer_index), a.data + a.offsets((m, index)))
            a_token = a_token.push(token)
            (copy, token) = copy.load(b_smem.index(buffer_index), b.data + b.offsets((index, n)))
            b_token = b_token.push(token)

        if schedule["compute"].active():
            buffer_index = schedule["compute"].raw_index() % len(schedule.stages)

            (a_token, token) = a_token.pull()
            copy.wait(token)
            (b_token, token) = b_token.pull()
            copy.wait(token)

            pipeline.shared_fence()

            (pipeline, accumulator) = pipeline.dot(
                dot, a_smem.index(buffer_index), b_smem.index(buffer_index), accumulator
            )
            accumulator = pipeline.wait(accumulator)

        schedule = schedule.advance()

    gl.store(c.data + c.offsets((m, n)), gl.convert_layout(accumulator.result, c.layout))


if __name__ == "__main__":
    torch.manual_seed(0)

    M = N = K = 4096
    M_BLOCK = 128
    N_BLOCK = 128
    K_BLOCK = 64
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    test[(triton.cdiv(M, M_BLOCK), triton.cdiv(N, N_BLOCK))](
        utils.Tensor.wrap(a, (M_BLOCK, K_BLOCK)),
        utils.Tensor.wrap(b, (K_BLOCK, N_BLOCK)),
        utils.Tensor.wrap(c, (M_BLOCK, N_BLOCK)),
    )
