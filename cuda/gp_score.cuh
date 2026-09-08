// 诊断与在线Engine复用同一个warp评分器。
#pragma once
#include "gp_faco/program.hpp"
#include "gp_faco/program_ops.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace gp_faco::cuda_detail {

static __global__ void score_actions(Program program, const float* features, const std::uint32_t* masks,
                             float* scores, std::int32_t* actions, int colonies) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int colony = thread / 32;
    const int lane = thread % 32;
    // 整个 warp 同时退出；有效 warp 的所有 lane 都参与后续 collective。
    if (colony >= colonies) return;
    float stack[6]{};
    int top = 0;
    for (int i = 0; i < program.length; ++i) {
        const int op = program.opcode[i];
        if (op == 0) {
            stack[top++] = features[(program.operand[i] * colonies + colony) * 32 + lane];
        } else if (op == 1) {
            stack[top++] = program.constants[program.operand[i]];
        } else {
            const float right = op == 7 ? 0.0f : stack[--top];
            const float left = stack[--top];
            stack[top++] = apply_operation(op, left, right);
        }
    }
    const float value = stack[0];
    scores[colony * 32 + lane] = value;
    const bool legal = (masks[colony] >> lane) & 1u;
    float maximum = legal ? value : -CUDART_INF_F;
    for (int offset = 16; offset; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    int selected = legal && maximum - value <= 1e-6f ? lane : 32;
    for (int offset = 16; offset; offset /= 2) {
        const int other = __shfl_down_sync(0xffffffffu, selected, offset);
        selected = selected < other ? selected : other;
    }
    if (lane == 0) actions[colony] = selected;
}

}  // namespace gp_faco::cuda_detail
