// 诊断与在线Engine复用同一个warp评分器。
#pragma once
#include "gp_faco/program.hpp"
#include "gp_faco/program_ops.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace gp_faco::cuda_detail {

static __device__ float controller_value(const Program& program, const float* features,
                                        int colonies, int colony, int action) {
    float stack[6]{}; int top = 0;
    for (int i = 0; i < program.length; ++i) {
        const int op = program.opcode[i];
        if (op == 0) stack[top++] = features[(program.operand[i] * colonies + colony) * 32 + action];
        else if (op == 1) stack[top++] = program.constants[program.operand[i]];
        else {
            const float right = op == 7 ? 0.f : stack[--top];
            const float left = stack[--top];
            stack[top++] = apply_operation(op, left, right);
        }
    }
    return stack[0];
}

static __device__ int controller_choice(float value, bool legal, int lane) {
    float maximum = legal ? value : -CUDART_INF_F;
    for (int offset = 16; offset; offset /= 2)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    int chosen = legal && maximum - value <= 1e-6f ? lane : 32;
    for (int offset = 16; offset; offset /= 2)
        chosen = min(chosen, __shfl_down_sync(0xffffffffu, chosen, offset));
    return __shfl_sync(0xffffffffu, chosen, 0);
}

static __global__ void score_controllers(const ControllerProgram* controllers, const float* features,
    const std::uint32_t* masks, float* scores, std::int32_t* actions, int colonies, int replicas) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int colony = thread / 32, lane = thread % 32;
    if (colony >= colonies) return;
    const auto& controller = controllers[colony / replicas];
    int prefix = 0;
    if (!controller.kind) {
        const float value = controller_value(controller.trees[0], features, colonies, colony, lane);
        scores[colony * 32 + lane] = value;
        prefix = controller_choice(value, (masks[colony] >> lane) & 1u, lane);
    } else {
        // 三个阶段留在同一warp。特征不修改，重启事务由后续apply_control_action唯一执行。
        scores[colony * 32 + lane] = 0.f;
        for (int role = 0; role < 3; ++role) {
            const int width = role == 0 ? 2 : 4;
            const int stride = role == 0 ? 16 : role == 1 ? 4 : 1;
            const int action = prefix + (lane < width ? lane : 0) * stride;
            const bool legal = lane < width && ((masks[colony] >> action) & ((1u << stride) - 1u));
            const float value = controller_value(controller.trees[role], features, colonies, colony, action);
            if (lane < width) scores[colony * 32 + (role == 0 ? 0 : role == 1 ? 2 : 6) + lane] = value;
            prefix += stride * controller_choice(value, legal, lane);
        }
    }
    if (lane == 0) actions[colony] = prefix;
}

static __global__ void score_actions(Program broadcast_program, const float* features, const std::uint32_t* masks,
                             float* scores, std::int32_t* actions, int colonies,
                             const Program* population = nullptr, int replicas = 0) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int colony = thread / 32;
    const int lane = thread % 32;
    // 整个 warp 同时退出；有效 warp 的所有 lane 都参与后续 collective。
    if (colony >= colonies) return;
    // 个体维度独立于 instance/seed 随机域；每个 warp 使用对应个体的程序。
    const Program& program = population ? population[colony / replicas] : broadcast_program;
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
