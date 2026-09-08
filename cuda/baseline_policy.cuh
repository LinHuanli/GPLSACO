// 基线只替换动作选择，共享特征/区域、档案、完整重启与构造/LS内核。
#pragma once

#include "gp_faco/baseline_policy.hpp"
#include "gp_faco/factorial_policy.hpp"
#include "faco_choices.cuh"

namespace gp_faco::cuda_detail {

__device__ inline double baseline_uniform(const BaselinePolicy& policy, std::uint64_t key, Node batch) {
    if (policy.kind != BaselineKind::Static || policy.restart_mode != StaticRestart::Bernoulli)
        return 0;
    // 与原基线相同的Philox保留槽，不读写其他随机域的状态。
    auto random = random_state(key, batch, UINT32_MAX, 1);
    return uniform53(random);
}

static __global__ void select_baseline_actions(BaselinePolicy policy, const ControllerState* controls,
    const std::uint64_t* keys, Node batch, const std::uint32_t* masks, std::int32_t* actions,
    Node colonies, double* diagnostic_uniforms) {
    const Node colony = blockIdx.x * blockDim.x + threadIdx.x;
    if (colony >= colonies) return;
    const double uniform = baseline_uniform(policy, keys[colony], batch);
    actions[colony] = baseline_action(policy, controls[colony].feedback, batch, uniform, masks[colony]);
    if (diagnostic_uniforms) diagnostic_uniforms[colony] = uniform;
}

static __global__ void restrict_factorial_actions(FactorialPolicy policy, const ControllerState* controls,
    const std::uint64_t* keys, Node batch, std::uint32_t* masks, Node colonies,
    double* diagnostic_uniforms) {
    const Node colony = blockIdx.x * blockDim.x + threadIdx.x;
    if (colony >= colonies) return;
    const double uniform = baseline_uniform(policy.baseline, keys[colony], batch);
    masks[colony] = factorial_mask(policy, controls[colony].feedback, batch, uniform, masks[colony]);
    if (diagnostic_uniforms) diagnostic_uniforms[colony] = uniform;
}

}  // namespace gp_faco::cuda_detail
