// 基线只替换动作选择，共享特征/区域、档案、完整重启与构造/LS内核。
#pragma once

#include "gp_faco/baseline_policy.hpp"
#include "faco_choices.cuh"

namespace gp_faco::cuda_detail {

static __global__ void select_baseline_actions(BaselinePolicy policy, const ControllerState* controls,
    const std::uint64_t* keys, Node batch, const std::uint32_t* masks, std::int32_t* actions,
    Node colonies, double* diagnostic_uniforms) {
    const Node colony = blockIdx.x * blockDim.x + threadIdx.x;
    if (colony >= colonies) return;
    double uniform = 0;
    if (policy.kind == BaselineKind::Static && policy.restart_mode == StaticRestart::Bernoulli) {
        // ant=UINT32_MAX的step 0保留给来源选择；step 1属于静态Bernoulli重启。
        auto random = random_state(keys[colony], batch, UINT32_MAX, 1);
        uniform = uniform53(random);
    }
    actions[colony] = baseline_action(policy, controls[colony].feedback, batch, uniform, masks[colony]);
    if (diagnostic_uniforms) diagnostic_uniforms[colony] = uniform;
}

}  // namespace gp_faco::cuda_detail
