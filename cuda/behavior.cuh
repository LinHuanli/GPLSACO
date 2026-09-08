// 全部输入均为const；额外内存随蚂蚁/colony数增长，不随tour维数增长。
#pragma once

#include "gp_faco/behavior.hpp"
#include "baseline_policy.cuh"
#include "colony_state.cuh"
#include "edge_observation.cuh"

namespace gp_faco::cuda_detail {

static __global__ void begin_behavior(Node n, Node ants, Node colonies, Node batch,
    const ControllerState* controls, const Colony* states, const Node* alternatives,
    const std::uint32_t* legal_masks, const std::uint64_t* keys,
    bool has_baseline, BaselinePolicy baseline, BatchBehaviorRow* rows) {
    const Node colony = blockIdx.x * blockDim.x + threadIdx.x;
    if (colony >= colonies) return;
    auto& row = rows[colony]; row = {};
    row.batch = batch; row.colony = colony; row.ants = ants; row.dimension = n;
    row.alternative = alternatives[colony]; row.legal_mask = legal_masks[colony];
    row.feedback_before = controls[colony].feedback;
    row.global_before = states[colony].global_cost;
    row.reference_before = states[colony].parent_cost;
    if (has_baseline) row.baseline_requested_action = baseline_action(baseline,
        controls[colony].feedback, batch, baseline_uniform(baseline, keys[colony], batch), UINT32_MAX);
}

static __global__ void observe_behavior_action(Node colonies, const ControllerState* controls,
    const Colony* states, const std::int32_t* actions, const std::uint32_t* masks,
    BatchBehaviorRow* rows) {
    const Node colony = blockIdx.x * blockDim.x + threadIdx.x;
    if (colony >= colonies) return;
    auto& row = rows[colony]; row.action = actions[colony]; row.action_mask = masks[colony];
    row.reference_used = row.action < 16 ? states[colony].parent_cost :
        controls[colony].archive_cost[row.alternative];
}

static __global__ void observe_terminal_edges(Node n, const Node* tours,
    const Node* parent_positions, Node* new_edges) {
    const auto base = static_cast<std::size_t>(blockIdx.x) * n;
    const Node count = observed_new_edges(tours + base, parent_positions + base, n);
    if (threadIdx.x == 0) new_edges[blockIdx.x] = count;
}

static __global__ void finish_behavior(Node ants, Node colonies, const ControllerState* controls,
    const Colony* states, const FacoDiagnosticInfo* all_info, const TourFingerprint* identities,
    const Node* construction_new_edges, const Node* final_new_edges, BatchBehaviorRow* rows) {
    const Node colony = blockIdx.x * blockDim.x + threadIdx.x;
    if (colony >= colonies) return;
    auto& row = rows[colony];
    row.global_after = states[colony].global_cost;
    row.iteration_best_cost = all_info[colony * ants + states[colony].iteration_best].final_cost;
    for (Node ant = colony * ants; ant < (colony + 1) * ants; ++ant) {
        const auto& info = all_info[ant];
        row.construction_mne += info.construction.mne;
        row.construction_steps += info.construction.steps;
        row.construction_relocations += info.construction.nonidentity_relocations;
        row.construction_exhausted_ants += info.construction.legal_exhausted;
        row.construction_new_edges += construction_new_edges[ant];
        row.final_new_edges += final_new_edges[ant];
        row.exact_returns += final_new_edges[ant] == 0;
        // 原q反馈仍按原双指纹近似；强制碰撞诊断时也不改其输入。
        row.fingerprint_returns += identities[ant] == controls[colony].active_identity;
        row.ls_move_evaluations += info.local_search.move_evaluations;
        row.ls_accepted_moves += info.local_search.accepted_moves;
        row.ls_limit_reached_ants += info.local_search.evaluation_limit_reached;
    }
}

}  // namespace gp_faco::cuda_detail
