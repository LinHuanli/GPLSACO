#pragma once

#include "gp_faco/control_ops.hpp"

namespace gp_faco {

// 只读观测；公共序列化逐字段输出，不暴露结构体padding或原始设备内存。
struct BatchBehaviorRow {
    Node batch = 0, colony = 0, ants = 0, dimension = 0, alternative = archive_capacity;
    std::uint32_t legal_mask = 0, action_mask = 0;
    std::int32_t action = 0, baseline_requested_action = -1;
    double global_before = 0, reference_before = 0, reference_used = 0;
    double global_after = 0, iteration_best_cost = 0;
    FeedbackState feedback_before, feedback_after;
    std::uint64_t construction_mne = 0, construction_steps = 0, construction_relocations = 0;
    std::uint64_t construction_exhausted_ants = 0, construction_new_edges = 0, final_new_edges = 0;
    std::uint64_t exact_returns = 0, fingerprint_returns = 0;
    std::uint64_t ls_move_evaluations = 0, ls_accepted_moves = 0, ls_limit_reached_ants = 0;
};

}  // namespace gp_faco
