#pragma once

#include "gp_faco/faco_cuda_diagnostic.hpp"

#include <memory>

namespace gp_faco {

// 固定迭代开发求解器；预算/GP/档案事务由后续Engine层接入。
struct FixedFacoSettings {
    Node ants = 0;  // 0 按 2022 论文随规模确定，正数仅供显式开发实验。
    Node primary_width = 16;
    Node backup_width = 64;
    Node ls_width = 20;
    double beta = 1;
    double retention = 0.5;
    double p_best = 0.1;
    double epoch_source_probability = 0.01;
    std::uint64_t ls_evaluation_limit = 100000;
    std::uint64_t initial_ls_evaluation_limit = 100000;
};

struct FixedFacoBatchTrace {
    std::vector<Node> parent_before, parent_after, epoch_best, global_best;
    std::vector<Node> selected_nodes;
    std::vector<double> selection_uniforms;
    std::vector<double> trails_before, trails_after, products_before;
    std::vector<FacoDiagnosticResult> ants;
    double default_before, default_after, epoch_cost, global_cost;
    double minimum, maximum, source_uniform;
    Node iteration_best;
    bool source_is_epoch;
};

struct FixedFacoResult {
    std::vector<Node> tour;
    double cost, initial_cost, preparation_seconds, solve_seconds;
    std::uint64_t batches, construction_steps, ls_evaluations;
    std::size_t allocated_device_bytes;
    std::vector<FixedFacoBatchTrace> trace;
};

class FixedFacoGpu {
public:
    FixedFacoGpu(std::vector<double> coordinates, std::uint64_t instance_key,
                 FixedFacoSettings settings = {});
    ~FixedFacoGpu();
    FixedFacoGpu(const FixedFacoGpu&) = delete;
    FixedFacoGpu& operator=(const FixedFacoGpu&) = delete;
    // 一次调用完成全部批次；每次调用重置动态状态，不复用旧ant/信息素。
    FixedFacoResult run_iterations(std::uint64_t seed, Node batches, Node mne_target,
                                  bool capture_trace = false);
    const CandidateRows& primary_candidates() const;
    const CandidateRows& backup_candidates() const;
    const CandidateRows& ls_candidates() const;
    const std::vector<Node>& initial_tour() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gp_faco
