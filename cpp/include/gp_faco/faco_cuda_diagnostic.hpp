#pragma once

#include "gp_faco/faco_cpu.hpp"

namespace gp_faco {

class SparseUndirectedGraph;

// 仅用于操作级对照：显式距离矩阵不作为完整求解器或10K表示。
struct FacoDiagnosticTask {
    std::vector<Node> tour;
    std::vector<Node> visit_order;  // 首项为起点，其余为显式给定的合法选点序列。
    Node mne_target;
};

struct FacoDiagnosticInfo {
    double construction_cost;
    double final_cost;
    ConstructionStats construction;
    LocalSearchStats local_search;
    Node checklist_size;
};

struct FacoDiagnosticResult {
    std::vector<Node> construction_tour;
    std::vector<Node> tour;
    std::vector<Node> positions;
    std::vector<Node> checklist;
    FacoDiagnosticInfo info;
};

// 同步返回诊断结果；不含roulette/信息素更新/完整Engine或deadline语义。
std::vector<FacoDiagnosticResult> cuda_faco_diagnostic(
    const std::vector<double>& distances, const CandidateRows& ls_candidates,
    const std::vector<FacoDiagnosticTask>& tasks, std::uint64_t evaluation_limit,
    const SparseUndirectedGraph* hard_graph = nullptr);

}  // namespace gp_faco
