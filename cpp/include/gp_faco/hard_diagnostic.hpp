#pragma once

#include "gp_faco/sparse_graph.hpp"

namespace gp_faco {

struct HardSelectionTask {
    std::vector<Node> tour;
    std::vector<std::uint8_t> visited;
    Node current;
    std::uint64_t seed;
};
struct HardSelectionResult {
    Node selected;  // dimension表示全部声明来源都无合法移动。
    double uniform;
};

// 仅用于真实Philox选点的CPU重放，矩阵规模限制与操作诊断一致。
std::vector<HardSelectionResult> cuda_hard_selection_diagnostic(
    const std::vector<double>& distances, const CandidateRows& primary,
    const CandidateRows& backup, const std::vector<double>& products,
    const SparseUndirectedGraph& graph, const std::vector<HardSelectionTask>& tasks);

}  // namespace gp_faco
