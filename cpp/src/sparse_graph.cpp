// 无向合法性图；保留真实成员和度数，枚举/距离排序在另一层处理。
#include "gp_faco/sparse_graph.hpp"

#include <algorithm>
#include <stdexcept>

namespace gp_faco {

SparseUndirectedGraph::SparseUndirectedGraph(Node dimension, const std::vector<Edge>& edges)
    : dimension_(dimension) {
    if (dimension < 3 || dimension > 10000) throw std::invalid_argument("图规模必须在3..10000");
    CandidateRows rows(dimension);
    for (const auto& [a, b] : edges) {
        if (a >= dimension || b >= dimension || a == b) throw std::invalid_argument("图边越界或自环");
        rows[a].push_back(b); rows[b].push_back(a);
    }
    offsets_.push_back(0);
    for (auto& row : rows) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        neighbors_.insert(neighbors_.end(), row.begin(), row.end());
        offsets_.push_back(static_cast<Node>(neighbors_.size()));
    }
}

SparseUndirectedGraph SparseUndirectedGraph::from_candidates(
    const CandidateRows& candidates, const std::vector<Node>& initial_tour) {
    const Node n = candidates.size();
    if (initial_tour.size() != n) throw std::invalid_argument("图与初始tour维数不符");
    auto sorted = initial_tour;
    std::sort(sorted.begin(), sorted.end());
    for (Node i = 0; i < n; ++i) if (sorted[i] != i) throw std::invalid_argument("初始tour不是排列");
    std::vector<Edge> edges;
    for (Node a = 0; a < n; ++a) for (Node b : candidates[a]) edges.emplace_back(a, b);
    for (Node i = 0; i < n; ++i) edges.emplace_back(initial_tour[i], initial_tour[(i + 1) % n]);
    return SparseUndirectedGraph(n, edges);
}

bool SparseUndirectedGraph::contains_tour(const std::vector<Node>& tour) const {
    if (tour.size() != dimension_) return false;
    std::vector<bool> seen(dimension_, false);
    for (Node i = 0; i < dimension_; ++i) {
        const Node node = tour[i];
        if (node >= dimension_ || seen[node] || !contains(node, tour[(i + 1) % dimension_])) return false;
        seen[node] = true;
    }
    return true;
}

Node SparseUndirectedGraph::maximum_degree() const {
    Node degree = 0;
    for (Node a = 0; a < dimension_; ++a) degree = std::max(degree, offsets_[a + 1] - offsets_[a]);
    return degree;
}

}  // namespace gp_faco
