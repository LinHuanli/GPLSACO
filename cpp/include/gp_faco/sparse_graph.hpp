#pragma once

#include "gp_faco/edge_constraints.hpp"

namespace gp_faco {

class SparseUndirectedGraph {
public:
    SparseUndirectedGraph(Node dimension, const std::vector<Edge>& edges);
    static SparseUndirectedGraph from_candidates(const CandidateRows& candidates,
                                                  const std::vector<Node>& initial_tour);
    SparseGraphView view() const { return {offsets_.data(), neighbors_.data(), dimension_}; }
    bool contains(Node a, Node b) const { return view()(a, b); }
    bool contains_tour(const std::vector<Node>& tour) const;
    Node size() const { return dimension_; }
    std::size_t edges() const { return neighbors_.size() / 2; }
    Node maximum_degree() const;
    const std::vector<Node>& offsets() const { return offsets_; }
    const std::vector<Node>& neighbors() const { return neighbors_; }
private:
    Node dimension_;
    std::vector<Node> offsets_, neighbors_;
};

}  // namespace gp_faco
