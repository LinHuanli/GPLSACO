#pragma once

#include "gp_faco/faco_cpu.hpp"

#ifdef __CUDACC__
#define GPFACO_EDGE_HD __host__ __device__
#else
#define GPFACO_EDGE_HD
#endif

namespace gp_faco {

// 合法性图独立于枚举候选行；CSR无向化后的实际度数不截断到k。
struct SparseGraphView {
    const Node* offsets;
    const Node* neighbors;
    Node dimension;
    GPFACO_EDGE_HD bool operator()(Node a, Node b) const {
        if (a >= dimension || b >= dimension || a == b) return false;
        Node first = offsets[a], last = offsets[a + 1];
        const Node end = last;
        while (first < last) {
            const Node middle = first + (last - first) / 2;
            if (neighbors[middle] < b) first = middle + 1;
            else last = middle;
        }
        return first < end && neighbors[first] == b;
    }
    GPFACO_EDGE_HD SparseGraphView for_ant(Node, Node) const { return *this; }
};

struct UnrestrictedEdges {
    GPFACO_EDGE_HD bool operator()(Node, Node) const { return true; }
    GPFACO_EDGE_HD UnrestrictedEdges for_ant(Node, Node) const { return *this; }
};

struct MoveEdge { Node a, b; };
GPFACO_EDGE_HD inline bool same_edge(MoveEdge x, MoveEdge y) {
    return (x.a == y.a && x.b == y.b) || (x.a == y.b && x.b == y.a);
}

template<unsigned Count, class Allowed>
GPFACO_EDGE_HD bool new_edges_allowed(const MoveEdge (&removed)[Count],
                                      const MoveEdge (&added)[Count], const Allowed& allowed) {
    // 多重集合逐项消去：相邻/环边可能被删除后原样补回，不能当作真正新增边。
    bool consumed[Count]{};
    for (unsigned i = 0; i < Count; ++i) {
        bool existing = false;
        for (unsigned j = 0; j < Count; ++j) {
            if (!consumed[j] && same_edge(added[i], removed[j])) {
                consumed[j] = true; existing = true; break;
            }
        }
        if (!existing && !allowed(added[i].a, added[i].b)) return false;
    }
    return true;
}

template<class Allowed>
GPFACO_EDGE_HD bool relocation_allowed(const Node* tour, const Node* positions, Node n,
                                      Node target, Node node, const Allowed& allowed) {
    if (target >= n || node >= n || target == node) return false;
    const Node target_after = tour[(positions[target] + 1) % n];
    if (target_after == node) return true;
    const Node before = tour[(positions[node] + n - 1) % n];
    const Node after = tour[(positions[node] + 1) % n];
    const MoveEdge removed[3]{{before, node}, {node, after}, {target, target_after}};
    const MoveEdge added[3]{{before, after}, {target, node}, {node, target_after}};
    return new_edges_allowed(removed, added, allowed);
}

template<class Allowed>
GPFACO_EDGE_HD bool two_opt_allowed(Node a, Node a_neighbor, Node b, Node b_neighbor,
                                   const Allowed& allowed) {
    const MoveEdge removed[2]{{a, a_neighbor}, {b, b_neighbor}};
    const MoveEdge added[2]{{a, b}, {a_neighbor, b_neighbor}};
    return new_edges_allowed(removed, added, allowed);
}

}  // namespace gp_faco

#undef GPFACO_EDGE_HD
