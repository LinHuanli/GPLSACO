#pragma once

#include "gp_faco/edge_constraints.hpp"

#ifdef __CUDACC__
#define GPFACO_ESCAPE_HD __host__ __device__
#else
#define GPFACO_ESCAPE_HD
#endif

namespace gp_faco {

// Escape spec 1：当前蚂蚁/当前批次的有限许可，不从父tour继承白名单。
inline constexpr Node kEscapeEdgeCapacity = 64;
inline constexpr Node kEscapePrimaryWidth = 16;
inline constexpr Node kEscapeBackupWidth = 64;
inline constexpr Node kEscapeLsWidth = 20;

struct EscapeCache {
    MoveEdge edges[kEscapeEdgeCapacity];
    Node size = 0;
    std::uint64_t epoch = 0;

    GPFACO_ESCAPE_HD void reset() { size = 0; ++epoch; }
    GPFACO_ESCAPE_HD bool contains(MoveEdge edge) const {
        for (Node i = 0; i < size; ++i) if (same_edge(edges[i], edge)) return true;
        return false;
    }
};

enum class EscapeProposalStatus : std::uint8_t { Allowed, InvalidEdge, OutsideGraph, Capacity };

struct EscapeProposal {
    MoveEdge edges[3]{};
    Node count = 0, expected_size = 0;
    std::uint64_t expected_epoch = 0;
    EscapeProposalStatus status = EscapeProposalStatus::Allowed;
    GPFACO_ESCAPE_HD explicit operator bool() const { return status == EscapeProposalStatus::Allowed; }
};

GPFACO_ESCAPE_HD inline MoveEdge canonical_edge(MoveEdge edge) {
    return edge.a < edge.b ? edge : MoveEdge{edge.b, edge.a};
}

template<unsigned Count>
GPFACO_ESCAPE_HD EscapeProposal prepare_escape_edges(
    const MoveEdge (&removed)[Count], const MoveEdge (&added)[Count],
    SparseGraphView graph, const EscapeCache& cache, bool permit_novel) {
    static_assert(Count <= 3, "Escape事务只用于重定位/2-opt的完整新增边组");
    EscapeProposal proposal;
    proposal.expected_size = cache.size;
    proposal.expected_epoch = cache.epoch;
    bool consumed[Count]{};
    for (unsigned i = 0; i < Count; ++i) {
        const MoveEdge edge = canonical_edge(added[i]);
        if (edge.a >= graph.dimension || edge.b >= graph.dimension || edge.a == edge.b) {
            proposal.status = EscapeProposalStatus::InvalidEdge; return proposal;
        }
        bool unchanged = false;
        for (unsigned j = 0; j < Count; ++j) if (!consumed[j] && same_edge(edge, removed[j])) {
            consumed[j] = true; unchanged = true; break;
        }
        if (unchanged || graph(edge.a, edge.b) || cache.contains(edge)) continue;
        if (!permit_novel) { proposal.status = EscapeProposalStatus::OutsideGraph; return proposal; }
        bool duplicate = false;
        for (Node j = 0; j < proposal.count; ++j) duplicate |= same_edge(edge, proposal.edges[j]);
        if (!duplicate) proposal.edges[proposal.count++] = edge;
    }
    if (cache.size > kEscapeEdgeCapacity || proposal.count > kEscapeEdgeCapacity - cache.size)
        proposal.status = EscapeProposalStatus::Capacity;
    return proposal;
}

GPFACO_ESCAPE_HD inline bool commit_escape_edges(EscapeCache& cache, const EscapeProposal& proposal) {
    // 只允许选择线程提交；版本/长度不符说明检查之后已有移动或发生批次失效。
    // 拒绝整组，绝不先写一部分边再发现容量不足。
    if (!proposal || cache.epoch != proposal.expected_epoch || cache.size != proposal.expected_size ||
        cache.size > kEscapeEdgeCapacity || proposal.count > kEscapeEdgeCapacity - cache.size) return false;
    for (Node i = 0; i < proposal.count; ++i) cache.edges[cache.size + i] = proposal.edges[i];
    cache.size += proposal.count;
    return true;
}

struct EscapeEdges {
    SparseGraphView graph;
    EscapeCache* cache;
    bool permit_novel = false;
    GPFACO_ESCAPE_HD EscapeEdges for_ant(Node, Node) const { return *this; }
    GPFACO_ESCAPE_HD EscapeEdges with_permission(bool value) const { return {graph, cache, value}; }
};

// ADL使共同移动函数对Escape执行整组预检，而非三个彼此独立的容量判断。
template<unsigned Count>
GPFACO_ESCAPE_HD bool new_edges_allowed(const MoveEdge (&removed)[Count],
                                        const MoveEdge (&added)[Count], const EscapeEdges& allowed) {
    return static_cast<bool>(prepare_escape_edges(removed, added, allowed.graph,
                                                  *allowed.cache, allowed.permit_novel));
}

GPFACO_ESCAPE_HD inline EscapeProposal prepare_escape_relocation(
    const Node* tour, const Node* positions, Node n, Node target, Node node, const EscapeEdges& allowed) {
    EscapeProposal proposal;
    proposal.expected_size = allowed.cache->size;
    proposal.expected_epoch = allowed.cache->epoch;
    if (target >= n || node >= n || target == node) {
        proposal.status = EscapeProposalStatus::InvalidEdge; return proposal;
    }
    const Node target_after = tour[(positions[target] + 1) % n];
    if (target_after == node) return proposal;
    const Node before = tour[(positions[node] + n - 1) % n];
    const Node after = tour[(positions[node] + 1) % n];
    const MoveEdge removed[3]{{before, node}, {node, after}, {target, target_after}};
    const MoveEdge added[3]{{before, after}, {target, node}, {node, target_after}};
    return prepare_escape_edges(removed, added, allowed.graph, *allowed.cache, allowed.permit_novel);
}

GPFACO_ESCAPE_HD inline EscapeProposal prepare_escape_two_opt(
    Node a, Node a_neighbor, Node b, Node b_neighbor, const EscapeEdges& allowed) {
    const MoveEdge removed[2]{{a, a_neighbor}, {b, b_neighbor}};
    const MoveEdge added[2]{{a, b}, {a_neighbor, b_neighbor}};
    return prepare_escape_edges(removed, added, allowed.graph, *allowed.cache, allowed.permit_novel);
}

}  // namespace gp_faco

#undef GPFACO_ESCAPE_HD
