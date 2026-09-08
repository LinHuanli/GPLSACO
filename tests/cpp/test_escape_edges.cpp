// 期望取自完整tour边差集；允许父tour保留上批已失效的图外边。
#include "gp_faco/escape_edges.hpp"
#include "gp_faco/sparse_graph.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using namespace gp_faco;
std::uint64_t moves = 0, permitted = 0, rejected = 0, inherited_outside = 0;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
double distance(Node a, Node b) { return a > b ? a - b : b - a; }
std::set<Edge> edges(const std::vector<Node>& tour) {
    std::set<Edge> result;
    for (Node i = 0; i < tour.size(); ++i) result.insert(std::minmax(tour[i], tour[(i + 1) % tour.size()]));
    return result;
}
bool equal(const EscapeCache& a, const EscapeCache& b) {
    if (a.size != b.size || a.epoch != b.epoch) return false;
    for (Node i = 0; i < kEscapeEdgeCapacity; ++i)
        if (a.edges[i].a != b.edges[i].a || a.edges[i].b != b.edges[i].b) return false;
    return true;
}

template<class Prepare, class Predicate>
void verify(const std::vector<Node>& before, const std::vector<Node>& after,
            const SparseUndirectedGraph& graph, Prepare prepare, Predicate predicate) {
    const auto old = edges(before), now = edges(after);
    for (bool seeded : {false, true}) for (bool permit : {false, true}) {
        EscapeCache cache{}; cache.reset();
        if (seeded) {
            for (Node a = 0; a < before.size(); ++a) for (Node b = a + 1; b < before.size(); ++b)
                if (!graph.contains(a, b) && (a + b) % 2 == 0) cache.edges[cache.size++] = {a, b};
        }
        const EscapeCache original = cache;
        std::set<Edge> novel;
        for (const auto& edge : now) if (!old.count(edge) && !graph.contains(edge.first, edge.second) &&
            !cache.contains({edge.first, edge.second})) novel.insert(edge);
        const bool expected = novel.empty() || (permit && cache.size + novel.size() <= kEscapeEdgeCapacity);
        const EscapeEdges allowed{graph.view(), &cache, permit};
        const auto proposal = prepare(allowed);
        check(static_cast<bool>(proposal) == expected && predicate(allowed) == expected,
              "Escape整组预检与完整tour边差集不一致");
        check(equal(original, cache), "未选择的预检修改了缓存");
        auto committed = cache;
        check(commit_escape_edges(committed, proposal) == expected, "事务提交与预检不一致");
        if (expected) {
            check(proposal.count == novel.size() && committed.size == cache.size + novel.size(),
                  "新增边登记数量不符或有重复");
            for (const auto& edge : novel) check(committed.contains({edge.first, edge.second}), "漏登记整组边");
            for (const auto& edge : now) if (old.count(edge) && !graph.contains(edge.first, edge.second) &&
                !cache.contains({edge.first, edge.second})) ++inherited_outside;
            ++permitted;
        } else { check(equal(cache, committed), "被拒绝事务留下部分写入"); ++rejected; }
        ++moves;
    }
}

void exhaustive() {
    for (Node n = 3; n <= 7; ++n) {
        std::vector<Node> order(n); std::iota(order.begin(), order.end(), 0);
        const auto initial = edges(order);
        const SparseUndirectedGraph graph(n, {initial.begin(), initial.end()});
        do {
            const CpuTour before(order, distance);
            for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) if (a != b) {
                auto after = before; after.relocate(a, b);
                verify(order, after.order(), graph,
                    [&](const auto& allowed) { return prepare_escape_relocation(order.data(),
                        before.positions().data(), n, a, b, allowed); },
                    [&](const auto& allowed) { return relocation_allowed(order.data(),
                        before.positions().data(), n, a, b, allowed); });
                for (unsigned kind = 0; kind < 2; ++kind) {
                    const Node an = kind == 0 ? before.successor(a) : before.predecessor(a);
                    const Node bn = kind == 0 ? before.successor(b) : before.predecessor(b);
                    after = before; after.reverse_section(kind == 0 ? an : a, kind == 0 ? bn : b);
                    verify(order, after.order(), graph,
                        [&](const auto& allowed) { return prepare_escape_two_opt(a, an, b, bn, allowed); },
                        [&](const auto& allowed) { return two_opt_allowed(a, an, b, bn, allowed); });
                }
            }
        } while (std::next_permutation(order.begin() + 1, order.end()));
    }
}

void capacity_and_expiry() {
    // 20点共有190条无向边：可填满64条且为事务预留真正未登记边。
    const SparseUndirectedGraph graph(20, {});
    EscapeCache cache{}; cache.reset();
    for (Node a = 0; a < 20 && cache.size < 62; ++a)
        for (Node b = a + 1; b < 20 && cache.size < 62; ++b) cache.edges[cache.size++] = {a, b};
    const MoveEdge removed[3]{{8, 9}, {9, 10}, {10, 11}};
    const MoveEdge added[3]{{12, 13}, {14, 15}, {16, 17}};
    auto proposal = prepare_escape_edges(removed, added, graph.view(), cache, true);
    const auto original = cache;
    check(proposal.status == EscapeProposalStatus::Capacity && !commit_escape_edges(cache, proposal),
          "三个新增边被逐边放行而超过剩余两槽");
    check(equal(cache, original), "容量拒绝写入缓存");
    const MoveEdge repeated[3]{{12, 13}, {13, 12}, {14, 15}};
    proposal = prepare_escape_edges(removed, repeated, graph.view(), cache, true);
    check(proposal.count == 2 && commit_escape_edges(cache, proposal) && cache.size == 64,
          "整组无向去重或恰好容量边界不正确");
    check(!commit_escape_edges(cache, proposal), "长度过期的事务可重复提交");
    const MoveEdge unchanged[3]{{9, 8}, {9, 10}, {10, 11}};
    proposal = prepare_escape_edges(removed, unchanged, graph.view(), cache, false);
    check(proposal && proposal.count == 0, "父tour图外边原样保留被误判");
    cache.reset();
    check(!commit_escape_edges(cache, proposal) && cache.size == 0, "跨批事务没有失效");
    const MoveEdge one_removed[1]{{1, 2}}, one_added[1]{{12, 13}};
    check(prepare_escape_edges(one_removed, one_added, graph.view(), cache, false).status ==
          EscapeProposalStatus::OutsideGraph, "上批例外变成永久白名单");
    const MoveEdge invalid[1]{{20, 3}};
    check(prepare_escape_edges(one_removed, invalid, graph.view(), cache, true).status ==
          EscapeProposalStatus::InvalidEdge, "哨兵或越界端点取得许可");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        exhaustive(); capacity_and_expiry();
        check(permitted && rejected && inherited_outside, "缺少许可/拒绝/失效父边覆盖");
        std::ostringstream report;
        report << "{\n  \"status\": \"passed\",\n  \"move_cases\": " << moves
               << ",\n  \"permitted\": " << permitted << ",\n  \"rejected\": " << rejected
               << ",\n  \"inherited_outside_edge_cases\": " << inherited_outside
               << ",\n  \"capacity\": 64,\n  \"oracle\": \"complete before/after tour edge sets\"\n}\n";
        if (argc > 1) { std::ofstream out(argv[1]); check(static_cast<bool>(out), "无法写报告"); out << report.str(); }
        std::cout << report.str(); return 0;
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
