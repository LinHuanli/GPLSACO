// 用实际移动前后完整边集做oracle，不用待测三/两边公式推导期望结果。
#include "gp_faco/sparse_graph.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using gp_faco::Node;
using gp_faco::Edge;
std::uint64_t relocations = 0, flips = 0, rejected = 0, predicate_checks = 0;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
double distance(Node a, Node b) { return a > b ? a - b : b - a; }
std::set<Edge> edges(const std::vector<Node>& tour) {
    std::set<Edge> result;
    for (Node i = 0; i < tour.size(); ++i)
        result.insert(std::minmax(tour[i], tour[(i + 1) % tour.size()]));
    return result;
}

template<class Predicate>
void compare_full_graph(const std::vector<Node>& before, const std::vector<Node>& after,
                        const Predicate& predicate) {
    const auto old = edges(before), now = edges(after);
    std::set<Edge> combined(old);
    combined.insert(now.begin(), now.end());
    const auto compare = [&](const std::set<Edge>& members) {
        gp_faco::SparseUndirectedGraph graph(before.size(), {members.begin(), members.end()});
        check(graph.contains_tour(before), "oracle父tour不合法");
        const bool expected = graph.contains_tour(after);
        check(predicate(graph.view()) == expected, "局部公式与完整tour边集不一致");
        ++predicate_checks; rejected += !expected;
    };
    compare(old); compare(combined);
    for (const Edge& edge : now) if (!old.count(edge)) {
        auto missing = combined; missing.erase(edge); compare(missing);
    }
}

void exhaustive() {
    for (Node n = 3; n <= 7; ++n) {
        std::vector<Node> order(n); std::iota(order.begin(), order.end(), 0);
        // 固定城市0的位置去除循环旋转重复；保留两种方向和全部环边端点。
        do {
            const gp_faco::CpuTour before(order, distance);
            for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) if (a != b) {
                auto after = before; after.relocate(a, b);
                compare_full_graph(order, after.order(), [&](const auto& allowed) {
                    return gp_faco::relocation_allowed(order.data(), before.positions().data(), n,
                                                       a, b, allowed);
                });
                ++relocations;
                for (unsigned kind = 0; kind < 2; ++kind) {
                    const Node an = kind == 0 ? before.successor(a) : before.predecessor(a);
                    const Node bn = kind == 0 ? before.successor(b) : before.predecessor(b);
                    after = before;
                    after.reverse_section(kind == 0 ? an : a, kind == 0 ? bn : b);
                    compare_full_graph(order, after.order(), [&](const auto& allowed) {
                        return gp_faco::two_opt_allowed(a, an, b, bn, allowed);
                    });
                    ++flips;
                }
            }
        } while (std::next_permutation(order.begin() + 1, order.end()));
    }
}

void regressions() {
    const std::vector<Node> order{0, 1, 2, 3, 4, 5};
    gp_faco::CpuTour tour(order, distance);
    auto members = edges(order); members.emplace(0, 3);
    gp_faco::SparseUndirectedGraph graph(6, {members.begin(), members.end()});
    check(graph.contains(0, 3), "应覆盖已枚举候选边合法的反例");
    check(!gp_faco::relocation_allowed(order.data(), tour.positions().data(), 6, 0, 3, graph.view()),
          "只检查选中的边而漏掉补边/插入另一边");
    check(!gp_faco::two_opt_allowed(0, 1, 3, 4, graph.view()), "2-opt漏掉另一条新增边");
    const auto cycle = gp_faco::SparseUndirectedGraph::from_candidates(gp_faco::CandidateRows(6), order);
    std::vector<std::uint8_t> visited{1, 1, 0, 0, 0, 0};
    const auto allowed = [&](Node node) {
        return gp_faco::relocation_allowed(order.data(), tour.positions().data(), 6, 0, node, cycle.view());
    };
    const auto selection = gp_faco::select_next(0, {2, 3}, {1, 1}, {4, 5}, visited, distance, 0.5, allowed);
    check(selection.stage == gp_faco::SelectionStage::Exhausted, "Hard全来源耗尽仍悄悄出图");

    gp_faco::CandidateRows star(6);
    star[0] = {1}; for (Node a = 1; a < 6; ++a) star[a] = {0};
    const auto undirected = gp_faco::SparseUndirectedGraph::from_candidates(star, order);
    check(undirected.maximum_degree() == 5 && undirected.edges() == 9,
          "无向化静默截断了度数或错误计算边数");
    for (Node a = 0; a < 6; ++a) for (Node b = 0; b < 6; ++b)
        check(undirected.contains(a, b) == undirected.contains(b, a), "图不对称");
    check(!undirected.contains_tour({0, 1, 2, 3, 4, 4}) && !undirected.contains(0, 6), "图输入边界未拒绝");
    bool invalid = false;
    try { gp_faco::SparseUndirectedGraph(6, {{0, 0}}); }
    catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "图自环未拒绝");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        exhaustive(); regressions();
        check(rejected > 0 && predicate_checks > relocations + flips, "未覆盖新增边逐项拒绝");
        std::ostringstream out;
        out << "{\n  \"status\": \"passed\",\n  \"relocations\": " << relocations
            << ",\n  \"two_opt_moves\": " << flips << ",\n  \"predicate_graph_checks\": " << predicate_checks
            << ",\n  \"rejected_moves\": " << rejected
            << ",\n  \"oracle\": \"complete before/after tour edge sets\"\n}\n";
        if (argc > 1) { std::ofstream file(argv[1]); if (!file) throw std::runtime_error("无法写报告"); file << out.str(); }
        std::cout << out.str(); return 0;
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
