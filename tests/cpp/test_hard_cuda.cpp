// Hard操作CPU/CUDA差异、完整图不变量与真实Philox过滤选择。
#include "gp_faco/faco_cuda_diagnostic.hpp"
#include "gp_faco/hard_diagnostic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using gp_faco::Node;
std::uint64_t cases = 0, moves = 0, rejections = 0, exhausted = 0, selections = 0;
std::array<std::uint64_t, 5> stages{};
double maximum_error = 0;
std::string context;
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(std::string(message) + " " + context);
}
void near(double a, double b) {
    maximum_error = std::max(maximum_error, std::abs(a - b));
    check(std::isfinite(a) && std::abs(a - b) <= 1e-8, "成本不同");
}

void check_choices(const std::vector<double>& matrix, const gp_faco::CandidateRows& rows,
    const gp_faco::SparseUndirectedGraph& graph, const std::vector<Node>& order, Node count) {
    const Node n = order.size(), width = rows[0].size();
    auto distance = [&](Node a, Node b) { return matrix[a * n + b]; };
    const gp_faco::CpuTour tour(order, distance);
    gp_faco::CandidateRows backup(n);
    const Node backup_width = std::min<Node>(8, n - 1 - width);
    for (Node a = 0; a < n; ++a) {
        for (Node b = 0; b < n && backup[a].size() < backup_width; ++b)
            if (a != b && std::find(rows[a].begin(), rows[a].end(), b) == rows[a].end()) backup[a].push_back(b);
    }
    std::mt19937_64 random(94127 + n);
    std::vector<gp_faco::HardSelectionTask> tasks;
    for (Node i = 0; i < count; ++i) {
        const Node current = i % n;
        std::vector<std::uint8_t> visited(n);
        for (auto& v : visited) v = random() % 4 == 0;
        visited[current] = 1;
        if (i % 4 == 0) visited[tour.successor(current)] = 1;
        tasks.push_back({order, std::move(visited), current, random()});
    }
    for (bool zero : {false, true}) {
        std::vector<double> products(n * width);
        for (Node i = 0; i < n * width; ++i) products[i] = zero ? 0 : (i % 5 ? 0.25 * (i % 11 + 1) : 0);
        const auto gpu = gp_faco::cuda_hard_selection_diagnostic(matrix, rows, backup, products, graph, tasks);
        for (Node i = 0; i < count; ++i) {
            const auto& task = tasks[i];
            const auto allowed = [&](Node node) {
                return gp_faco::relocation_allowed(order.data(), tour.positions().data(), n,
                                                   task.current, node, graph.view());
            };
            const std::vector<double> weights(products.begin() + task.current * width,
                                               products.begin() + (task.current + 1) * width);
            const auto cpu = gp_faco::select_next(task.current, rows[task.current], weights,
                backup[task.current], task.visited, distance, gpu[i].uniform, allowed);
            const Node selected = cpu.stage == gp_faco::SelectionStage::Exhausted ? n : cpu.node;
            check(gpu[i].selected == selected, "Hard真实随机选择不符");
            ++selections; ++stages[static_cast<unsigned>(cpu.stage)];
        }
    }
}

void run(Node n, unsigned metric, std::uint64_t cap, Node ants, bool small) {
    context = "n=" + std::to_string(n) + " metric=" + std::to_string(metric) + " cap=" + std::to_string(cap);
    std::mt19937_64 random(7193 + n + metric * 431);
    std::uniform_real_distribution<double> unit(0, 1);
    std::vector<std::pair<double, double>> points(n);
    for (auto& point : points) point = {unit(random), unit(random)};
    if (metric == 2) for (Node i = 3; i < n; i += 5) points[i] = points[0];
    std::vector<double> matrix(n * n);
    for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) {
        const double value = std::hypot(points[a].first - points[b].first, points[a].second - points[b].second);
        matrix[a * n + b] = metric == 1 ? std::floor(value * 10000 + 0.5) : value;
    }
    const auto distance = [&](Node a, Node b) { return matrix[a * n + b]; };
    gp_faco::CandidateRows rows(n);
    const Node width = std::min<Node>(16, n - 1);
    std::vector<gp_faco::Edge> graph_edges;
    for (Node a = 0; a < n; ++a) {
        for (Node b = 0; b < n; ++b) if (a != b) rows[a].push_back(b);
        std::stable_sort(rows[a].begin(), rows[a].end(), [&](Node b, Node c) { return distance(a, b) < distance(a, c); });
        rows[a].resize(width);
        for (Node b : rows[a]) graph_edges.emplace_back(a, b);
        // 成员先验故意乱序，CPU/GPU必须各自使用稳定的真实距离LS视图。
        std::reverse(rows[a].begin(), rows[a].end());
    }
    std::vector<Node> order(n); std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), random);
    for (Node i = 0; i < n; ++i) graph_edges.emplace_back(order[i], order[(i + 1) % n]);
    std::vector<gp_faco::FacoDiagnosticTask> tasks;
    for (Node ant = 0; ant < ants; ++ant) {
        auto source = order;
        std::rotate(source.begin(), source.begin() + ant % n, source.end());
        if (ant % 2) std::reverse(source.begin(), source.end());
        std::vector<Node> visits(n); std::iota(visits.begin(), visits.end(), 0);
        std::shuffle(visits.begin(), visits.end(), random);
        gp_faco::CpuTour probe(source, distance);
        // 人工图支持不同长度前缀，确保既执行真正的非恒等移动又覆盖合法性耗尽。
        const Node length = std::min<Node>(n - 1, std::array<Node, 4>{0, 2, 8, 16}[ant % 4]);
        for (Node j = 1; j <= length; ++j) {
            probe.relocate(visits[j - 1], visits[j]);
            for (Node i = 0; i < n; ++i) graph_edges.emplace_back(probe.order()[i], probe.order()[(i + 1) % n]);
        }
        tasks.push_back({std::move(source), std::move(visits), std::array<Node, 4>{2, 4, 8, 16}[ant % 4]});
    }
    const gp_faco::SparseUndirectedGraph graph(n, graph_edges);
    const gp_faco::DistanceOrderedCandidates candidates(rows, distance);
    const auto gpu = gp_faco::cuda_faco_diagnostic(matrix, rows, tasks, cap, &graph);
    for (Node ant = 0; ant < ants; ++ant) {
        const auto& task = tasks[ant]; const auto& result = gpu[ant];
        gp_faco::FocusedConstruction cpu(gp_faco::CpuTour(task.tour, distance), task.visit_order[0], task.mne_target);
        bool stopped = false;
        while (!cpu.done()) {
            const Node selected = task.visit_order[cpu.stats().steps + 1];
            if (!gp_faco::relocation_allowed(cpu.tour().order().data(), cpu.tour().positions().data(),
                                             n, cpu.current(), selected, graph.view())) { stopped = true; break; }
            cpu.step(selected);
            check(graph.contains_tour(cpu.tour().order()), "CPU构造实际出图");
        }
        check(result.construction_tour == cpu.tour().order(), "Hard构造tour不同");
        const auto& construction = result.info.construction;
        check(construction.mne == cpu.stats().mne && construction.steps == cpu.stats().steps &&
              construction.nonidentity_relocations == cpu.stats().nonidentity_relocations &&
              construction.legal_exhausted == stopped, "Hard构造计数/耗尽不同");
        near(result.info.construction_cost, cpu.tour().cost());
        const auto ls = cpu.tour().checklist_two_opt(candidates, cpu.checklist(), cap, graph.view());
        const auto& observed = result.info.local_search;
        check(result.tour == cpu.tour().order() && result.positions == cpu.tour().positions() &&
              result.checklist == cpu.checklist(), "Hard LS路线/逆映射/pending不同");
        check(observed.processed_nodes == ls.processed_nodes && observed.candidate_checks == ls.candidate_checks &&
              observed.move_evaluations == ls.move_evaluations && observed.accepted_moves == ls.accepted_moves &&
              observed.reactivations == ls.reactivations && observed.constraint_rejections == ls.constraint_rejections &&
              observed.evaluation_limit_reached == ls.evaluation_limit_reached, "Hard LS工作量不同");
        check(graph.contains_tour(result.construction_tour) && graph.contains_tour(result.tour), "GPU结果实际出图");
        near(result.info.final_cost, cpu.tour().cost());
        near(result.info.final_cost, gp_faco::CpuTour(result.tour, distance).cost());
        ++cases; moves += ls.accepted_moves; rejections += ls.constraint_rejections; exhausted += stopped;
    }
    if (cap == 0) check_choices(matrix, rows, graph, order, small ? 32 : 128);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool small = argc == 3 && std::string(argv[2]) == "--small";
        for (Node n : {3u, 4u, 5u, 7u, 17u, 100u, 500u, 1000u})
            for (unsigned metric = 0; metric < 3; ++metric)
                for (std::uint64_t cap : {std::uint64_t{0}, std::uint64_t{7}, std::uint64_t{100000}})
                    run(n, metric, cap, small ? 4 : 8, small);
        check(moves > 0 && rejections > 0 && exhausted > 0, "未覆盖真实LS/拒绝/耗尽");
        for (auto count : stages) check(count > 0, "随机选点遗漏分支");
        std::ostringstream out; out.precision(17);
        out << "{\n  \"status\": \"passed\",\n  \"operation_cases\": " << cases
            << ",\n  \"ls_accepted\": " << moves << ",\n  \"ls_constraint_rejections\": " << rejections
            << ",\n  \"construction_exhausted\": " << exhausted
            << ",\n  \"stochastic_choices\": " << selections << ",\n  \"selection_stages\": [";
        for (unsigned i = 0; i < stages.size(); ++i) out << (i ? ", " : "") << stages[i];
        out << "],\n  \"maximum_absolute_cost_error\": " << maximum_error
            << ",\n  \"scope\": \"Hard operation diagnostic; Engine/Escape/candidate exporters pending\"\n}\n";
        if (argc > 1) { std::ofstream file(argv[1]); if (!file) throw std::runtime_error("无法写报告"); file << out.str(); }
        std::cout << out.str(); return 0;
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
