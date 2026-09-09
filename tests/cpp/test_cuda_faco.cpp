// 人工实例上的CPU/CUDA构造与LS差异测试；不接触训练/验证/测试标签。
#include "gp_faco/faco_cuda_diagnostic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using gp_faco::Node;
std::string context;
double maximum_error = 0;
std::uint64_t cases = 0, transitions = 0, evaluations = 0, reactivations = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string(message) + " " + context);
}

void compare_cost(double actual, double expected) {
    const double error = std::abs(actual - expected);
    maximum_error = std::max(maximum_error, error);
    require(std::isfinite(actual) && error <= 1e-8, "成本不符");
}

void run(Node n, int metric, std::uint64_t cap, Node ants) {
    std::mt19937_64 random(83741 + n + metric * 1117);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<std::pair<double, double>> points(n);
    for (auto& p : points) p = {uniform(random), uniform(random)};
    if (metric == 2) for (Node i = 3; i < n; i += 3) points[i] = points[0];
    std::vector<double> matrix(static_cast<std::size_t>(n) * n);
    for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) {
        const double d = std::hypot(points[a].first - points[b].first,
                                    points[a].second - points[b].second);
        matrix[a * n + b] = metric == 1 ? std::floor(d * 10000 + 0.5) : d;
    }
    auto distance = [&](Node a, Node b) { return matrix[a * n + b]; };
    const Node width = std::min<Node>(20, n - 1);
    gp_faco::CandidateRows rows(n);
    for (Node a = 0; a < n; ++a) {
        for (Node b = 0; b < n; ++b) if (a != b) rows[a].push_back(b);
        std::stable_sort(rows[a].begin(), rows[a].end(), [&](Node b, Node c) {
            return distance(a, b) < distance(a, c);
        });
        rows[a].resize(width);
        // 模拟先验成员乱序，两个后端均应建立真实距离的LS视图。
        if (metric == 0) std::reverse(rows[a].begin(), rows[a].end());
    }
    gp_faco::DistanceOrderedCandidates candidates(rows, distance);
    std::vector<gp_faco::FacoDiagnosticTask> tasks;
    for (Node ant = 0; ant < ants; ++ant) {
        gp_faco::FacoDiagnosticTask task;
        task.tour.resize(n); task.visit_order.resize(n);
        std::iota(task.tour.begin(), task.tour.end(), 0);
        std::iota(task.visit_order.begin(), task.visit_order.end(), 0);
        std::shuffle(task.tour.begin(), task.tour.end(), random);
        std::shuffle(task.visit_order.begin(), task.visit_order.end(), random);
        task.mne_target = std::array<Node, 4>{2, 4, 8, 16}[ant % 4];
        tasks.push_back(std::move(task));
    }
    const auto results = gp_faco::cuda_faco_diagnostic(matrix, rows, tasks, cap);
    for (Node ant = 0; ant < ants; ++ant) {
        context = "n=" + std::to_string(n) + " metric=" + std::to_string(metric) +
                  " cap=" + std::to_string(cap) + " ant=" + std::to_string(ant);
        const auto& task = tasks[ant];
        const auto& gpu = results[ant];
        gp_faco::CpuTour parent(task.tour, distance);
        gp_faco::FocusedConstruction cpu(parent, task.visit_order[0], task.mne_target);
        Node visit = 1;
        while (!cpu.done()) cpu.step(task.visit_order[visit++]);
        require(gpu.construction_tour == cpu.tour().order(), "构造tour不同");
        require(gpu.info.construction.mne == cpu.stats().mne &&
                gpu.info.construction.steps == cpu.stats().steps &&
                gpu.info.construction.nonidentity_relocations == cpu.stats().nonidentity_relocations,
                "构造计数不同");
        compare_cost(gpu.info.construction_cost, cpu.tour().cost());
        const auto stats = cpu.tour().checklist_two_opt(candidates, cpu.checklist(), cap);
        require(gpu.tour == cpu.tour().order(), "LS tour不同");
        require(gpu.positions == cpu.tour().positions(), "inverse positions不同");
        require(gpu.checklist == cpu.checklist(), "checklist顺序不同");
        const auto& observed = gpu.info.local_search;
        require(observed.processed_nodes == stats.processed_nodes &&
                observed.candidate_checks == stats.candidate_checks &&
                observed.move_evaluations == stats.move_evaluations &&
                observed.accepted_moves == stats.accepted_moves &&
                observed.reactivations == stats.reactivations &&
                observed.evaluation_limit_reached == stats.evaluation_limit_reached,
                "LS精确工作量/截断计数不同");
        compare_cost(gpu.info.final_cost, cpu.tour().cost());
        // CpuTour重新构造也验证完整排列，不只相信两后端增量值。
        gp_faco::CpuTour independent(gpu.tour, distance);
        compare_cost(gpu.info.final_cost, independent.cost());
        ++cases;
        transitions += cpu.stats().steps;
        evaluations += stats.move_evaluations;
        reactivations += stats.reactivations;
    }
}

void identity_move_regression() {
    // 三节点闭环没有真正的 2-opt 移动，但原浮点表达式会把恒等移动记成正 gain。
    volatile double x = 0.3, y = 0.1;
    require(x + y - y - x > 0, "恒等移动回归样例未触发浮点假增益");
    const std::vector<double> matrix{0, .25, .3, .25, 0, .1, .3, .1, 0};
    const gp_faco::CandidateRows rows{{1,2},{2,0},{1,0}};
    const auto distance = [&](Node a, Node b) { return matrix[a * 3 + b]; };
    gp_faco::CpuTour cpu({0,1,2}, distance);
    gp_faco::DistanceOrderedCandidates candidates(rows, distance);
    std::vector<Node> checklist{0,1,2};
    const auto stats = cpu.checklist_two_opt(candidates, checklist, 1000);
    require(stats.accepted_moves == 0 && cpu.order() == std::vector<Node>({0,1,2}),
            "CPU 把恒等移动记成局部搜索改善");
    gp_faco::FacoDiagnosticTask task;
    task.tour = task.visit_order = {0,1,2}; task.mne_target = 2;
    const auto result = gp_faco::cuda_faco_diagnostic(matrix, rows, {task}, 1000).front();
    require(result.info.local_search.accepted_moves == 0 && result.tour == task.tour,
            "CUDA 把恒等移动记成局部搜索改善");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        cudaDeviceProp device{};
        if (cudaGetDeviceProperties(&device, 0) != cudaSuccess) throw std::runtime_error("没有可用CUDA设备");
        identity_move_regression();
        // --small供sanitizer加速，仍覆盖1K、三种距离、全部评价上限及4个MNE档。
        const bool small = argc == 3 && std::string(argv[2]) == "--small";
        const Node ants = small ? 4 : 16;
        for (Node n : {3u, 4u, 5u, 7u, 8u, 17u, 100u, 500u, 1000u}) {
            for (int metric = 0; metric < 3; ++metric) {
                for (std::uint64_t cap : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{7},
                                         std::uint64_t{127}, std::numeric_limits<std::uint64_t>::max()}) {
                    run(n, metric, cap, ants);
                }
            }
        }
        require(evaluations > 0 && reactivations > 0, "没有实际执行LS");
        std::ostringstream report;
        report.precision(17);
        report << "{\n  \"status\": \"passed\",\n  \"device\": \"" << device.name << "\",\n"
               << "  \"cases\": " << cases << ",\n  \"construction_transitions\": " << transitions
               << ",\n  \"ls_evaluations\": " << evaluations
               << ",\n  \"ls_reactivations\": " << reactivations
               << ",\n  \"maximum_absolute_cost_error\": " << maximum_error
               << ",\n  \"scope\": \"explicit-choice CPU/CUDA operation diagnostic; full Engine pending\"\n}\n";
        if (argc >= 2) {
            std::ofstream stream(argv[1]);
            if (!stream) throw std::runtime_error("无法写入CUDA诊断报告");
            stream << report.str();
        }
        std::cout << report.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
