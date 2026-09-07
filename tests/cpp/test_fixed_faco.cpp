// 重放实际GPU随机选点，逐批检查构造、LS、参考与稀疏信息素状态。
#include "gp_faco/fixed_faco_gpu.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using gp_faco::Node;
std::string context;
std::uint64_t runs = 0, ant_cases = 0, decisions = 0, backup_choices = 0, global_choices = 0;
double max_cost_error = 0, max_pheromone_error = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string(message) + " " + context);
}
void cost(double a, double b) {
    max_cost_error = std::max(max_cost_error, std::abs(a - b));
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < 1e-8, "路线成本不符");
}
void trail(double a, double b) {
    max_pheromone_error = std::max(max_pheromone_error, std::abs(a - b));
    require(std::isfinite(a) && std::abs(a - b) < 1e-12, "信息素状态不符");
}

void test(Node n, int variant, bool small) {
    std::mt19937_64 random(4703 + n);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<double> xy(n * 2);
    for (double& v : xy) v = uniform(random);
    if (variant == 3) for (Node i = 3; i < n; i += 3) {
        xy[i * 2] = xy[0]; xy[i * 2 + 1] = xy[1];
    }
    auto distance = [&](Node a, Node b) {
        const double x = xy[a * 2] - xy[b * 2], y = xy[a * 2 + 1] - xy[b * 2 + 1];
        return std::sqrt(x*x + y*y);
    };
    gp_faco::FixedFacoSettings settings;
    settings.ants = small ? 4 : 8;
    settings.primary_width = variant % 2 ? 2 : 16;
    settings.backup_width = variant % 2 ? 1 : 64;
    settings.retention = std::array<double, 4>{0.25, 0.5, 0.75, 0.9}[variant];
    settings.epoch_source_probability = std::array<double, 4>{0, 0.01, 1, 0.3}[variant];
    settings.ls_evaluation_limit = 5000;
    gp_faco::FixedFacoGpu engine(xy, n + 8999, settings);
    const auto first = engine.run_iterations(17, small ? 3 : 8, 16, true);
    const auto& primary = engine.primary_candidates();
    const auto& backup = engine.backup_candidates();
    gp_faco::DistanceOrderedCandidates ls(engine.ls_candidates(), distance);
    const auto width = static_cast<Node>(primary[0].size());
    std::vector<Node> expected_parent = engine.initial_tour(), expected_best = expected_parent;
    double best_cost = first.initial_cost;
    const auto initial_bounds = gp_faco::candidate_trail_limits(width, settings.p_best, settings.retention, best_cost);
    gp_faco::SparsePheromone pheromone(primary, initial_bounds.maximum, true);
    for (std::size_t batch = 0; batch < first.trace.size(); ++batch) {
        const auto& trace = first.trace[batch];
        context = "n=" + std::to_string(n) + " variant=" + std::to_string(variant) +
                  " batch=" + std::to_string(batch);
        require(trace.parent_before == expected_parent, "批次源tour不符");
        trail(trace.default_before, pheromone.default_value());
        for (Node a = 0; a < n; ++a) for (Node j = 0; j < width; ++j)
            trail(trace.trails_before[a * width + j], pheromone.get(a, primary[a][j]));
        Node iteration_best = 0;
        double iteration_cost = std::numeric_limits<double>::infinity();
        for (Node ant = 0; ant < settings.ants; ++ant) {
            const auto& gpu = trace.ants[ant];
            const auto offset = static_cast<std::size_t>(ant) * n;
            gp_faco::CpuTour parent(expected_parent, distance);
            gp_faco::FocusedConstruction cpu(parent, trace.selected_nodes[offset], 16);
            while (!cpu.done()) {
                const Node current = cpu.current();
                const Node step = cpu.stats().steps + 1;
                std::vector<double> products(trace.products_before.begin() + current * width,
                                              trace.products_before.begin() + (current + 1) * width);
                const auto chosen = gp_faco::select_next(current, primary[current], products,
                    backup[current], cpu.visited(), distance, trace.selection_uniforms[offset + step]);
                require(chosen.node == trace.selected_nodes[offset + step], "GPU随机选点与CPU概率/回退不同");
                backup_choices += chosen.stage == gp_faco::SelectionStage::Backup;
                global_choices += chosen.stage == gp_faco::SelectionStage::GlobalFallback;
                ++decisions;
                cpu.step(chosen.node);
            }
            require(cpu.tour().order() == gpu.construction_tour, "构造tour不符");
            require(cpu.stats().mne == gpu.info.construction.mne &&
                    cpu.stats().steps == gpu.info.construction.steps &&
                    cpu.stats().nonidentity_relocations == gpu.info.construction.nonidentity_relocations,
                    "构造状态不符");
            cost(cpu.tour().cost(), gpu.info.construction_cost);
            const auto stats = cpu.tour().checklist_two_opt(ls, cpu.checklist(), settings.ls_evaluation_limit);
            require(cpu.tour().order() == gpu.tour && cpu.tour().positions() == gpu.positions &&
                    cpu.checklist() == gpu.checklist, "LS状态不符");
            require(stats.move_evaluations == gpu.info.local_search.move_evaluations &&
                    stats.accepted_moves == gpu.info.local_search.accepted_moves &&
                    stats.reactivations == gpu.info.local_search.reactivations, "LS工作量不符");
            cost(cpu.tour().cost(), gpu.info.final_cost);
            cost(cpu.tour().recomputed_cost(), gpu.info.final_cost);
            if (gpu.info.final_cost < iteration_cost) { iteration_cost = gpu.info.final_cost; iteration_best = ant; }
            ++ant_cases;
        }
        require(trace.iteration_best == iteration_best, "批次最好解归约不符");
        if (iteration_cost < best_cost) { best_cost = iteration_cost; expected_best = trace.ants[iteration_best].tour; }
        require(trace.global_best == expected_best && trace.epoch_best == expected_best, "GB/epoch最好解丢失");
        cost(trace.global_cost, best_cost); cost(trace.epoch_cost, best_cost);
        require(trace.source_is_epoch == (trace.source_uniform < settings.epoch_source_probability), "强化概率分支不符");
        expected_parent = trace.source_is_epoch ? expected_best : trace.ants[iteration_best].tour;
        require(trace.parent_after == expected_parent, "下一批源未使用实际强化tour");
        const double deposit_cost = trace.source_is_epoch ? best_cost : iteration_cost;
        const auto limits = gp_faco::candidate_trail_limits(width, settings.p_best, settings.retention, best_cost);
        trail(trace.minimum, limits.minimum); trail(trace.maximum, limits.maximum);
        pheromone.evaporate(settings.retention, limits.minimum);
        for (Node i = 0; i < n; ++i)
            pheromone.deposit(expected_parent[(i + n - 1) % n], expected_parent[i], 1.0 / deposit_cost, limits.maximum);
        for (Node a = 0; a < n; ++a) for (Node j = 0; j < width; ++j)
            trail(trace.trails_after[a * width + j], pheromone.get(a, primary[a][j]));
        trail(trace.default_after, pheromone.default_value());
    }
    require(first.tour == expected_best, "返回结果不是已登记最好解");
    cost(first.cost, gp_faco::CpuTour(first.tour, distance).cost());
    // 同一个持久对象先运行不同seed，再恢复原任务；轨迹收集不能改变求解行为。
    engine.run_iterations(29, 2, 2);
    const auto again = engine.run_iterations(17, small ? 3 : 8, 16);
    require(again.tour == first.tour && again.cost == first.cost &&
            again.construction_steps == first.construction_steps && again.ls_evaluations == first.ls_evaluations,
            "seed切换或trace模式污染动态状态");
    const auto zero = engine.run_iterations(17, 0, 8);
    require(zero.tour == engine.initial_tour() && zero.cost == first.initial_cost, "零批次没有重置初始解");
    ++runs;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool small = argc == 3 && std::string(argv[2]) == "--small";
        for (Node n : {7u, 17u, 100u, 500u, 1000u}) for (int variant = 0; variant < 4; ++variant) test(n, variant, small);
        require(backup_choices > 0 && global_choices > 0, "没有覆盖备用/全局回退");
        std::ostringstream report;
        report.precision(17);
        report << "{\n  \"status\": \"passed\",\n  \"colony_configurations\": " << runs
            << ",\n  \"ant_batch_cases\": " << ant_cases << ",\n  \"selection_decisions\": " << decisions
            << ",\n  \"backup_choices\": " << backup_choices << ",\n  \"global_fallback_choices\": " << global_choices
            << ",\n  \"maximum_cost_error\": " << max_cost_error
            << ",\n  \"maximum_pheromone_error\": " << max_pheromone_error
            << ",\n  \"scope\": \"fixed-iteration GPU pipeline; deadline/GP Engine pending\"\n}\n";
        if (argc >= 2) { std::ofstream file(argv[1]); if (!file) throw std::runtime_error("无法写报告"); file << report.str(); }
        std::cout << report.str();
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
