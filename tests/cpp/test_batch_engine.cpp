// 固定面板等价、模式切换、费用和实际迟到批次的验收。
#include "gp_faco/batch_engine.hpp"
#include "gp_faco/prepared_problem.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using gp_faco::Node;
using gp_faco::PreparationMode;
void check(bool result, const char* message) { if (!result) throw std::runtime_error(message); }
void near(double a, double b) { check(std::isfinite(a) && std::abs(a - b) < 1e-9, "成本不同"); }

std::vector<double> points(Node n, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> unit(0, 1);
    std::vector<double> result(n * 2);
    for (auto& v : result) v = unit(random);
    return result;
}

std::uint64_t configurations = 0, task_comparisons = 0;
double maximum_charged_seconds = 0, observed_late_seconds = 0;

void check_result(const gp_faco::BatchEvaluation& result) {
    for (const auto& item : result.incumbents) {
        if (item.present) check(item.completed_seconds <= result.budget_seconds, "接受了迟到incumbent");
    }
    near(result.elapsed_seconds, result.actual_seconds + result.charged_seconds);
    check(result.completed_batches + result.discarded_batches == result.launched_batches, "批次账目不平");
    check(result.discarded_batches <= 1, "迟到之后继续启动批次");
}

void compare_panel(Node n, bool small) {
    gp_faco::FixedFacoSettings settings;
    settings.ants = small ? 4 : 8;
    settings.retention = 0.75;
    const Node batches = small ? 2 : 5;
    const auto a = points(n, 301), b = points(n, 809);
    const std::vector<gp_faco::BatchTask> tasks{{71, 17}, {71, 29}, {93, 17}, {71, 17}};
    gp_faco::FacoBatchEngine engine(n, tasks.size(), settings);
    const auto first_fee = engine.register_problem(71, a), second_fee = engine.register_problem(93, b);
    const double total_fee = first_fee.cheap_seconds + first_fee.preparation_seconds +
                             second_fee.cheap_seconds + second_fee.preparation_seconds;
    gp_faco::BatchDiagnosticControls controls; controls.fixed_batches = batches;
    const auto cached = engine.evaluate_diagnostic(tasks, 30, 8, PreparationMode::CachedCharged, controls);
    check_result(cached);
    check(cached.preparation_completed && cached.completed_batches == batches, "并发固定批次未完成");
    near(cached.charged_seconds, total_fee);
    maximum_charged_seconds = std::max(maximum_charged_seconds, cached.charged_seconds);
    std::uint64_t steps = 0, evaluations = 0;
    for (Node i = 0; i < tasks.size(); ++i) {
        const auto& coordinates = tasks[i].instance_key == 71 ? a : b;
        gp_faco::FixedFacoGpu single(coordinates, tasks[i].instance_key, settings);
        const auto reference = single.run_iterations(tasks[i].seed, batches, 8);
        check(cached.incumbents[i].present, "并发任务缺少可行解");
        near(cached.incumbents[i].cost, reference.cost);
        steps += reference.construction_steps; evaluations += reference.ls_evaluations;
        ++task_comparisons;
    }
    check(cached.completed_construction_steps == steps && cached.completed_ls_evaluations == evaluations,
          "并发任务的构造/LS路径与单实例不一致");
    check(cached.incumbents[0].tour == cached.incumbents[3].tour, "重复任务受colony编号影响");
    const auto cold = engine.evaluate_diagnostic(tasks, 30, 8, PreparationMode::EndToEnd, controls);
    check_result(cold);
    check(cold.charged_seconds == 0 && cold.completed_batches == batches, "端到端模式重复收费或未完成");
    for (Node i = 0; i < tasks.size(); ++i) {
        near(cold.incumbents[i].cost, cached.incumbents[i].cost);
        check(cold.incumbents[i].tour == cached.incumbents[i].tour, "重新准备改变了结果");
    }
    auto shuffled = tasks;
    std::reverse(shuffled.begin(), shuffled.end());
    const auto reordered = engine.evaluate_diagnostic(shuffled, 30, 8, PreparationMode::CachedCharged, controls);
    check_result(reordered);
    for (Node i = 0; i < tasks.size(); ++i)
        check(reordered.incumbents[tasks.size() - 1 - i].tour == cached.incumbents[i].tour,
              "任务顺序改变了随机流或状态");
    const auto zero = engine.evaluate(tasks, 0, 8, PreparationMode::CachedCharged);
    check_result(zero);
    check(zero.launched_batches == 0 && !zero.preparation_completed, "零预算执行了搜索");
    for (const auto& result : zero.incumbents) check(!result.present, "零预算泄漏旧GPU incumbent");
    ++configurations;
}

void preparation_cutoff() {
    const auto a = points(1000, 821), b = points(1000, 937);
    gp_faco::FixedFacoSettings settings; settings.ants = 4;
    gp_faco::FacoBatchEngine engine(1000, 2, settings);
    const auto x = engine.register_problem(11, a), y = engine.register_problem(12, b);
    const double budget = x.cheap_seconds + y.cheap_seconds + 0.25 * x.preparation_seconds;
    const auto result = engine.evaluate({{11, 17}, {12, 17}}, budget, 8, PreparationMode::CachedCharged);
    check_result(result);
    check(!result.preparation_completed && result.launched_batches == 0, "准备超预算后仍启动GPU搜索");
    for (Node i = 0; i < 2; ++i) {
        const auto problem = gp_faco::make_cheap_problem(i ? b : a, settings);
        check(result.incumbents[i].present && result.incumbents[i].tour == problem.cheap_tour,
              "昂贵准备超时丢失先前廉价解");
    }
}

void actual_late_improvement() {
    const auto xy = points(100, 4411);
    gp_faco::FixedFacoSettings settings; settings.ants = 8; settings.initial_ls_evaluation_limit = 0;
    gp_faco::FacoBatchEngine engine(100, 1, settings);
    engine.register_problem(501, xy);
    gp_faco::BatchDiagnosticControls one; one.fixed_batches = 1;
    const auto before = gp_faco::make_cheap_problem(xy, settings);
    auto prepared = before; gp_faco::prepare_problem(prepared);
    const auto reference = engine.evaluate_diagnostic({{501, 17}}, 10, 16, PreparationMode::CachedCharged, one);
    check(reference.incumbents[0].cost < prepared.initial_cost, "迟到测试fixture没有实际改进");
    gp_faco::BatchDiagnosticControls delay;
    delay.completion_delay_ms = 1000; delay.capture_discarded = true;
    const auto late = engine.evaluate_diagnostic({{501, 17}}, 0.5, 16, PreparationMode::CachedCharged, delay);
    check_result(late);
    check(late.launched_batches == 1 && late.discarded_batches == 1 && late.completed_batches == 0,
          "未丢弃实际迟到批次");
    check(late.discarded_costs.size() == 1 && late.discarded_costs[0] < late.incumbents[0].cost,
          "未覆盖真实更优的迟到候选");
    near(late.incumbents[0].cost, prepared.initial_cost);
    check(late.last_batch_completed_seconds > late.budget_seconds && late.overrun_seconds > 0,
          "未记录实际超限");
    observed_late_seconds = late.last_batch_completed_seconds;
    const auto again = engine.evaluate_diagnostic({{501, 17}}, 10, 16, PreparationMode::CachedCharged, one);
    check(again.incumbents[0].tour == reference.incumbents[0].tour, "迟到状态污染下一次评价");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool small = argc == 3 && std::string(argv[2]) == "--small";
        for (Node n : {17u, 100u, 500u, 1000u}) compare_panel(n, small);
        preparation_cutoff(); actual_late_improvement();
        std::ostringstream out;
        out.precision(17);
        out << "{\n  \"status\": \"passed\",\n  \"panel_configurations\": " << configurations
            << ",\n  \"single_colony_comparisons\": " << task_comparisons
            << ",\n  \"maximum_charged_seconds\": " << maximum_charged_seconds
            << ",\n  \"late_batch_completion_seconds\": " << observed_late_seconds
            << ",\n  \"late_budget_seconds\": 0.5,\n  \"real_late_improvement_discarded\": true,\n"
            << "  \"scope\": \"batch and deadline Engine; GP/restart/graph constraints pending\"\n}\n";
        if (argc >= 2) { std::ofstream file(argv[1]); if (!file) throw std::runtime_error("无法写报告"); file << out.str(); }
        std::cout << out.str(); return 0;
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
