// 固定相同elapsed输入，逐批对照计时/未计时的真实CUDA程序状态；不把profile当fitness。
#include "gp_faco/batch_engine.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>

namespace {
using namespace gp_faco;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

Program program() {
    // restart + region_excess + (elapsed - return_rate) * mne_level。
    Program p; p.length = 9;
    const std::uint8_t op[]{0, 0, 2, 0, 0, 3, 0, 4, 2};
    const std::uint8_t arg[]{4, 8, 0, 0, 2, 0, 5, 0, 0};
    std::copy(std::begin(op), std::end(op), p.opcode);
    std::copy(std::begin(arg), std::end(arg), p.operand);
    validate_program(p); return p;
}

auto feedback(const ControllerState& c) {
    return std::make_tuple(c.archive_size, c.tracked_global, c.tracked_epoch,
        c.feedback.return_rate, c.feedback.ls_work, c.feedback.stagnant_batches,
        c.feedback.epoch_batches, c.feedback.restarts, c.active_identity.first,
        c.active_identity.second, c.epoch_identity.first, c.epoch_identity.second);
}

void same(const BatchEvaluation& a, const BatchEvaluation& b) {
    require(a.incumbents.size() == b.incumbents.size() && a.completed_batches == b.completed_batches &&
            a.completed_construction_steps == b.completed_construction_steps &&
            a.completed_ls_evaluations == b.completed_ls_evaluations, "profile改变搜索计数");
    require(a.discarded_batches == 0 && b.discarded_batches == 0, "固定批次检查意外截止");
    for (unsigned i = 0; i < a.incumbents.size(); ++i) {
        require(a.incumbents[i].tour == b.incumbents[i].tour && a.incumbents[i].cost == b.incumbents[i].cost,
                "profile改变完整tour或成本");
        require(feedback(a.completed_control_states[i]) == feedback(b.completed_control_states[i]),
                "profile改变控制状态");
    }
    require(a.control_trace.size() == b.control_trace.size(), "profile改变trace批数");
    for (unsigned i = 0; i < a.control_trace.size(); ++i) {
        const auto& x = a.control_trace[i]; const auto& y = b.control_trace[i];
        require(x.features == y.features && x.scores == y.scores && x.actions == y.actions &&
                x.alternatives == y.alternatives && x.masks == y.masks, "profile改变特征/评分/动作");
        require(x.after_batch.archive == y.after_batch.archive &&
                x.after_batch.archive_positions == y.after_batch.archive_positions &&
                x.after_batch.parent == y.after_batch.parent && x.after_batch.epoch == y.after_batch.epoch &&
                x.after_batch.trails == y.after_batch.trails && x.after_batch.products == y.after_batch.products,
                "profile改变档案、参考或信息素");
        require(x.after_batch.tours == y.after_batch.tours && x.after_batch.positions == y.after_batch.positions &&
                x.after_restart.tours == y.after_restart.tours &&
                x.after_restart.pending == y.after_restart.pending &&
                x.after_restart.visited == y.after_restart.visited, "profile改变蚂蚁或重启工作状态");
    }
}

std::uint64_t configurations = 0, batches = 0, ants_checked = 0;
void check_configuration(Node n, Node colonies, Node ants, Node count, bool trace) {
    FixedFacoSettings settings; settings.ants = ants;
    FacoBatchEngine engine(n, colonies, settings);
    std::vector<BatchTask> tasks;
    std::mt19937_64 rng(7711 + n);
    for (Node i = 0; i < colonies; ++i) {
        std::vector<double> xy(2 * n);
        for (auto& x : xy) x = std::generate_canonical<double, 53>(rng);
        const auto fees = engine.register_problem(i + 1, xy);
        const auto preparation = engine.preparation_profile(i + 1);
        const double total = preparation.candidates_and_scales_seconds + preparation.nearest_neighbor_seconds +
            preparation.initial_ls_seconds + preparation.finalization_seconds;
        require(total > 0 && std::abs(total - fees.preparation_seconds) < 1e-10,
                "CPU准备分项没有覆盖完整昂贵阶段");
        engine.set_preparation_charges(i + 1, fees);
        tasks.push_back({i + 1, 83 + i});
    }
    BatchDiagnosticControls control;
    control.fixed_batches = count; control.fixed_elapsed_ratio = 0.25;
    control.capture_control = trace;
    const auto baseline = engine.evaluate_program_diagnostic(tasks, 120, program(), PreparationMode::CachedCharged,
                                                             UINT32_MAX, control);
    require(baseline.profile.batches.empty() && baseline.profile.diagnostic_device_bytes == 0,
            "未开启profile却创建或返回诊断数据");
    control.profile = true;
    const auto profiled = engine.evaluate_program_diagnostic(tasks, 120, program(), PreparationMode::CachedCharged,
                                                             UINT32_MAX, control);
    same(baseline, profiled);
    require(profiled.profile.batches.size() == count && profiled.completed_batches == count,
            "没有收齐全部固定批次的profile");
    require(profiled.profile.diagnostic_device_bytes == colonies * ants * sizeof(AntPhaseCycles) &&
            profiled.allocated_device_bytes == baseline.allocated_device_bytes, "诊断显存未与通用显存分列");
    require(profiled.profile.initialization_gpu_milliseconds > 0, "初始化event未记录");
    for (const auto& batch : profiled.profile.batches) {
        require(batch.committed && batch.ant_cycles.size() == colonies * ants && batch.wall_seconds > 0,
                "profile批次或周期数组不完整");
        double gpu_ms = 0;
        for (double value : batch.gpu_milliseconds) {
            require(std::isfinite(value) && value >= 0, "GPU event计时非有限或为负");
            gpu_ms += value;
        }
        require(gpu_ms > 0 && gpu_ms / 1000 <= batch.wall_seconds * 1.01 + 0.001,
                "GPU阶段event合计超过包围它们的host区间");
        for (const auto& value : batch.ant_cycles) {
            require(value.initialization > 0 && value.construction > 0 && value.local_search > 0 &&
                    value.finalization > 0, "设备周期分项缺失");
            ++ants_checked;
        }
        ++batches;
    }
    control.profile = false;
    same(baseline, engine.evaluate_program_diagnostic(tasks, 120, program(), PreparationMode::CachedCharged,
                                                      UINT32_MAX, control));
    control.profile = true; control.fixed_elapsed_ratio = -1;
    bool rejected = false;
    try {
        engine.evaluate_program_diagnostic(tasks, 120, program(), PreparationMode::CachedCharged,
                                            UINT32_MAX, control);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "profile允许动态wall-clock特征，无法作等轨迹计时对照");
    ++configurations;
}
}

int main(int argc, char** argv) {
    try {
        check_configuration(31, 4, 4, 5, true);
        check_configuration(101, 4, 8, 5, true);
        if (!(argc > 2 && std::string(argv[2]) == "--small")) {
            check_configuration(500, 32, 32, 2, false);
            check_configuration(1000, 32, 32, 2, false);
        }
        const std::string path = argc > 1 ? argv[1] : "profiling_semantics.json";
        std::ofstream out(path);
        require(static_cast<bool>(out), "无法写入profile检查报告");
        out << "{\"status\":\"passed\",\"configurations\":" << configurations
            << ",\"batches\":" << batches << ",\"ant_phase_checks\":" << ants_checked << "}\n";
        std::cout << "profile逐字段与固定批次对照通过: " << configurations << " configurations\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
