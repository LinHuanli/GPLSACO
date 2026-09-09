// 次数限额的精确提交、无墙钟截止、新progress语义与批量隔离。
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
Program controller() {
    Program p; p.feature_spec_id = 2; p.length = 9;
    const std::uint8_t op[]{0,0,2,0,0,3,0,4,2}, arg[]{4,8,0,0,2,0,5,0,0};
    std::copy(std::begin(op), std::end(op), p.opcode);
    std::copy(std::begin(arg), std::end(arg), p.operand);
    validate_program(p); return p;
}
auto state(const ControllerState& c) {
    return std::make_tuple(c.archive_size, c.tracked_global, c.tracked_epoch,
        c.feedback.return_rate, c.feedback.ls_work, c.feedback.stagnant_batches,
        c.feedback.epoch_batches, c.feedback.restarts, c.active_identity.first, c.active_identity.second);
}
void identical(const BatchEvaluation& a, const BatchEvaluation& b) {
    require(a.completed_batches == b.completed_batches &&
        a.completed_construction_steps == b.completed_construction_steps &&
        a.completed_ls_evaluations == b.completed_ls_evaluations, "计时改变FE或搜索工作");
    for (unsigned i = 0; i < a.incumbents.size(); ++i) {
        require(a.incumbents[i].tour == b.incumbents[i].tour &&
                a.incumbents[i].cost == b.incumbents[i].cost, "计时或准备模式改变轨迹");
        require(state(a.completed_control_states[i]) == state(b.completed_control_states[i]), "控制状态改变");
    }
    require(a.control_trace.size() == b.control_trace.size(), "轨迹批数改变");
    for (unsigned i = 0; i < a.control_trace.size(); ++i) {
        const auto& x = a.control_trace[i]; const auto& y = b.control_trace[i];
        require(x.features == y.features && x.actions == y.actions && x.scores == y.scores &&
            x.after_batch.archive == y.after_batch.archive &&
            x.after_batch.trails == y.after_batch.trails &&
            x.after_batch.tours == y.after_batch.tours, "进度/档案/信息素或工作tour改变");
    }
}
template<class Operation>
void invalid(Operation&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "非法限额或特征版本未被拒绝");
}
}

int main(int argc, char** argv) {
    try {
        constexpr Node n = 61, ants = 8, colonies = 4, batches = 6;
        FixedFacoSettings settings; settings.ants = ants;
        FacoBatchEngine engine(n, colonies, settings), single(n, 1, settings);
        std::mt19937_64 random(403);
        for (Node key = 1; key <= 2; ++key) {
            std::vector<double> xy(2 * n);
            for (auto& value : xy) value = std::generate_canonical<double, 53>(random);
            engine.register_problem(key, xy); single.register_problem(key, xy);
            // 极大旧时间扣费不能减少新次数入口的工作。
            engine.set_preparation_charges(key, {1e9, 1e9});
        }
        const std::vector<BatchTask> tasks{{1,17},{1,29},{2,17},{2,29}};
        auto program = controller();
        BatchDiagnosticControls controls; controls.capture_control = true;
        auto reference = engine.evaluate_program_evaluations(tasks, ants*batches, program,
            PreparationMode::CachedCharged, UINT32_MAX, controls);
        require(reference.count_limited && reference.charged_seconds == 0 &&
            reference.overrun_seconds == 0 && reference.discarded_batches == 0 &&
            reference.completed_batches == batches && reference.launched_batches == batches &&
            reference.completed_tour_evaluations_per_colony == ants*batches &&
            reference.total_tour_evaluations == ants*batches*colonies, "次数或资源账目不符");
        for (Node batch = 0; batch < batches; ++batch) {
            const auto& trace = reference.control_trace[batch];
            const auto progress = static_cast<float>(static_cast<double>(batch) / batches);
            require(trace.elapsed_ratio == static_cast<double>(batch) / batches, "progress使用了墙钟");
            for (Node i = 0; i < colonies*32; ++i) require(trace.features[i] == progress, "特征0未完整广播FE进度");
            const auto cpu = score_cpu(program, trace.features, trace.masks);
            require(cpu.actions == trace.actions && cpu.scores == trace.scores, "v2程序CPU/GPU评分不符");
        }
        controls.completion_delay_ms = 200; controls.delay_batch = 0;
        const auto delayed = engine.evaluate_program_evaluations(tasks, ants*batches, program,
            PreparationMode::CachedCharged, UINT32_MAX, controls);
        identical(reference, delayed);
        require(delayed.actual_seconds >= 0.2 && delayed.discarded_batches == 0, "诊断延迟变成了搜索截止");
        controls.completion_delay_ms = 0; controls.profile = true;
        const auto profiled = engine.evaluate_program_evaluations(tasks, ants*batches, program,
            PreparationMode::CachedCharged, UINT32_MAX, controls);
        identical(reference, profiled);
        require(profiled.profile.batches.size() == batches, "次数profile没有完整记录");
        controls.profile = false;
        identical(reference, engine.evaluate_program_evaluations(tasks, ants*batches, program,
            PreparationMode::EndToEnd, UINT32_MAX, controls));
        for (Node i = 0; i < colonies; ++i) {
            const auto solo = single.evaluate_program_evaluations({tasks[i]}, ants*batches, program,
                PreparationMode::CachedCharged);
            require(solo.incumbents[0].tour == reference.incumbents[i].tour,
                "相同FE下批量形状改变单实例路线");
            // 生产入口返回设备累计成本，最终路线仅由外部 evaluator 重算一次。
            // 诊断入口保留独立重算，二者只允许浮点累计误差，路线与控制状态仍须精确相同。
            require(std::abs(solo.incumbents[0].cost - reference.incumbents[i].cost) <= 1e-10,
                "设备累计成本与诊断最终评分不符");
            require(state(solo.completed_control_states[0]) == state(reference.completed_control_states[i]),
                "相同FE下批量形状改变单实例控制状态");
        }
        const auto zero = engine.evaluate_program_evaluations(tasks, 0, program, PreparationMode::EndToEnd);
        require(zero.completed_batches == 0 && zero.launched_batches == 0 &&
            zero.preparation_completed && zero.total_tour_evaluations == 0, "零FE启动蚂蚁搜索");
        for (const auto& item : zero.incumbents) require(item.present, "零FE未保留共同初解");
        invalid([&]() { engine.evaluate_program_evaluations(tasks, ants+1, program, PreparationMode::CachedCharged); });
        invalid([&]() { engine.evaluate_program(tasks, 1, program, PreparationMode::CachedCharged); });
        program.feature_spec_id = 1;
        invalid([&]() { engine.evaluate_program_evaluations(tasks, ants, program, PreparationMode::CachedCharged); });
        std::ofstream output(argc > 1 ? argv[1] : "evaluation_count_semantics.json");
        require(static_cast<bool>(output), "不能写入次数验收记录");
        output << "{\"status\":\"passed\",\"colonies\":4,\"ants\":8,\"batches\":6,"
                  "\"evaluations_per_colony\":48,\"shape_pairs\":4,\"delay_ms\":200}\n";
        std::cout << "evaluation-count exactness and delay/shape invariance: passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
