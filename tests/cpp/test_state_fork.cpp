// 完整未切断轨迹作为独立对照；分支必须从同一真实状态重新恢复。
#include "gp_faco/batch_engine.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {
using namespace gp_faco;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Operation> void invalid(Operation operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "非法快照或分叉参数未被拒绝");
}
Program program() {
    Program p; p.feature_spec_id = 2; p.length = 9;
    const std::uint8_t op[]{0,0,2,0,0,3,0,4,2}, arg[]{4,8,0,0,2,0,5,0,0};
    std::copy(std::begin(op), std::end(op), p.opcode);
    std::copy(std::begin(arg), std::end(arg), p.operand); validate_program(p); return p;
}
void controller_equal(const ControllerState& a, const ControllerState& b) {
    const auto tuple = [](const auto& c) { return std::make_tuple(c.archive_size,
        c.tracked_global, c.tracked_epoch, c.feedback.return_rate, c.feedback.ls_work,
        c.feedback.stagnant_batches, c.feedback.epoch_batches, c.feedback.restarts); };
    check(tuple(a) == tuple(b) && a.active_identity == b.active_identity && a.epoch_identity == b.epoch_identity,
          "控制器反馈/身份改变");
    for (Node i = 0; i < archive_capacity; ++i)
        check(a.archive_cost[i] == b.archive_cost[i] && a.archive_identity[i] == b.archive_identity[i],
              "档案成本/身份改变");
}
auto info_tuple(const FacoDiagnosticInfo& i) {
    const auto& c = i.construction; const auto& l = i.local_search;
    return std::make_tuple(i.construction_cost, i.final_cost, i.checklist_size, i.start_node,
        c.mne, c.steps, c.nonidentity_relocations, c.legal_exhausted, l.processed_nodes,
        l.candidate_checks, l.move_evaluations, l.accepted_moves, l.reactivations,
        l.constraint_rejections, l.evaluation_limit_reached);
}
void state_equal(const ControlDeviceSnapshot& a, const ControlDeviceSnapshot& b,
                 bool compare_targets = true) {
#define EQUAL(member) check(a.member == b.member, "状态字段改变: " #member)
    EQUAL(parent); EQUAL(parent_positions); EQUAL(epoch); EQUAL(global); EQUAL(archive);
    EQUAL(archive_positions); EQUAL(tours); EQUAL(positions); EQUAL(per_ant_parent_positions);
    EQUAL(scratch); EQUAL(pending); EQUAL(visited); EQUAL(trails); EQUAL(products); EQUAL(gains);
    EQUAL(ant_identities);
    if (compare_targets) { EQUAL(targets); }
#undef EQUAL
    check(a.controls.size() == b.controls.size() && a.colonies.size() == b.colonies.size() &&
          a.info.size() == b.info.size(), "诊断形状改变");
    for (std::size_t i = 0; i < a.controls.size(); ++i) controller_equal(a.controls[i], b.controls[i]);
    const auto tuple = [](const auto& c) { return std::make_tuple(c.global_cost, c.epoch_cost,
        c.parent_cost, c.minimum, c.maximum, c.default_trail, c.source_uniform,
        c.iteration_best, c.source_is_epoch); };
    for (std::size_t i = 0; i < a.colonies.size(); ++i)
        check(tuple(a.colonies[i]) == tuple(b.colonies[i]), "colony成本/default或参考来源改变");
    for (std::size_t i = 0; i < a.info.size(); ++i)
        check(info_tuple(a.info[i]) == info_tuple(b.info[i]), "ant工作量或起点改变");
}
void regions_equal(const std::vector<StartRegions>& a, const std::vector<StartRegions>& b) {
    check(a.size() == b.size(), "区域形状改变");
    for (std::size_t i = 0; i < a.size(); ++i)
        check(a[i].count == b[i].count && std::memcmp(a[i].nodes, b[i].nodes, sizeof(a[i].nodes)) == 0,
              "区域节点改变");
}
void trace_equal(const ControlBatchTrace& a, const ControlBatchTrace& b) {
    check(a.batch == b.batch && a.elapsed_ratio == b.elapsed_ratio && a.intervened == b.intervened &&
        a.features == b.features && a.scores == b.scores && a.actions == b.actions &&
        a.alternatives == b.alternatives && a.masks == b.masks && a.baseline_uniforms == b.baseline_uniforms,
        "动作、progress、特征或随机轨迹改变");
    regions_equal(a.regions, b.regions);
    state_equal(a.before, b.before); state_equal(a.after_restart, b.after_restart); state_equal(a.after_batch, b.after_batch);
}
void final_equal(const BatchEvaluation& a, const BatchEvaluation& b) {
    check(a.incumbents.size() == b.incumbents.size(), "incumbent数量改变");
    for (std::size_t i = 0; i < a.incumbents.size(); ++i) {
        check(a.incumbents[i].tour == b.incumbents[i].tour && a.incumbents[i].cost == b.incumbents[i].cost,
              "完整结果改变");
        controller_equal(a.completed_control_states[i], b.completed_control_states[i]);
    }
}
void replay_equal(const BatchEvaluation& a, const BatchEvaluation& b) {
    final_equal(a, b);
    check(a.total_tour_evaluations == b.total_tour_evaluations && a.completed_batches == b.completed_batches &&
        a.completed_construction_steps == b.completed_construction_steps &&
        a.completed_ls_evaluations == b.completed_ls_evaluations && a.control_trace.size() == b.control_trace.size(),
        "重放FE或工作量改变");
    for (std::size_t i = 0; i < a.control_trace.size(); ++i) trace_equal(a.control_trace[i], b.control_trace[i]);
}
}

int main(int argc, char** argv) {
    try {
        constexpr Node n = 61, colonies = 4, ants = 8, batches = 7;
        FixedFacoSettings settings; settings.ants = ants;
        std::mt19937_64 random(623);
        std::vector<std::vector<double>> xy(2, std::vector<double>(n * 2));
        for (auto& points : xy) for (auto& value : points) value = std::generate_canonical<double, 53>(random);
        const auto register_all = [&](FacoBatchEngine& e) {
            e.register_problem(1, xy[0]); e.register_problem(2, xy[1]);
        };
        FacoBatchEngine engine(n, colonies, settings); register_all(engine);
        const std::vector<BatchTask> tasks{{1,17},{1,29},{2,17},{2,29}};
        const auto gp = program();
        BatchDiagnosticControls trace; trace.capture_control = true;
        const auto reference = engine.evaluate_program_evaluations(tasks, ants * batches, gp,
            PreparationMode::CachedCharged, UINT32_MAX, trace);
        BaselinePolicy continuation; continuation.mne_level = continuation.max_mne_level = 2;
        continuation.region = 1;
        CountedState fork_state;
        std::size_t compared_batches = 0;
        for (Node cut : {0u, 1u, 4u, 7u}) {
            CountedState saved;
            const auto prefix = engine.capture_program_state(tasks, ants * batches, ants * cut, gp,
                saved, UINT32_MAX, trace);
            check(prefix.completed_batches == cut && prefix.total_tour_evaluations == ants * cut * colonies &&
                saved.completed_batches == cut && saved.progress_evaluation_limit == ants * batches,
                "切点缩减源progress分母或改变FE");
            const auto bytes = saved.serialize();
            const auto restored = CountedState::deserialize(bytes);
            check(restored.serialize() == bytes && restored.buffers.size() == 44, "序列化字节或完整缓冲集合改变");
            FacoBatchEngine rebuilt(n, colonies, settings); register_all(rebuilt);
            rebuilt.evaluate_baseline_evaluations(tasks, ants * 3, continuation, PreparationMode::CachedCharged);
            const auto suffix = rebuilt.continue_program_state(restored, ants * (batches - cut), gp, trace);
            check(prefix.completed_construction_steps + suffix.completed_construction_steps == reference.completed_construction_steps &&
                prefix.completed_ls_evaluations + suffix.completed_ls_evaluations == reference.completed_ls_evaluations &&
                prefix.total_tour_evaluations + suffix.total_tour_evaluations == reference.total_tour_evaluations,
                "切断后的总工作量或FE改变");
            for (Node i = 0; i < cut; ++i) { trace_equal(prefix.control_trace[i], reference.control_trace[i]); ++compared_batches; }
            for (Node i = cut; i < batches; ++i) { trace_equal(suffix.control_trace[i-cut], reference.control_trace[i]); ++compared_batches; }
            final_equal(suffix, reference); check(restored.serialize() == bytes, "恢复修改了输入快照");
            if (cut == 4) fork_state = saved;
        }
        // 基线同样恢复原批次编号；周期重启不能从零重新计数。
        BaselinePolicy periodic = continuation; periodic.restart_mode = StaticRestart::Periodic; periodic.restart_period = 2;
        const auto base_reference = engine.evaluate_baseline_evaluations(tasks, ants * batches, periodic,
            PreparationMode::CachedCharged, UINT32_MAX, trace);
        CountedState base_state;
        engine.capture_baseline_state(tasks, ants * batches, ants * 3, periodic, base_state);
        const auto base_suffix = engine.continue_baseline_state(base_state, ants * 4, periodic, {}, trace);
        final_equal(base_suffix, base_reference);
        for (Node i = 3; i < batches; ++i) { trace_equal(base_suffix.control_trace[i-3], base_reference.control_trace[i]); ++compared_batches; }

        const auto original_bytes = fork_state.serialize();
        std::set<Node> starts;
        for (Node region = 0; region < 4; ++region) for (std::uint64_t seed : {11ULL, 23ULL, 47ULL}) {
            const ForkIntervention small{true, seed, region, 2}, large{true, seed, region, 16};
            const auto a = engine.continue_baseline_state(fork_state, ants * 3, continuation, small, trace);
            const auto b = engine.continue_baseline_state(fork_state, ants * 3, continuation, large, trace);
            const auto again = engine.continue_baseline_state(fork_state, ants * 3, continuation, small, trace);
            replay_equal(a, again); check(fork_state.serialize() == original_bytes, "A-B-A改写了源快照");
            const auto& x = a.control_trace.front(); const auto& y = b.control_trace.front();
            state_equal(x.before, y.before); regions_equal(x.regions, y.regions);
            state_equal(x.after_restart, y.after_restart, false);
            check(a.total_tour_evaluations == b.total_tour_evaluations && a.total_tour_evaluations == ants * 3 * colonies &&
                a.charged_seconds == 0 && a.overrun_seconds == 0 && a.discarded_batches == 0 &&
                x.features == y.features && x.intervened && y.intervened, "配对初态、FE或次数资源语义不符");
            for (Node c = 0; c < colonies; ++c) {
                check(x.actions[c] == static_cast<std::int32_t>(region * 4) && y.actions[c] == static_cast<std::int32_t>(region * 4 + 3),
                    "干预没有保持reference或MNE错误");
                for (Node ant = 0; ant < ants; ++ant) {
                    check(x.after_restart.targets[c*ants+ant] == 2 && y.after_restart.targets[c*ants+ant] == 16,
                        "干预MNE没有进入实际ant目标");
                    check(x.after_batch.info[c*ants+ant].start_node == y.after_batch.info[c*ants+ant].start_node,
                        "配对随机起点不同");
                    starts.insert(x.after_batch.info[c*ants+ant].start_node);
                }
            }
            for (std::size_t i = 1; i < a.control_trace.size(); ++i)
                check(!a.control_trace[i].intervened && !b.control_trace[i].intervened &&
                    a.control_trace[i].actions == b.control_trace[i].actions,
                    "干预持续超过一个动作或继续策略不同");
            // 单colony提取须处理feature-major布局，并保留该成员的所有随机域。
            FacoBatchEngine solo(n, 1, settings); register_all(solo);
            const auto selected = fork_state.select_colony(2);
            const auto one = solo.continue_baseline_state(selected, ants * 3, continuation, small, trace);
            check(one.incumbents[0].tour == a.incumbents[2].tour && one.incumbents[0].cost == a.incumbents[2].cost,
                "单colony提取改变配对结果");
            for (Node i = 0; i < 3; ++i) {
                controller_equal(one.control_trace[i].after_batch.controls[0], a.control_trace[i].after_batch.controls[2]);
                for (Node f = 0; f < 12; ++f) for (Node action = 0; action < 32; ++action)
                    check(one.control_trace[i].features[f*32+action] == a.control_trace[i].features[(f*colonies+2)*32+action],
                        "单colony特征切片错误");
            }
        }
        check(starts.size() > 1, "多个分叉seed未覆盖不同随机起点");
        auto corrupted = original_bytes; corrupted.pop_back(); invalid([&] { CountedState::deserialize(corrupted); });
        auto missing = fork_state; missing.buffers.pop_back(); missing.seal();
        invalid([&] { engine.continue_program_state(missing, ants, gp); });
        auto wrong = fork_state; wrong.settings.retention = 0.6; wrong.seal();
        invalid([&] { engine.continue_program_state(wrong, ants, gp); });
        invalid([&] { engine.continue_program_state(fork_state, 1, gp); });
        invalid([&] { engine.continue_program_state(fork_state, ants * 4, gp); });
        invalid([&] { engine.continue_baseline_state(fork_state, ants, continuation, {true,1,0,4}); });
        invalid([&] { engine.continue_baseline_state(fork_state, 0, continuation, {true,1,0,2}); });
        invalid([&] { fork_state.select_colony(colonies); });
        FacoBatchEngine other_data(n, colonies, settings);
        auto changed_xy = xy[0]; changed_xy[0] += 0.01;
        other_data.register_problem(1, changed_xy); other_data.register_problem(2, xy[1]);
        invalid([&] { other_data.continue_program_state(fork_state, ants, gp); });
        auto delayed_trace = trace; delayed_trace.delay_batch = 4; delayed_trace.completion_delay_ms = 100;
        const ForkIntervention choice{true,17,0,16};
        const auto no_delay = engine.continue_baseline_state(fork_state, ants * 3, continuation, choice, trace);
        const auto delay = engine.continue_baseline_state(fork_state, ants * 3, continuation, choice, delayed_trace);
        replay_equal(no_delay, delay); check(delay.actual_seconds >= 0.1, "延迟诊断没有执行");
        std::ofstream out(argc > 1 ? argv[1] : "state_fork_results.json");
        check(static_cast<bool>(out), "不能写状态分叉验收记录");
        out << "{\"status\":\"passed\",\"cut_points\":4,\"compared_source_batches\":" << compared_batches
            << ",\"paired_interventions\":12,\"single_colony_pairs\":12,\"buffers\":44,\"snapshot_bytes\":"
            << original_bytes.size() << ",\"formal_mechanism_result\":false}\n";
        std::cout << "counted full-state restoration and paired intervention: passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
