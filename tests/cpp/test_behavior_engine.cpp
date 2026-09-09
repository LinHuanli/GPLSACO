// 完整tour边集oracle独立于设备位置差计数；记录开关不能影响任何已定义求解状态。
#include "gp_faco/batch_engine.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace gp_faco;
namespace {
std::uint64_t row_checks = 0, edge_checks = 0, collision_differences = 0, pairs = 0, restarts = 0;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
auto feedback(const FeedbackState& f) {
    return std::make_tuple(f.return_rate, f.ls_work, f.stagnant_batches, f.epoch_batches, f.restarts);
}
auto work(const FacoDiagnosticInfo& x) {
    return std::make_tuple(x.construction_cost, x.final_cost, x.construction.mne, x.construction.steps,
        x.construction.nonidentity_relocations, x.construction.legal_exhausted, x.start_node,
        x.checklist_size, x.local_search.processed_nodes, x.local_search.candidate_checks,
        x.local_search.move_evaluations, x.local_search.accepted_moves, x.local_search.reactivations,
        x.local_search.constraint_rejections, x.local_search.evaluation_limit_reached);
}
void same_state(const ControlDeviceSnapshot& a, const ControlDeviceSnapshot& b) {
    check(a.parent == b.parent && a.parent_positions == b.parent_positions && a.epoch == b.epoch &&
        a.global == b.global && a.archive == b.archive && a.archive_positions == b.archive_positions &&
        a.targets == b.targets && a.tours == b.tours && a.positions == b.positions &&
        a.per_ant_parent_positions == b.per_ant_parent_positions && a.visited == b.visited &&
        a.trails == b.trails && a.products == b.products && a.ant_identities == b.ant_identities,
        "观测改变tour/信息素/参考/档案/蚂蚁状态");
    for (std::size_t i = 0; i < a.controls.size(); ++i) {
        const auto& x = a.controls[i]; const auto& y = b.controls[i];
        check(feedback(x.feedback) == feedback(y.feedback) && x.archive_size == y.archive_size &&
            x.tracked_global == y.tracked_global && x.tracked_epoch == y.tracked_epoch &&
            x.active_identity == y.active_identity && x.epoch_identity == y.epoch_identity,
            "观测改变控制器状态");
        for (Node slot = 0; slot < archive_capacity; ++slot)
            check(x.archive_cost[slot] == y.archive_cost[slot] &&
                x.archive_identity[slot] == y.archive_identity[slot], "观测改变档案元数据");
        const auto& u = a.colonies[i]; const auto& v = b.colonies[i];
        check(std::tie(u.global_cost,u.epoch_cost,u.parent_cost,u.minimum,u.maximum,u.default_trail,
                       u.source_uniform,u.iteration_best,u.source_is_epoch) ==
              std::tie(v.global_cost,v.epoch_cost,v.parent_cost,v.minimum,v.maximum,v.default_trail,
                       v.source_uniform,v.iteration_best,v.source_is_epoch), "观测改变来源随机流");
    }
    for (std::size_t i = 0; i < a.info.size(); ++i) check(work(a.info[i]) == work(b.info[i]), "内部工作改变");
}
void same_result(const BatchEvaluation& a, const BatchEvaluation& b) {
    check(a.total_tour_evaluations == b.total_tour_evaluations &&
          a.completed_construction_steps == b.completed_construction_steps &&
          a.completed_ls_evaluations == b.completed_ls_evaluations &&
          a.control_trace.size() == b.control_trace.size(), "FE/工作账目改变");
    for (std::size_t i = 0; i < a.incumbents.size(); ++i)
        check(a.incumbents[i].tour == b.incumbents[i].tour && a.incumbents[i].cost == b.incumbents[i].cost,
              "最终解改变");
    for (std::size_t i = 0; i < a.control_trace.size(); ++i) {
        const auto& x = a.control_trace[i]; const auto& y = b.control_trace[i];
        check(x.actions == y.actions && x.masks == y.masks && x.legal_masks == y.legal_masks &&
              x.features == y.features && x.scores == y.scores && x.baseline_uniforms == y.baseline_uniforms &&
              x.alternatives == y.alternatives, "动作/评分/特征/基线随机值改变");
        same_state(x.after_batch, y.after_batch);
        check(x.after_restart.parent == y.after_restart.parent && x.after_restart.trails == y.after_restart.trails
              && x.after_restart.products == y.after_restart.products, "重启事务改变");
    }
    ++pairs;
}
using Edges = std::set<std::pair<Node, Node>>;
Edges edges(const std::vector<Node>& tours, std::size_t offset, Node n) {
    Edges result; std::set<Node> nodes;
    for (Node i = 0; i < n; ++i) {
        const Node a = tours[offset + i], b = tours[offset + (i + 1) % n];
        check(a < n && nodes.insert(a).second, "oracle收到非法排列");
        result.insert(std::minmax(a, b));
    }
    check(result.size() == n, "oracle边数不完整");
    return result;
}
Node difference(const Edges& tour, const Edges& reference) {
    Node result = 0;
    for (const auto& edge : tour) result += reference.count(edge) == 0;
    return result;
}
void verify(const BatchEvaluation& result, Node n, Node ants, Node colonies, bool has_baseline) {
    check(result.behavior_recorded && result.behavior_rows.size() == result.completed_batches * colonies &&
          result.behavior_device_bytes == colonies * sizeof(BatchBehaviorRow) + 2 * ants * colonies * sizeof(Node),
          "观测行/内存账目错误");
    for (const auto& row : result.behavior_rows) {
        const auto& t = result.control_trace[row.batch]; const Node c = row.colony;
        const auto& before = t.before.colonies[c]; const auto& used = t.after_restart.colonies[c];
        const auto& after = t.after_batch.colonies[c];
        check(row.ants == ants && row.dimension == n && row.action == t.actions[c] &&
            row.alternative == t.alternatives[c] && row.action_mask == t.masks[c] &&
            row.legal_mask == (t.legal_masks.empty() ? t.masks[c] : t.legal_masks[c]) &&
            row.global_before == before.global_cost && row.reference_before == before.parent_cost &&
            row.reference_used == used.parent_cost && row.global_after == after.global_cost &&
            row.iteration_best_cost == t.after_batch.info[c * ants + after.iteration_best].final_cost &&
            feedback(row.feedback_before) == feedback(t.before.controls[c].feedback) &&
            feedback(row.feedback_after) == feedback(t.after_batch.controls[c].feedback), "批次原始观测不符");
        check((row.baseline_requested_action >= 0) == has_baseline, "基线请求适用范围错误");
        BatchBehaviorRow expected;
        const auto reference = edges(t.after_restart.parent, static_cast<std::size_t>(c) * n, n);
        for (Node ant = c * ants; ant < (c + 1) * ants; ++ant) {
            const auto base = static_cast<std::size_t>(ant) * n;
            const auto constructed = edges(t.construction_tours, base, n);
            const auto final = edges(t.after_batch.tours, base, n);
            expected.construction_new_edges += difference(constructed, reference);
            expected.final_new_edges += difference(final, reference);
            expected.exact_returns += final == reference;
            expected.fingerprint_returns += t.after_batch.ant_identities[ant] ==
                t.after_restart.controls[c].active_identity;
            const auto& info = t.after_batch.info[ant];
            expected.construction_mne += info.construction.mne;
            expected.construction_steps += info.construction.steps;
            expected.construction_relocations += info.construction.nonidentity_relocations;
            expected.construction_exhausted_ants += info.construction.legal_exhausted;
            expected.ls_move_evaluations += info.local_search.move_evaluations;
            expected.ls_accepted_moves += info.local_search.accepted_moves;
            expected.ls_limit_reached_ants += info.local_search.evaluation_limit_reached;
            edge_checks += 2;
        }
        check(row.construction_new_edges == expected.construction_new_edges &&
              row.final_new_edges == expected.final_new_edges && row.exact_returns == expected.exact_returns &&
              row.fingerprint_returns == expected.fingerprint_returns, "独立完整边集/指纹oracle不符");
        check(row.construction_mne == expected.construction_mne && row.construction_steps == expected.construction_steps
              && row.construction_relocations == expected.construction_relocations &&
              row.construction_exhausted_ants == expected.construction_exhausted_ants &&
              row.ls_move_evaluations == expected.ls_move_evaluations &&
              row.ls_accepted_moves == expected.ls_accepted_moves &&
              row.ls_limit_reached_ants == expected.ls_limit_reached_ants, "蚂蚁累计工作oracle不符");
        collision_differences += row.fingerprint_returns != row.exact_returns;
        restarts += row.action >= 16;
        ++row_checks;
    }
}
}

int main(int argc, char** argv) {
    try {
        constexpr Node n = 31, ants = 8, colonies = 4, batches = 12;
        FixedFacoSettings settings; settings.ants = ants;
        FacoBatchEngine engine(n, colonies, settings);
        std::vector<double> xy(2 * n), tied(2 * n); tied[0] = 1;
        for (Node i = 0; i < 2 * n; ++i) xy[i] = (control_mix(i + 917) >> 11) * 0x1p-53;
        engine.register_problem(1, xy); engine.register_problem(2, tied);
        const std::vector<BatchTask> tasks{{1,17},{1,29},{2,17},{2,29}};
        Program program; program.feature_spec_id = 2; program.length = 3;
        program.opcode[2] = 2; program.operand[0] = 4; program.operand[1] = 5;
        BaselinePolicy none; none.region = 1;
        auto periodic = none; periodic.restart_mode = StaticRestart::Periodic; periodic.restart_period = 3;
        auto random = none; random.restart_mode = StaticRestart::Bernoulli; random.restart_probability = 0.5;
        auto rule = none; rule.kind = BaselineKind::Rule; rule.max_mne_level = 3; rule.stagnation_step = 2;
        rule.restart_stagnation = 3; rule.restart_cooldown = 3;
        BatchDiagnosticControls off; off.capture_control = true;
        auto on = off; on.record_behavior = true;
        for (const auto& base : {none, periodic, random, rule})
            for (auto variant : {FactorialVariant::M00,FactorialVariant::M10,
                                 FactorialVariant::M01,FactorialVariant::M11})
                for (auto mask : {UINT32_MAX, 0xffffu}) {
                    const FactorialPolicy policy{variant, base};
                    const auto original = engine.evaluate_factorial_evaluations(tasks, ants * batches, program,
                        policy, PreparationMode::CachedCharged, mask, off);
                    const auto recorded = engine.evaluate_factorial_evaluations(tasks, ants * batches, program,
                        policy, PreparationMode::CachedCharged, mask, on);
                    same_result(original, recorded);
                    verify(recorded, n, ants, colonies, variant != FactorialVariant::M11);
                    check(!original.behavior_recorded && original.behavior_rows.empty() &&
                          original.behavior_device_bytes == 0, "未请求观测仍返回记录");
                }
        check(collision_differences == 0, "普通诊断出现指纹/边集差异");
        for (auto variant : {FactorialVariant::M00,FactorialVariant::M10,
                             FactorialVariant::M01,FactorialVariant::M11}) {
            const FactorialPolicy policy{variant, periodic};
            auto forced_on = on, forced_off = off;
            forced_on.force_fingerprint_collisions = forced_off.force_fingerprint_collisions = true;
            const auto original = engine.evaluate_factorial_evaluations(tasks, ants * batches, program, policy,
                PreparationMode::CachedCharged, UINT32_MAX, forced_off);
            const auto recorded = engine.evaluate_factorial_evaluations(tasks, ants * batches, program, policy,
                PreparationMode::CachedCharged, UINT32_MAX, forced_on);
            same_result(original, recorded); verify(recorded, n, ants, colonies, variant != FactorialVariant::M11);
        }
        check(collision_differences > 0 && restarts > 0, "强制碰撞/实际重启覆盖不足");
        const FactorialPolicy policy{FactorialVariant::M10, rule};
        const auto expected = engine.evaluate_factorial_evaluations(tasks, ants * batches, program, policy,
            PreparationMode::CachedCharged, UINT32_MAX, on);
        for (bool profiling : {false, true}) {
            auto altered = on; altered.profile = profiling; altered.completion_delay_ms = 20;
            const auto actual = engine.evaluate_factorial_evaluations(tasks, ants * batches, program, policy,
                PreparationMode::EndToEnd, UINT32_MAX, altered);
            same_result(expected, actual); verify(actual, n, ants, colonies, true);
        }
        const auto zero = engine.evaluate_factorial_evaluations(tasks, 0, program, policy,
            PreparationMode::CachedCharged, UINT32_MAX, on);
        check(zero.behavior_recorded && zero.behavior_rows.empty() && zero.total_tour_evaluations == 0,
              "零FE行为账目错误");
        auto legacy = program; legacy.feature_spec_id = 1;
        bool rejected = false;
        try { engine.evaluate_program_diagnostic(tasks, 1, legacy, PreparationMode::CachedCharged, UINT32_MAX, on); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "墙钟入口未拒绝行为记录");
        std::ofstream output(argc > 1 ? argv[1] : "behavior_engine_results.json");
        check(static_cast<bool>(output), "不能写验收记录");
        output << "{\"status\":\"passed\",\"trajectory_pairs\":" << pairs << ",\"row_checks\":" << row_checks
               << ",\"edge_set_checks\":" << edge_checks << ",\"collision_difference_rows\":" << collision_differences
               << ",\"restart_rows\":" << restarts << "}\n";
        std::cout << "behavior full-state and independent edge-set checks passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
