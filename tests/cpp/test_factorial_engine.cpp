// 独立枚举动作组合，并对照CPU评分、两个原入口及实际控制/随机轨迹。
#include "gp_faco/batch_engine.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {
using namespace gp_faco;
std::uint64_t mask_checks = 0, action_checks = 0, endpoint_pairs = 0, replay_pairs = 0,
    shape_pairs = 0, restart_checks = 0, missing_alt = 0;
std::set<Node> actions_seen, levels_seen;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

// CPU oracle直接枚举可行(b,j,level)，不调用生产baseline_action/factorial_mask。
std::uint32_t expected_mask(const FactorialPolicy& p, const FeedbackState& f, Node batch,
                           double u, std::uint32_t legal) {
    const auto& b = p.baseline;
    Node level = b.mne_level;
    bool restart = false;
    if (b.kind == BaselineKind::Static) {
        if (b.restart_mode == StaticRestart::Periodic)
            restart = batch != 0 && batch % b.restart_period == 0;
        if (b.restart_mode == StaticRestart::Bernoulli) restart = u < b.restart_probability;
    } else {
        while (level < b.max_mne_level &&
               f.stagnant_batches >= static_cast<std::uint64_t>(level-b.mne_level+1)*b.stagnation_step)
            ++level;
        restart = b.restart_stagnation != 0 && f.stagnant_batches >= b.restart_stagnation &&
            f.epoch_batches >= b.restart_cooldown;
    }
    const Node local = 4*b.region+level;
    if (p.variant == FactorialVariant::M00) restart &= (legal & (1u << (16+local))) != 0;
    else if (p.variant == FactorialVariant::M10) restart &= (legal & 0xffff0000u) != 0;
    std::uint32_t result = 0;
    for (Node action = 0; action < 32; ++action) {
        if (!(legal & (1u << action))) continue;
        if ((p.variant == FactorialVariant::M00 || p.variant == FactorialVariant::M01) &&
            action % 16 != local) continue;
        if ((p.variant == FactorialVariant::M00 || p.variant == FactorialVariant::M10) &&
            static_cast<bool>(action/16) != restart) continue;
        result |= 1u << action;
    }
    return result;
}

auto feedback(const FeedbackState& f) {
    return std::make_tuple(f.return_rate, f.ls_work, f.stagnant_batches, f.epoch_batches, f.restarts);
}
void same_control(const ControllerState& a, const ControllerState& b) {
    check(feedback(a.feedback) == feedback(b.feedback) && a.archive_size == b.archive_size &&
        a.tracked_global == b.tracked_global && a.tracked_epoch == b.tracked_epoch &&
        a.active_identity == b.active_identity && a.epoch_identity == b.epoch_identity,
        "控制/反馈状态改变");
    for (Node i = 0; i < archive_capacity; ++i)
        check(a.archive_cost[i] == b.archive_cost[i] && a.archive_identity[i] == b.archive_identity[i],
              "档案成本/指纹改变");
}
void same_snapshot(const ControlDeviceSnapshot& a, const ControlDeviceSnapshot& b) {
    check(a.parent == b.parent && a.parent_positions == b.parent_positions && a.epoch == b.epoch &&
        a.global == b.global && a.archive == b.archive && a.archive_positions == b.archive_positions &&
        a.targets == b.targets && a.tours == b.tours && a.positions == b.positions &&
        a.per_ant_parent_positions == b.per_ant_parent_positions && a.visited == b.visited &&
        a.trails == b.trails && a.products == b.products && a.ant_identities == b.ant_identities,
        "参考/档案/蚂蚁/信息素状态改变");
    for (std::size_t i = 0; i < a.controls.size(); ++i) {
        same_control(a.controls[i], b.controls[i]);
        const auto& x = a.colonies[i]; const auto& y = b.colonies[i];
        check(std::tie(x.global_cost,x.epoch_cost,x.parent_cost,x.minimum,x.maximum,x.default_trail,
                      x.source_uniform,x.iteration_best,x.source_is_epoch) ==
              std::tie(y.global_cost,y.epoch_cost,y.parent_cost,y.minimum,y.maximum,y.default_trail,
                      y.source_uniform,y.iteration_best,y.source_is_epoch), "colony状态/来源随机流改变");
    }
}
void identical(const BatchEvaluation& a, const BatchEvaluation& b) {
    check(a.total_tour_evaluations == b.total_tour_evaluations &&
        a.completed_batches == b.completed_batches &&
        a.completed_construction_steps == b.completed_construction_steps &&
        a.completed_ls_evaluations == b.completed_ls_evaluations &&
        a.control_trace.size() == b.control_trace.size(), "FE/内部工作量改变");
    for (std::size_t i = 0; i < a.incumbents.size(); ++i) {
        check(a.incumbents[i].tour == b.incumbents[i].tour && a.incumbents[i].cost == b.incumbents[i].cost,
              "相同控制改变最终路线");
        same_control(a.completed_control_states[i], b.completed_control_states[i]);
    }
    for (std::size_t i = 0; i < a.control_trace.size(); ++i) {
        const auto& x = a.control_trace[i]; const auto& y = b.control_trace[i];
        check(x.actions == y.actions && x.features == y.features && x.masks == y.masks &&
            x.legal_masks == y.legal_masks && x.scores == y.scores &&
            x.baseline_uniforms == y.baseline_uniforms && x.alternatives == y.alternatives,
            "动作/评分/特征/随机流改变");
        // 第一次before的蚂蚁临时工作区尚未产生语义值；只比较已完成批次的完整状态。
        same_snapshot(x.after_batch, y.after_batch);
        for (std::size_t j = 0; j < x.after_restart.controls.size(); ++j)
            same_control(x.after_restart.controls[j], y.after_restart.controls[j]);
        check(x.after_restart.parent == y.after_restart.parent &&
            x.after_restart.trails == y.after_restart.trails &&
            x.after_restart.products == y.after_restart.products, "重启参考/信息素事务改变");
    }
}
void verify(const BatchEvaluation& result, const FactorialPolicy& p, const Program& program) {
    check(result.count_limited && result.charged_seconds == 0 && result.discarded_batches == 0 &&
        result.overrun_seconds == 0, "析因出现墙钟停止/扣费/丢批");
    for (const auto& t : result.control_trace) {
        std::vector<std::uint32_t> expected;
        for (std::size_t i = 0; i < t.actions.size(); ++i) {
            expected.push_back(expected_mask(p, t.before.controls[i].feedback, t.batch,
                                            t.baseline_uniforms[i], t.legal_masks[i]));
            const Node action = t.actions[i];
            check(expected.back() == t.masks[i] && (expected.back() & (1u << action)),
                  "单因素mask越界或错误");
            ++action_checks; actions_seen.insert(action); levels_seen.insert(action%4);
            if (t.alternatives[i] >= archive_capacity) {
                ++missing_alt; check(action < 16, "无替代解仍然重启");
            }
            if (action >= 16) {
                const auto& f = t.before.controls[i].feedback;
                const auto& after = t.after_restart.controls[i].feedback;
                check(after.stagnant_batches == f.stagnant_batches && after.epoch_batches == 0 &&
                    after.return_rate == 0 && after.ls_work == 0 && after.restarts == f.restarts+1,
                    "析因未复用完整重启反馈事务");
                ++restart_checks;
            }
        }
        const auto cpu = score_cpu(program, t.features, expected);
        check(cpu.scores == t.scores && cpu.actions == t.actions, "独立CPU评分与GPU选择不符");
    }
}
template<class F> void invalid(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "非法析因输入未拒绝");
}
}

int main(int argc, char** argv) {
    try {
        BaselinePolicy none; none.region = 2; none.mne_level = none.max_mne_level = 1;
        auto periodic = none; periodic.restart_mode = StaticRestart::Periodic; periodic.restart_period = 3;
        auto random = none; random.restart_mode = StaticRestart::Bernoulli; random.restart_probability = 0.5;
        BaselinePolicy rule; rule.kind = BaselineKind::Rule; rule.region = 1; rule.max_mne_level = 3;
        rule.stagnation_step = 2; rule.restart_stagnation = 3; rule.restart_cooldown = 3;
        for (const auto& base : {none, periodic, random, rule})
            for (auto v : {FactorialVariant::M00,FactorialVariant::M10,FactorialVariant::M01,FactorialVariant::M11})
                for (Node batch = 0; batch < 9; ++batch)
                    for (Node stagnant : {0u,1u,2u,3u,7u})
                        for (Node epoch : {0u,2u,3u,8u})
                            for (double u : {0.0,0.499999999,0.5,0.999999999})
                                for (auto legal : {UINT32_MAX,0xffffu,0x8000ffffu,0x0200ffffu}) {
                                    FeedbackState f; f.stagnant_batches = stagnant; f.epoch_batches = epoch;
                                    const FactorialPolicy p{v,base};
                                    const auto expected = expected_mask(p,f,batch,u,legal);
                                    check(expected && expected == factorial_mask(p,f,batch,u,legal), "CPU动作组合不符");
                                    ++mask_checks;
                                }
        constexpr Node ants = 8, colonies = 4, batches = 12;
        FixedFacoSettings settings; settings.ants = ants;
        FacoBatchEngine engine(31, colonies, settings), single(31,1,settings);
        std::vector<double> xy(62); xy[0] = 1;
        engine.register_problem(1,xy); single.register_problem(1,xy);
        const std::vector<BatchTask> tasks{{1,17},{1,29},{1,41},{1,53}};
        BatchDiagnosticControls diagnostic; diagnostic.capture_control = true;
        Program restart; restart.feature_spec_id = 2; restart.length = 1; restart.operand[0] = 4;
        Program complex; complex.feature_spec_id = 2; complex.length = 9;
        const std::uint8_t op[]{0,0,2,0,0,3,0,4,2}, arg[]{4,8,0,0,2,0,5,0,0};
        std::copy(std::begin(op),std::end(op),complex.opcode);
        std::copy(std::begin(arg),std::end(arg),complex.operand);
        for (const auto& base : {none,periodic,random,rule}) {
            identical(engine.evaluate_baseline_evaluations(tasks,ants*batches,base,
                          PreparationMode::CachedCharged,UINT32_MAX,diagnostic),
                      engine.evaluate_factorial_evaluations(tasks,ants*batches,complex,
                          {FactorialVariant::M00,base},PreparationMode::CachedCharged,UINT32_MAX,diagnostic));
            identical(engine.evaluate_program_evaluations(tasks,ants*batches,complex,
                          PreparationMode::CachedCharged,UINT32_MAX,diagnostic),
                      engine.evaluate_factorial_evaluations(tasks,ants*batches,complex,
                          {FactorialVariant::M11,base},PreparationMode::CachedCharged,UINT32_MAX,diagnostic));
            endpoint_pairs += 2;
            for (auto v : {FactorialVariant::M10,FactorialVariant::M01})
                for (const auto& program : {restart,complex}) {
                    const FactorialPolicy p{v,base};
                    const auto result = engine.evaluate_factorial_evaluations(tasks,ants*batches,program,p,
                        PreparationMode::CachedCharged,UINT32_MAX,diagnostic);
                    verify(result,p,program);
                    engine.evaluate_baseline_evaluations(tasks,ants*3,random,PreparationMode::CachedCharged);
                    auto delayed = diagnostic; delayed.completion_delay_ms = 20; delayed.delay_batch = 0;
                    identical(result,engine.evaluate_factorial_evaluations(tasks,ants*batches,program,p,
                        PreparationMode::EndToEnd,UINT32_MAX,delayed)); ++replay_pairs;
                    auto profiled = diagnostic; profiled.profile = true;
                    identical(result,engine.evaluate_factorial_evaluations(tasks,ants*batches,program,p,
                        PreparationMode::CachedCharged,UINT32_MAX,profiled)); ++replay_pairs;
                    for (Node i = 0; i < colonies; ++i) {
                        const auto solo = single.evaluate_factorial_evaluations({tasks[i]},ants*batches,
                            program,p,PreparationMode::CachedCharged,UINT32_MAX,diagnostic);
                        check(solo.incumbents[0].tour == result.incumbents[i].tour, "单/批量路线不符");
                        same_control(solo.completed_control_states[0],result.completed_control_states[i]);
                        for (Node batch = 0; batch < batches; ++batch)
                            check(solo.control_trace[batch].actions[0] == result.control_trace[batch].actions[i] &&
                                solo.control_trace[batch].baseline_uniforms[0] ==
                                result.control_trace[batch].baseline_uniforms[i], "单/批量动作/随机流不符");
                        ++shape_pairs;
                    }
                }
        }
        for (auto v : {FactorialVariant::M10,FactorialVariant::M01}) {
            const auto zero = engine.evaluate_factorial_evaluations(tasks,0,complex,{v,rule},
                PreparationMode::CachedCharged);
            check(zero.preparation_completed && zero.total_tour_evaluations == 0, "零FE发生搜索");
        }
        invalid([&] { validate_factorial({static_cast<FactorialVariant>(17),rule}); });
        invalid([&] { validate_factorial({FactorialVariant::M01,rule},1u); });
        validate_factorial({FactorialVariant::M10,rule},1u | (1u << 31));
        validate_factorial({FactorialVariant::M11,rule},1u);
        invalid([&] { engine.evaluate_factorial_evaluations(tasks,ants+1,complex,
            {FactorialVariant::M10,rule},PreparationMode::CachedCharged); });
        check(restart_checks && missing_alt && levels_seen.size() == 4, "真实重启/无alt/档位覆盖不足");
        std::ofstream output(argc > 1 ? argv[1] : "factorial_engine_results.json");
        check(static_cast<bool>(output), "不能保存析因验收记录");
        output << "{\"status\":\"passed\",\"mask_checks\":" << mask_checks <<
            ",\"action_checks\":" << action_checks << ",\"endpoint_pairs\":" << endpoint_pairs <<
            ",\"replay_pairs\":" << replay_pairs << ",\"shape_pairs\":" << shape_pairs <<
            ",\"restart_checks\":" << restart_checks << ",\"missing_alt_checks\":" << missing_alt <<
            ",\"covered_actions\":" << actions_seen.size() << "}\n";
        std::cout << "factorial mask, trajectory, restart and RNG checks passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
