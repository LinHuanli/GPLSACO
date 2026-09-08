// 基线与等动作GP的完整轨迹对照；实际规则冷却和随机流独立性另行验证。
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
std::uint64_t gp_pairs = 0, shape_pairs = 0, action_checks = 0, restart_checks = 0;
std::set<Node> covered_actions;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
auto feedback(const FeedbackState& f) {
    return std::make_tuple(f.return_rate, f.ls_work, f.stagnant_batches, f.epoch_batches, f.restarts);
}
void identical(const BatchEvaluation& a, const BatchEvaluation& b) {
    check(a.completed_batches == b.completed_batches && a.total_tour_evaluations == b.total_tour_evaluations &&
        a.completed_construction_steps == b.completed_construction_steps &&
        a.completed_ls_evaluations == b.completed_ls_evaluations, "等动作控制改变FE/内部工作量");
    check(a.control_trace.size() == b.control_trace.size(), "完整轨迹数量不符");
    for (std::size_t i = 0; i < a.incumbents.size(); ++i) {
        check(a.incumbents[i].tour == b.incumbents[i].tour && a.incumbents[i].cost == b.incumbents[i].cost,
            "等动作控制改变最终tour");
        check(feedback(a.completed_control_states[i].feedback) == feedback(b.completed_control_states[i].feedback),
            "最终反馈状态不符");
    }
    for (std::size_t i = 0; i < a.control_trace.size(); ++i) {
        const auto& x = a.control_trace[i]; const auto& y = b.control_trace[i];
        check(x.actions == y.actions && x.features == y.features && x.masks == y.masks, "等动作选择/特征不符");
        check(x.after_batch.archive == y.after_batch.archive && x.after_batch.trails == y.after_batch.trails &&
            x.after_batch.products == y.after_batch.products && x.after_batch.tours == y.after_batch.tours &&
            x.after_restart.tours == y.after_restart.tours && x.after_restart.visited == y.after_restart.visited,
            "共同档案、信息素、重启或蚂蚁轨迹被改变");
        for (std::size_t j = 0; j < x.after_batch.colonies.size(); ++j)
            check(x.after_batch.colonies[j].source_uniform == y.after_batch.colonies[j].source_uniform,
                "基线随机数扰动了来源随机流");
    }
}
void verify_actions(const BatchEvaluation& result, const BaselinePolicy& policy, Node colonies, Node ants, Node batches) {
    check(result.count_limited && result.completed_batches == batches && result.discarded_batches == 0 &&
        result.charged_seconds == 0 && result.overrun_seconds == 0 &&
        result.total_tour_evaluations == static_cast<std::uint64_t>(colonies)*ants*batches, "基线FE账目错误");
    std::vector<std::int64_t> last_restart(colonies, -1);
    for (const auto& trace : result.control_trace) {
        check(trace.scores.empty() && trace.baseline_uniforms.size() == colonies, "基线误用旧GP评分缓冲");
        for (Node colony = 0; colony < colonies; ++colony) {
            const double u = trace.baseline_uniforms[colony];
            check(u >= 0 && u < 1, "基线随机值越界");
            const auto& before = trace.before.controls[colony].feedback;
            const Node expected = baseline_action(policy, before, trace.batch, u, trace.masks[colony]);
            check(trace.actions[colony] == static_cast<std::int32_t>(expected), "CPU/GPU基线决策不符");
            ++action_checks; covered_actions.insert(expected);
            if (expected >= 16) {
                ++restart_checks;
                const auto& after = trace.after_restart.controls[colony].feedback;
                check(after.stagnant_batches == before.stagnant_batches && after.epoch_batches == 0 &&
                    after.return_rate == 0 && after.ls_work == 0 && after.restarts == before.restarts + 1,
                    "规则重启未复用完整反馈事务");
                if (policy.kind == BaselineKind::Rule && last_restart[colony] >= 0)
                    check(trace.batch - last_restart[colony] >= policy.restart_cooldown, "实际重启违反epoch冷却");
                last_restart[colony] = trace.batch;
            }
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        constexpr Node ants = 8, colonies = 4, batches = 8;
        FixedFacoSettings settings; settings.ants = ants;
        FacoBatchEngine engine(31, colonies, settings), single(31, 1, settings);
        // 相同正成本的不同tour确保档案有合法替代且GB不改善，稳定触发停滞阈值。
        std::vector<double> xy(62); xy[0] = 1;
        engine.register_problem(1, xy); single.register_problem(1, xy);
        const std::vector<BatchTask> tasks{{1,17},{1,29},{1,41},{1,53}};
        BatchDiagnosticControls diagnostic; diagnostic.capture_control = true;
        Program restart_score; restart_score.feature_spec_id = 2; restart_score.length = 1;
        restart_score.operand[0] = 4;
        for (Node region = 0; region < 4; ++region) for (Node level = 0; level < 4; ++level)
            for (bool restart : {false, true}) {
                BaselinePolicy policy; policy.region = region; policy.mne_level = policy.max_mne_level = level;
                policy.restart_mode = StaticRestart::Bernoulli; policy.restart_probability = restart ? 1 : 0;
                const Node keep = region*4 + level;
                const auto mask = (1u << keep) | (restart ? 1u << (keep+16) : 0u);
                const auto baseline = engine.evaluate_baseline_evaluations(tasks, ants*batches, policy,
                    PreparationMode::CachedCharged, mask, diagnostic);
                const auto gp = engine.evaluate_program_evaluations(tasks, ants*batches, restart_score,
                    PreparationMode::CachedCharged, mask, diagnostic);
                // 相同mask把GP限制为已知动作，独立基线的完整底座行为必须一致。
                identical(baseline, gp); ++gp_pairs;
                verify_actions(baseline, policy, colonies, ants, batches);
            }
        BaselinePolicy rule; rule.kind = BaselineKind::Rule; rule.max_mne_level = 3;
        rule.stagnation_step = 2; rule.restart_stagnation = 3; rule.restart_cooldown = 3;
        BaselinePolicy periodic; periodic.restart_mode = StaticRestart::Periodic; periodic.restart_period = 3;
        BaselinePolicy random; random.restart_mode = StaticRestart::Bernoulli; random.restart_probability = 0.5;
        bool random_kept = false, random_restarted = false;
        for (const auto& policy : {rule, periodic, random}) {
            const auto result = engine.evaluate_baseline_evaluations(tasks, ants*12, policy,
                PreparationMode::CachedCharged, UINT32_MAX, diagnostic);
            verify_actions(result, policy, colonies, ants, 12);
            for (const auto& trace : result.control_trace) for (Node colony = 0; colony < colonies; ++colony)
                if (policy.restart_mode == StaticRestart::Bernoulli && trace.masks[colony] & (1u << 16)) {
                    random_kept |= trace.actions[colony] == 0;
                    random_restarted |= trace.actions[colony] == 16;
                }
            auto delayed = diagnostic; delayed.completion_delay_ms = 100; delayed.delay_batch = 0;
            identical(result, engine.evaluate_baseline_evaluations(tasks, ants*12, policy,
                PreparationMode::EndToEnd, UINT32_MAX, delayed));
            auto profiled = diagnostic; profiled.profile = true;
            identical(result, engine.evaluate_baseline_evaluations(tasks, ants*12, policy,
                PreparationMode::CachedCharged, UINT32_MAX, profiled));
            for (Node colony = 0; colony < colonies; ++colony) {
                const auto solo = single.evaluate_baseline_evaluations({tasks[colony]}, ants*12, policy,
                    PreparationMode::CachedCharged, UINT32_MAX, diagnostic);
                check(solo.incumbents[0].tour == result.incumbents[colony].tour &&
                    feedback(solo.completed_control_states[0].feedback) == feedback(result.completed_control_states[colony].feedback),
                    "基线单/批量最终状态不符");
                for (Node batch = 0; batch < 12; ++batch) {
                    check(solo.control_trace[batch].actions[0] == result.control_trace[batch].actions[colony] &&
                        solo.control_trace[batch].baseline_uniforms[0] == result.control_trace[batch].baseline_uniforms[colony],
                        "形状改变基线随机流或动作");
                }
                ++shape_pairs;
            }
        }
        check(random_kept && random_restarted && restart_checks > 0 && covered_actions.size() == 32,
            "真实动作/概率分支覆盖不足");
        const auto zero = engine.evaluate_baseline_evaluations(tasks, 0, rule, PreparationMode::CachedCharged);
        check(zero.total_tour_evaluations == 0 && zero.preparation_completed, "基线零FE错误");
        std::ofstream output(argc > 1 ? argv[1] : "baseline_engine_results.json");
        check(static_cast<bool>(output), "不能保存基线验收记录");
        output << "{\"status\":\"passed\",\"gp_equivalence_pairs\":" << gp_pairs <<
            ",\"shape_pairs\":" << shape_pairs << ",\"action_checks\":" << action_checks <<
            ",\"restart_checks\":" << restart_checks << ",\"covered_actions\":" << covered_actions.size() << "}\n";
        std::cout << "baseline shared-engine action, transaction and RNG checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
