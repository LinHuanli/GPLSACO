// 手算边界：完整32动作、周期零点、概率严格比较、停滞升级和跨epoch冷却。
#include "gp_faco/baseline_policy.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace gp_faco;
std::uint64_t checks = 0;
void check(bool ok) { ++checks; if (!ok) throw std::runtime_error("基线手算动作或输入契约不符"); }
void invalid(BaselinePolicy p, std::uint32_t mask = UINT32_MAX) {
    bool rejected = false;
    try { validate_baseline(p, mask); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);
}
}

int main(int argc, char** argv) {
    try {
        for (Node region = 0; region < 4; ++region) for (Node level = 0; level < 4; ++level) {
            BaselinePolicy p; p.region = region; p.mne_level = p.max_mne_level = level;
            validate_baseline(p);
            FeedbackState feedback; feedback.stagnant_batches = UINT64_MAX; feedback.epoch_batches = 93;
            const Node keep = region * 4 + level;
            check(baseline_action(p, feedback, 23, 0, UINT32_MAX) == keep);
            p.restart_mode = StaticRestart::Bernoulli; p.restart_probability = 1;
            validate_baseline(p);
            check(baseline_action(p, feedback, 23, 0.999999, UINT32_MAX) == keep + 16);
            check(baseline_action(p, feedback, 23, 0, 0xffffu) == keep);
            invalid(p, UINT32_MAX ^ (1u << keep));
        }
        BaselinePolicy periodic; periodic.restart_mode = StaticRestart::Periodic; periodic.restart_period = 3;
        validate_baseline(periodic);
        for (Node batch : {0u, 1u, 2u, 4u, 5u}) check(baseline_action(periodic, {}, batch, 0, UINT32_MAX) == 0);
        for (Node batch : {3u, 6u, 9u}) check(baseline_action(periodic, {}, batch, 0, UINT32_MAX) == 16);
        periodic.restart_period = 0; invalid(periodic);
        BaselinePolicy bernoulli; bernoulli.restart_mode = StaticRestart::Bernoulli; bernoulli.restart_probability = 0.5;
        validate_baseline(bernoulli);
        check(baseline_action(bernoulli, {}, 0, 0.499999, UINT32_MAX) == 16);
        check(baseline_action(bernoulli, {}, 0, 0.5, UINT32_MAX) == 0);
        bernoulli.restart_probability = 0;
        check(baseline_action(bernoulli, {}, 0, 0, UINT32_MAX) == 0);

        BaselinePolicy rule; rule.kind = BaselineKind::Rule; rule.region = 2;
        rule.mne_level = 1; rule.max_mne_level = 3; rule.stagnation_step = 4;
        rule.restart_stagnation = 9; rule.restart_cooldown = 3;
        validate_baseline(rule);
        struct Case { std::uint64_t stagnant, epoch; Node expected; };
        for (auto row : {Case{0,0,9}, {3,3,9}, {4,4,10}, {7,7,10}, {8,8,11},
                         {9,2,11}, {9,3,27}, {UINT64_MAX,UINT64_MAX,27}}) {
            FeedbackState f; f.stagnant_batches = row.stagnant; f.epoch_batches = row.epoch;
            check(baseline_action(rule, f, 0, 0, UINT32_MAX) == row.expected);
        }
        FeedbackState feedback; feedback.stagnant_batches = 9; feedback.epoch_batches = 9;
        restart_feedback(feedback);
        check(feedback.stagnant_batches == 9 && feedback.epoch_batches == 0);
        check(baseline_action(rule, feedback, 100, 0, UINT32_MAX) == 11);
        update_feedback(feedback, true, 0, 0, 1);
        check(baseline_action(rule, feedback, 101, 0, UINT32_MAX) == 9);
        rule.restart_cooldown = 0; invalid(rule);
        rule.restart_stagnation = 0; validate_baseline(rule);
        rule.stagnation_step = 0; invalid(rule);
        rule.max_mne_level = 1; validate_baseline(rule);
        rule.region = 4; invalid(rule);
        rule.region = 0; rule.kind = static_cast<BaselineKind>(99); invalid(rule);
        BaselinePolicy bad; bad.restart_probability = std::numeric_limits<double>::quiet_NaN(); invalid(bad);
        bad.restart_probability = 0; bad.restart_mode = static_cast<StaticRestart>(99); invalid(bad);
        bad.restart_mode = StaticRestart::None; bad.mne_level = 4; invalid(bad);
        std::ofstream output(argc > 1 ? argv[1] : "baseline_policy_results.json");
        check(static_cast<bool>(output));
        output << "{\"status\":\"passed\",\"checks\":" << checks << ",\"action_ids\":32}\n";
        std::cout << "baseline policy hand-derived boundary checks: " << checks << " passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
