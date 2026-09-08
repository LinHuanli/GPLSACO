// 用独立CPU档案和信息素重放整个GPU控制循环，检查事务边界上的真实状态。
#include "gp_faco/batch_engine.hpp"
#include "gp_faco/control_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {
using namespace gp_faco;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::uint64_t configurations = 0, checked_batches = 0, checked_ants = 0, checked_features = 0,
    checked_restarts = 0, full_identity_comparisons = 0, checked_start_nodes = 0, collision_batches = 0;
double max_feature_error = 0, max_cost_error = 0;
std::uint64_t source_epoch_count = 0, source_iteration_count = 0;

void near(double a, double b, const char* message, double tolerance = 1e-10) {
    check(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tolerance * (1 + std::abs(b)), message);
}
template<class T> void segment(const std::vector<T>& actual, std::size_t begin,
                              const std::vector<T>& expected, const char* message) {
    check(begin + expected.size() <= actual.size() &&
          std::equal(expected.begin(), expected.end(), actual.begin() + begin), message);
}
template<class T> void zero_range(const std::vector<T>& actual, std::size_t begin, std::size_t count) {
    check(begin + count <= actual.size(), "清空检查越界");
    for (std::size_t i = begin; i < begin + count; ++i) check(actual[i] == T{}, "重启残留工作缓冲");
}
std::vector<double> points(Node n, std::uint64_t seed, bool equal_cost) {
    std::vector<double> xy(n * 2);
    if (equal_cost) { xy[0] = 1; return xy; }  // 所有tour成本为2，确保不同结构均进入质量带。
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<double> unit(0, 1);
    for (auto& value : xy) value = unit(random);
    return xy;
}
Program terminal(Node feature) {
    Program p; p.length = 1; p.opcode[0] = 0; p.operand[0] = feature; return p;
}
Program regional_program() {
    // b + e_R - c*k；使用实际反馈影响扰动程度，树深与操作数均由原IR验证。
    Program p; p.length = 9;
    const std::uint8_t op[]{0, 0, 2, 0, 0, 4, 3, 0, 2};
    const std::uint8_t arg[]{4, 8, 0, 3, 5, 0, 0, 11, 0};
    std::copy(std::begin(op), std::end(op), p.opcode);
    std::copy(std::begin(arg), std::end(arg), p.operand);
    validate_program(p); return p;
}
void feedback_equal(const FeedbackState& a, const FeedbackState& b) {
    near(a.return_rate, b.return_rate, "返回率EMA不一致", 1e-7);
    near(a.ls_work, b.ls_work, "LS工作EMA不一致", 1e-7);
    check(a.stagnant_batches == b.stagnant_batches && a.epoch_batches == b.epoch_batches &&
          a.restarts == b.restarts, "反馈计数或重启保留规则不一致");
}
auto info_tuple(const FacoDiagnosticInfo& i) {
    const auto& c = i.construction; const auto& s = i.local_search;
    return std::make_tuple(i.construction_cost, i.final_cost, i.checklist_size, i.start_node,
        c.mne, c.steps, c.nonidentity_relocations, c.legal_exhausted,
        s.processed_nodes, s.candidate_checks, s.move_evaluations, s.accepted_moves,
        s.reactivations, s.constraint_rejections, s.evaluation_limit_reached);
}

struct Model {
    const PreparedProblem& p;
    DistanceFunction distance;
    ControlStaticData data;
    TourRecord global, epoch, parent;
    ReferenceArchive archive;
    FeedbackState feedback;
    TrailLimits limits;
    SparsePheromone pheromone;
    bool collisions;
    Model(const PreparedProblem& problem, std::uint64_t key, bool force_collisions)
        : p(problem), distance([&problem](Node a, Node b) { return problem.distance(a, b); }),
          data(make_control_static(problem, key)), global(problem.initial_tour, distance), epoch(global),
          parent(global), archive(global), limits(candidate_trail_limits(problem.settings.primary_width,
          problem.settings.p_best, problem.settings.retention, global.cost)),
          pheromone(problem.primary, limits.maximum), collisions(force_collisions) {
        if (collisions) {
            global.identity = epoch.identity = parent.identity = {};
            archive = ReferenceArchive(global);
        }
    }

    void verify(const ControlDeviceSnapshot& s, Node colony) const {
        const auto n = p.size(); const auto base = static_cast<std::size_t>(colony) * n;
        const auto& control = s.controls.at(colony); const auto& state = s.colonies.at(colony);
        feedback_equal(control.feedback, feedback);
        check(control.archive_size == archive.entries().size(), "档案容量/裁剪不一致");
        near(state.global_cost, global.cost, "GB成本不一致");
        near(state.epoch_cost, epoch.cost, "epoch成本不一致");
        near(state.parent_cost, parent.cost, "parent成本不一致");
        near(state.minimum, limits.minimum, "下界不一致"); near(state.maximum, limits.maximum, "上界不一致");
        near(state.default_trail, pheromone.default_value(), "default信息素不一致");
        near(control.tracked_global, global.cost, "控制层GB跟踪值不一致");
        near(control.tracked_epoch, epoch.cost, "控制层epoch跟踪值不一致");
        check(control.active_identity == parent.identity && control.epoch_identity == epoch.identity,
              "参考身份未按真正参考更新");
        segment(s.parent, base, parent.order, "parent排列不一致");
        segment(s.parent_positions, base, parent.positions, "parent逆映射不一致");
        segment(s.epoch, base, epoch.order, "epoch排列不一致");
        segment(s.global, base, global.order, "GB排列不一致");
        for (Node slot = 0; slot < archive_capacity; ++slot) {
            const auto offset = (base * archive_capacity) + slot * n;
            if (slot < archive.entries().size()) {
                const auto& record = archive.entries()[slot];
                near(control.archive_cost[slot], record.cost, "档案成本不一致");
                check(control.archive_identity[slot] == record.identity, "档案指纹不一致");
                segment(s.archive, offset, record.order, "档案排序/去重/内容不一致");
                segment(s.archive_positions, offset, record.positions, "档案逆映射不一致");
            } else {
                check(control.archive_cost[slot] == 0 && control.archive_identity[slot] == TourFingerprint{},
                      "档案空slot残留身份");
                zero_range(s.archive, offset, n); zero_range(s.archive_positions, offset, n);
            }
        }
        const auto products = pheromone.product_cache(distance, p.settings.beta);
        for (Node i = 0; i < products.size(); ++i) {
            const auto index = base * p.settings.primary_width + i;
            near(s.trails[index], pheromone.stored()[i], "stored信息素不一致");
            near(s.products[index], products[i], "产品缓存不一致");
        }
    }

    void verify_reset(const ControlDeviceSnapshot& s, Node colony) const {
        const auto n = p.size(), ants = p.settings.ants;
        const auto base = static_cast<std::size_t>(colony) * n * ants;
        const auto& state = s.colonies[colony];
        check(state.source_uniform == 0 && !state.source_is_epoch && state.iteration_best == 0,
              "重启残留source选择状态");
        for (Node ant = 0; ant < ants; ++ant) {
            segment(s.tours, base + ant * n, parent.order, "重启ant工作tour错误");
            segment(s.positions, base + ant * n, parent.positions, "重启ant逆映射错误");
            segment(s.per_ant_parent_positions, base + ant * n, parent.positions, "重启ant冻结逆映射错误");
            check(info_tuple(s.info[colony * ants + ant]) == info_tuple(FacoDiagnosticInfo{}),
                  "重启残留ant统计/起点");
            check(s.ant_identities[colony * ants + ant] == TourFingerprint{}, "重启残留ant身份");
        }
        zero_range(s.visited, base, ants * n); zero_range(s.scratch, base, ants * n);
        zero_range(s.pending, base * 5, ants * n * 5);
        zero_range(s.gains, static_cast<std::size_t>(colony) * ants * p.settings.ls_width * 2,
                   ants * p.settings.ls_width * 2);
    }

    void accept_batch(const ControlDeviceSnapshot& s, Node colony, const StartRegions& regions, Node action) {
        const auto n = p.size(), ants = p.settings.ants;
        std::vector<TourRecord> records;
        const auto& state = s.colonies[colony];
        double returned = 0, full_returned = 0, work = 0;
        for (Node ant = 0; ant < ants; ++ant) {
            const auto index = colony * ants + ant;
            const auto offset = static_cast<std::size_t>(index) * n;
            const auto& info = s.info[index];
            const auto* start_region = regions.nodes[action / 16][action / 4 % 4];
            check(std::find(start_region, start_region + regions.count, info.start_node) != start_region + regions.count,
                  "构造起点不属于评分选择的区域");
            ++checked_start_nodes;
            records.emplace_back(std::vector<Node>(s.tours.begin() + offset, s.tours.begin() + offset + n), distance);
            auto& record = records.back();
            max_cost_error = std::max(max_cost_error, std::abs(record.cost - info.final_cost));
            near(record.cost, info.final_cost, "ant输出成本与独立全tour重算不符");
            // 先完整校验，再使用设备增量成本重放严格比较，防止求和尾差改变平局顺序。
            record.cost = info.final_cost;
            segment(s.positions, offset, record.positions, "ant逆映射错误");
            segment(s.per_ant_parent_positions, offset, parent.positions, "批内参考未被冻结");
            if (collisions) record.identity = {};
            check(record.identity == s.ant_identities[index], "CUDA边集归约指纹错误");
            returned += record.identity == parent.identity;
            full_returned += same_tour(record.view(), parent.view());
            ++full_identity_comparisons;
            if (p.settings.ls_evaluation_limit)
                work += static_cast<double>(info.local_search.move_evaluations) / p.settings.ls_evaluation_limit;
            check(info.local_search.move_evaluations <= p.settings.ls_evaluation_limit, "LS超过固定上限");
            ++checked_ants;
        }
        if (collisions) {
            check(returned == ants, "强制碰撞未生效");
            ++collision_batches;
        } else check(returned == full_returned, "此fixture中指纹返回率不等于完整边集返回率");
        const auto best = std::min_element(records.begin(), records.end(),
            [](const auto& a, const auto& b) { return a.cost < b.cost; });
        check(state.iteration_best == static_cast<Node>(best - records.begin()), "iteration-best平局规则错误");
        const bool improved = best->cost < global.cost;
        if (improved) global = *best;
        if (best->cost < epoch.cost) epoch = *best;
        archive.update(global, records);
        update_feedback(feedback, improved, returned / ants, work / ants, ants);
        check(state.source_uniform >= 0 && state.source_uniform < 1, "强化随机值越界");
        check(state.source_is_epoch == (state.source_uniform < p.settings.epoch_source_probability),
              "强化参考分支错误");
        if (state.source_is_epoch) { parent = epoch; ++source_epoch_count; }
        else { parent = *best; ++source_iteration_count; }
        limits = candidate_trail_limits(p.settings.primary_width, p.settings.p_best, p.settings.retention, epoch.cost);
        pheromone.evaporate(p.settings.retention, limits.minimum);
        for (Node i = 0; i < n; ++i)
            pheromone.deposit(parent.order[i], parent.order[(i + 1) % n], 1.0 / parent.cost, limits.maximum);
        verify(s, colony);
    }
};

void compare_panel(Node n, bool equal_cost, Node variant, bool small, bool collisions = false) {
    FixedFacoSettings settings;
    settings.ants = small ? 4 : 8;
    settings.retention = variant % 2 ? 0.75 : 0.5;
    settings.epoch_source_probability = variant == 2 ? 1.0 : variant == 1 ? 0.0 : 0.5;
    settings.ls_evaluation_limit = variant == 2 ? 0 : 100000;
    const auto program = variant == 0 ? terminal(4) : variant == 1 ? terminal(5) : regional_program();
    const std::uint32_t mask = variant == 1 ? 0x0000a5a5u : UINT32_MAX;
    const std::vector<BatchTask> tasks{{71, 17}, {71, 29}, {93, 17}, {71, 17}};
    const Node batches = small ? 3 : 7;
    FacoBatchEngine engine(n, tasks.size(), settings);
    std::map<std::uint64_t, PreparedProblem> problems;
    for (std::uint64_t key : {71u, 93u}) {
        auto xy = points(n, key * 47, equal_cost);
        engine.register_problem(key, xy);
        auto prepared = make_cheap_problem(xy, settings); prepare_problem(prepared);
        problems.emplace(key, std::move(prepared));
    }
    BatchDiagnosticControls controls; controls.fixed_batches = batches; controls.capture_control = true;
    controls.force_fingerprint_collisions = collisions;
    const auto result = engine.evaluate_program_diagnostic(tasks, 60, program,
        PreparationMode::CachedCharged, mask, controls);
    check(result.completed_batches == batches && result.control_trace.size() == batches &&
          result.discarded_batches == 0, "固定控制诊断批次未完成");
    std::vector<std::unique_ptr<Model>> models;
    for (const auto& task : tasks)
        models.push_back(std::make_unique<Model>(problems.at(task.instance_key), task.instance_key, collisions));
    for (const auto& trace : result.control_trace) {
        std::vector<float> features(12 * tasks.size() * 32);
        std::vector<std::uint32_t> masks;
        for (Node colony = 0; colony < tasks.size(); ++colony) {
            auto& m = *models[colony]; const auto& task = tasks[colony];
            m.verify(trace.before, colony);
            if (trace.batch == 0) m.verify_reset(trace.before, colony);
            const auto key = control_mix(task.seed ^ control_mix(task.instance_key + 0xd1b54a32d192ed03ULL));
            const auto expected = make_action_snapshot(m.data, m.parent, m.archive, m.feedback, m.pheromone,
                m.limits, m.distance, trace.elapsed_ratio, key, trace.batch, mask);
            check(trace.alternatives[colony] == expected.alternative_slot &&
                  trace.masks[colony] == expected.legal_mask, "替代参考/非法动作mask错误");
            masks.push_back(expected.legal_mask);
            check(trace.regions[colony].count == expected.regions.count, "区域大小错误");
            for (Node mode = 0; mode < 2; ++mode) for (Node region = 0; region < 4; ++region)
                for (Node j = 0; j < region_capacity; ++j)
                    check(trace.regions[colony].nodes[mode][region][j] == expected.regions.nodes[mode][region][j],
                          "CPU/GPU起始区域不一致");
            for (Node feature = 0; feature < 12; ++feature) for (Node action = 0; action < 32; ++action) {
                const auto index = (feature * tasks.size() + colony) * 32 + action;
                features[index] = expected.features[feature * 32 + action];
                max_feature_error = std::max(max_feature_error, static_cast<double>(std::abs(features[index] - trace.features[index])));
                near(trace.features[index], features[index], "十二特征CPU/GPU不一致", 1e-6);
                ++checked_features;
            }
        }
        const auto scored = score_cpu(program, features, masks);
        check(scored.actions == trace.actions, "GPU实际GP动作不等于CPU评分");
        for (std::size_t i = 0; i < scored.scores.size(); ++i)
            near(trace.scores[i], scored.scores[i], "GPU实际GP分数不同", 2e-6);
        for (Node colony = 0; colony < tasks.size(); ++colony) {
            auto& m = *models[colony]; const Node action = trace.actions[colony];
            if (action >= 16) {
                m.parent = m.epoch = m.archive.entries().at(trace.alternatives[colony]);
                m.limits = candidate_trail_limits(m.p.settings.primary_width, settings.p_best, settings.retention, m.epoch.cost);
                m.pheromone.reset(m.limits.maximum); restart_feedback(m.feedback);
                ++checked_restarts;
                m.verify_reset(trace.after_restart, colony);
            }
            m.verify(trace.after_restart, colony);
            for (Node ant = 0; ant < settings.ants; ++ant)
                check(trace.after_restart.targets[colony * settings.ants + ant] == (2u << (action % 4)),
                      "动作MNE没有广播到全部蚂蚁");
            m.accept_batch(trace.after_batch, colony, trace.regions[colony], action);
            ++checked_batches;
        }
    }
    for (Node i = 0; i < tasks.size(); ++i) {
        feedback_equal(result.completed_control_states[i].feedback, models[i]->feedback);
        check(result.incumbents[i].present && result.incumbents[i].completed_seconds <= 60, "未按时提交GP结果");
    }
    if (equal_cost && variant == 0)
        for (const auto& model : models) check(model->feedback.restarts > 0, "等成本fixture未覆盖真实重启");
    if (collisions) {
        for (const auto& model : models) check(model->archive.entries().size() == 4,
              "完整指纹碰撞错误合并不同结构的档案成员");
        ++configurations; return;  // 强制碰撞只用于诊断，不与正常公开入口混算反馈。
    }
    check(result.incumbents[0].tour == result.incumbents[3].tour, "重复任务受colony位置影响");
    // A-B-A交错、冷准备和重新排列，验证状态隔离；这些程序不读取时间u。
    BatchDiagnosticControls fixed; fixed.fixed_batches = batches;
    engine.evaluate_diagnostic(tasks, 60, 8, PreparationMode::CachedCharged, fixed);
    const auto again = engine.evaluate_program_diagnostic(tasks, 60, program, PreparationMode::EndToEnd, mask, fixed);
    check(again.charged_seconds == 0 && again.completed_batches == batches, "GP冷准备收费/批次错误");
    auto reversed = tasks; std::reverse(reversed.begin(), reversed.end());
    const auto shuffled = engine.evaluate_program_diagnostic(reversed, 60, program, PreparationMode::CachedCharged, mask, fixed);
    for (Node i = 0; i < tasks.size(); ++i) {
        check(again.incumbents[i].tour == result.incumbents[i].tour &&
              shuffled.incumbents[tasks.size() - 1 - i].tour == result.incumbents[i].tour,
              "跨任务/准备模式/任务顺序泄漏控制状态");
        feedback_equal(again.completed_control_states[i].feedback, result.completed_control_states[i].feedback);
    }
    const auto zero = engine.evaluate_program(tasks, 0, program, PreparationMode::CachedCharged, mask);
    check(!zero.preparation_completed && zero.launched_batches == 0, "零预算进入GP循环");
    for (Node i = 0; i < tasks.size(); ++i)
        check(!zero.incumbents[i].present && zero.completed_control_states[i].archive_size == 0,
              "零预算返回旧incumbent/档案");
    ++configurations;
}

void late_transaction() {
    const auto xy = points(100, 4411, false);
    FixedFacoSettings settings; settings.ants = 8; settings.initial_ls_evaluation_limit = 0;
    FacoBatchEngine engine(100, 1, settings); engine.register_problem(501, xy);
    const std::vector<BatchTask> tasks{{501, 17}};
    const auto program = terminal(5);
    BatchDiagnosticControls fixed; fixed.fixed_batches = 1;
    const auto good = engine.evaluate_program_diagnostic(tasks, 10, program, PreparationMode::CachedCharged, UINT32_MAX, fixed);
    auto prepared = make_cheap_problem(xy, settings); prepare_problem(prepared);
    check(good.incumbents[0].cost < prepared.initial_cost, "GP迟到fixture没有真实改进");
    BatchDiagnosticControls delay; delay.completion_delay_ms = 1000;
    delay.capture_discarded = delay.capture_control = true;
    const auto late = engine.evaluate_program_diagnostic(tasks, 0.5, program, PreparationMode::CachedCharged, UINT32_MAX, delay);
    check(late.completed_batches == 0 && late.discarded_batches == 1 && late.launched_batches == 1,
          "GP实际迟到批次未丢弃");
    check(late.control_trace.empty() && late.completed_control_states[0].archive_size == 1 &&
          late.completed_control_states[0].feedback.epoch_batches == 0, "迟到控制状态对外泄漏");
    check(late.discarded_costs.size() == 1 && late.discarded_costs[0] < late.incumbents[0].cost,
          "GP迟到测试没有覆盖更优候选");
    near(late.incumbents[0].cost, prepared.initial_cost, "GP迟到候选覆盖合法GB");
    const auto again = engine.evaluate_program_diagnostic(tasks, 10, program, PreparationMode::CachedCharged, UINT32_MAX, fixed);
    check(again.incumbents[0].tour == good.incumbents[0].tour, "迟到GP状态污染下一评价");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool small = argc == 3 && std::string(argv[2]) == "--small";
        // 小于区域容量，以及n=3时信息素上下界相等的设备分支。
        compare_panel(3, false, 0, small); compare_panel(7, false, 1, small);
        for (Node n : {17u, 100u, 500u, 1000u}) compare_panel(n, false, n == 17 ? 2 : n == 500 ? 1 : 0, small);
        compare_panel(17, true, 0, small); compare_panel(100, true, 0, small);
        compare_panel(17, true, 0, small, true);
        late_transaction();
        check(checked_restarts > 0 && source_epoch_count > 0 && source_iteration_count > 0, "控制分支覆盖不足");
        std::ostringstream out; out.precision(17);
        out << "{\n  \"status\": \"passed\",\n  \"panel_configurations\": " << configurations
            << ",\n  \"colony_batches\": " << checked_batches << ",\n  \"ants\": " << checked_ants
            << ",\n  \"features\": " << checked_features << ",\n  \"restart_transactions\": " << checked_restarts
            << ",\n  \"actual_region_starts\": " << checked_start_nodes
            << ",\n  \"full_identity_comparisons\": " << full_identity_comparisons
            << ",\n  \"forced_collision_colony_batches\": " << collision_batches
            << ",\n  \"epoch_sources\": " << source_epoch_count << ",\n  \"iteration_sources\": " << source_iteration_count
            << ",\n  \"maximum_feature_error\": " << max_feature_error
            << ",\n  \"maximum_cost_error\": " << max_cost_error
            << ",\n  \"real_late_improvement_discarded\": true,\n  \"scope\": \"untrained main control engine; formal GP learning and E1-E4 pending\"\n}\n";
        if (argc >= 2) { std::ofstream file(argv[1]); check(static_cast<bool>(file), "无法写控制报告"); file << out.str(); }
        std::cout << out.str(); return 0;
    } catch (const std::exception& e) { std::cerr << "FAILED: " << e.what() << '\n'; return 1; }
}
