// 控制层手算、完整身份与有界采样检查；不读取任何实例标签。
#include "gp_faco/control_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using gp_faco::Node;
std::uint64_t sample_checks = 0, region_checks = 0, identity_checks = 0;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(double a, double b) { check(std::abs(a - b) < 1e-7, "特征手算值不符"); }
double unit_distance(Node a, Node b) { return a == b ? 0 : 2; }
std::vector<Node> sequence(Node n) { std::vector<Node> tour(n); std::iota(tour.begin(), tour.end(), 0); return tour; }

void sampling_and_regions() {
    std::mt19937_64 random(49178);
    for (Node n : {3u, 7u, 16u, 17u, 64u, 100u, 1000u}) {
        const auto order = sequence(n);
        const gp_faco::TourRecord reference(order, unit_distance);
        // 每行仅邻接自己所在的二元分量，专门触发候选邻域不足时的全局补齐。
        std::vector<Node> primary(n);
        for (Node a = 0; a < n; ++a) primary[a] = a % 2 ? a - 1 : a + 1 < n ? a + 1 : a - 1;
        for (Node repeat = 0; repeat < 40; ++repeat) {
            const auto key = random();
            Node samples[gp_faco::sample_capacity];
            gp_faco::sample_nodes(n, key, samples);
            const Node count = std::min(n, gp_faco::sample_capacity);
            std::set<Node> selected(samples, samples + count);
            check(selected.size() == count && *selected.rbegin() < n, "城市采样重复或越界");
            for (Node used = 0; used < count && used < n; ++used) {
                std::vector<Node> remaining;
                for (Node node = 0; node < n; ++node)
                    if (std::find(samples, samples + used, node) == samples + used) remaining.push_back(node);
                for (Node rank = 0; rank < remaining.size(); ++rank) {
                    check(gp_faco::remaining_node(rank, samples, used) == remaining[rank], "剩余秩映射不符");
                    ++sample_checks;
                }
            }
            for (Node mode = 0; mode < 2; ++mode) for (Node region = 0; region < 4; ++region) {
                Node nodes[gp_faco::region_capacity], again[gp_faco::region_capacity];
                gp_faco::build_region(reference.view(), primary.data(), 1, key, repeat, mode, region, nodes);
                gp_faco::build_region(reference.view(), primary.data(), 1, key, repeat, mode, region, again);
                const Node r = std::min(n, gp_faco::region_capacity);
                std::set<Node> unique(nodes, nodes + r);
                check(unique.size() == r && *unique.rbegin() < n, "区域重复、未补齐或越界");
                check(std::equal(nodes, nodes + r, again), "区域重复调用不确定");
                if (region < 2) for (Node i = 1; i < r; ++i)
                    check(nodes[i] == reference.view().successor(nodes[i - 1]), "连续区域不连续");
                ++region_checks;
            }
        }
    }
}

void identities_and_archive() {
    const auto order = sequence(17);
    const gp_faco::TourRecord original(order, unit_distance);
    for (Node shift = 0; shift < 17; ++shift) for (bool reverse : {false, true}) {
        auto changed = order;
        std::rotate(changed.begin(), changed.begin() + shift, changed.end());
        if (reverse) std::reverse(changed.begin(), changed.end());
        const gp_faco::TourRecord candidate(changed, unit_distance);
        check(candidate.identity == original.identity && gp_faco::duplicate_record(candidate, original),
              "旋转/反向改变无向tour身份");
        ++identity_checks;
    }
    auto changed = order; std::swap(changed[5], changed[8]);
    gp_faco::TourRecord alternative(changed, unit_distance);
    gp_faco::ReferenceArchive archive(original);
    archive.update(original, {alternative, original});
    check(archive.entries().size() == 2, "档案未去重");
    check(gp_faco::sampled_difference(alternative.view(), original.view(), order.data(), 1) == 0,
          "采样漏检fixture失效");
    check(archive.alternative(original, {0}) < gp_faco::archive_capacity,
          "采样差异0错误屏蔽了完整结构不同的替代解");
    // 人工注入指纹碰撞；真正的身份判定仍需完整邻接。
    alternative.identity = original.identity;
    check(!gp_faco::duplicate_record(original, alternative), "把指纹碰撞误当同一tour");
    archive = gp_faco::ReferenceArchive(original);
    archive.update(original, {alternative});
    check(archive.entries().size() == 2 && archive.alternative(original, {0}) == 1, "碰撞丢失了不同解");

    // 人工成本只用于档案质量带与GB保留规则；路线本身仍是合法的不同排列。
    auto old = original; old.cost = 100;
    archive = gp_faco::ReferenceArchive(old);
    std::vector<gp_faco::TourRecord> pool;
    for (Node i = 1; i <= 6; ++i) {
        changed = order; std::swap(changed[i], changed[i + 3]);
        pool.emplace_back(changed, unit_distance); pool.back().cost = 100 + i * 0.3;
    }
    archive.update(old, pool);
    check(archive.entries().size() == 4 && gp_faco::same_tour(archive.entries()[0].view(), old.view()),
          "容量或GB保留错误");
    const auto first_order = archive.entries();
    std::reverse(pool.begin(), pool.end());
    gp_faco::ReferenceArchive reordered(old); reordered.update(old, pool);
    for (Node i = 0; i < 4; ++i) check(gp_faco::duplicate_record(first_order[i], reordered.entries()[i]),
                                     "批次输入顺序改变档案选择");
    auto new_best = pool[0]; new_best.cost = 98;
    archive.update(new_best, {});
    check(archive.entries().size() == 1 && archive.entries()[0].cost == 98, "GB改善后没有重新裁剪质量带");
}

void feedback_and_features() {
    gp_faco::FeedbackState feedback;
    gp_faco::update_feedback(feedback, false, 1, 0.5, 32);
    near(feedback.return_rate, 1.0 / 16); near(feedback.ls_work, 1.0 / 32);
    check(feedback.stagnant_batches == 1 && feedback.epoch_batches == 1, "反馈计数错误");
    gp_faco::update_feedback(feedback, true, 0, 0, 0);
    check(feedback.stagnant_batches == 1 && feedback.epoch_batches == 1, "空批改变反馈");
    gp_faco::restart_feedback(feedback);
    check(feedback.return_rate == 0 && feedback.ls_work == 0 && feedback.epoch_batches == 0 &&
          feedback.stagnant_batches == 1 && feedback.restarts == 1, "重启清理了全局停滞或遗留epoch反馈");
    feedback.stagnant_batches = UINT64_MAX; feedback.epoch_batches = UINT64_MAX;
    gp_faco::update_feedback(feedback, false, 0, 0, 1);
    check(feedback.stagnant_batches == UINT64_MAX && feedback.epoch_batches == UINT64_MAX, "计数回绕");

    const Node n = 17;
    const auto order = sequence(n);
    const gp_faco::TourRecord active(order, unit_distance);
    gp_faco::ReferenceArchive archive(active);
    gp_faco::CandidateRows rows(n);
    for (Node a = 0; a < n; ++a) rows[a] = {(a + 1) % n, (a + 2) % n};
    gp_faco::ControlStaticData problem{n, 2, gp_faco::flattened(rows), order, std::vector<double>(n, 1), 1e-12};
    gp_faco::SparsePheromone pheromone(rows, 3);
    feedback = {}; feedback.stagnant_batches = 16; feedback.return_rate = 0.125f; feedback.ls_work = 0.75f;
    const auto snapshot = gp_faco::make_action_snapshot(problem, active, archive, feedback, pheromone,
        {1, 5}, unit_distance, 0.25, 9107, 3);
    check(snapshot.legal_mask == 0xffffu && snapshot.alternative_slot == gp_faco::archive_capacity,
          "无替代参考时restart未屏蔽");
    for (Node action = 0; action < 32; ++action) {
        const auto feature = [&](Node f) { return snapshot.features[f * 32 + action]; };
        near(feature(0), 0.25); near(feature(1), 0.5); near(feature(2), 0.125); near(feature(3), 0.75);
        near(feature(4), action / 16); near(feature(5), (action % 4 + 1) * 0.25);
        near(feature(6), 0); near(feature(7), 0);
        near(feature(8), action < 16 ? 1.0 / 3 : 0);
        near(feature(9), 0); near(feature(10), action < 16 ? 0.5 : 0);
        near(feature(11), action < 16 ? 1.0 / 16 : 0);
    }
    auto changed = order; std::swap(changed[5], changed[8]);
    archive.update(active, {gp_faco::TourRecord(changed, unit_distance)});
    const auto with_restart = gp_faco::make_action_snapshot(problem, active, archive, feedback, pheromone,
        {1, 5}, unit_distance, 0.25, 9107, 3);
    check(with_restart.legal_mask == UINT32_MAX, "不同替代解未激活restart");
    for (Node action = 16; action < 32; ++action) {
        near(with_restart.features[10 * 32 + action], 0.5);
        check(with_restart.features[7 * 32 + action] > 0, "替代解差异缺失");
    }
    const auto equal_bounds = gp_faco::make_action_snapshot(problem, active, archive, feedback, pheromone,
        {3, 3}, unit_distance, 1.5, 9107, 3, 1u << 7);
    check(equal_bounds.legal_mask == (1u << 7), "实验mask未保持");
    for (Node action = 0; action < 32; ++action) {
        near(equal_bounds.features[action], 1); near(equal_bounds.features[10 * 32 + action], 1);
    }
}

void static_local_scale() {
    gp_faco::FixedFacoSettings settings;
    settings.primary_width = 2; settings.backup_width = 0; settings.ls_width = 2;
    std::vector<double> coordinates;
    for (Node i = 0; i < 12; ++i) { coordinates.push_back(i); coordinates.push_back(0); }
    auto problem = gp_faco::make_cheap_problem(coordinates, settings);
    gp_faco::prepare_problem(problem);
    const auto control = gp_faco::make_control_static(problem, 731);
    near(control.local_scale[0], 4.5); near(control.local_scale[5], 2.5);
    check(control.samples.size() == 12 && control.primary_width == 2, "局部尺度改变了候选容量");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        sampling_and_regions(); identities_and_archive(); feedback_and_features(); static_local_scale();
        std::ostringstream out;
        out << "{\n  \"status\": \"passed\",\n  \"remaining_rank_checks\": " << sample_checks
            << ",\n  \"region_checks\": " << region_checks << ",\n  \"identity_checks\": " << identity_checks
            << ",\n  \"scope\": \"CPU control reference; CUDA transaction/Engine integration pending\"\n}\n";
        if (argc > 1) { std::ofstream file(argv[1]); if (!file) throw std::runtime_error("无法写报告"); file << out.str(); }
        std::cout << out.str(); return 0;
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
