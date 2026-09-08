// 主底座档案、替代参考和十二特征的CPU参考，不读取最优标签。
#include "gp_faco/control_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gp_faco {
namespace {
void require(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
}

TourRecord::TourRecord(std::vector<Node> tour, const DistanceFunction& distance) : order(std::move(tour)) {
    const CpuTour checked(order, distance);
    positions = checked.positions(); cost = checked.cost(); identity = fingerprint(view());
    require(cost > 0, "控制层tour成本必须为正");
}

bool record_less(const TourRecord& a, const TourRecord& b) {
    if (a.cost != b.cost) return a.cost < b.cost;
    if (a.identity != b.identity) return a.identity < b.identity;
    return canonical_less(a.view(), b.view());
}

bool duplicate_record(const TourRecord& a, const TourRecord& b) {
    return a.identity == b.identity && same_tour(a.view(), b.view());
}

void ReferenceArchive::update(const TourRecord& global, const std::vector<TourRecord>& completed) {
    require(global.order.size() == entries_[0].order.size() && global.cost <= entries_[0].cost,
            "档案GB维数不符或出现恶化");
    auto pool = entries_;
    for (const auto& candidate : completed) {
        require(candidate.order.size() == global.order.size() && std::isfinite(candidate.cost) &&
                candidate.cost >= global.cost, "批次候选与GB不一致");
        pool.push_back(candidate);
    }
    std::stable_sort(pool.begin(), pool.end(), record_less);
    std::vector<TourRecord> chosen{global};
    for (const auto& candidate : pool) {
        if (candidate.cost > global.cost * (1 + archive_quality_band)) continue;
        if (std::any_of(chosen.begin(), chosen.end(), [&](const auto& value) { return duplicate_record(candidate, value); }))
            continue;
        chosen.push_back(candidate);
        if (chosen.size() == archive_capacity) break;
    }
    entries_ = std::move(chosen);
}

Node ReferenceArchive::alternative(const TourRecord& active, const std::vector<Node>& samples) const {
    require(!samples.empty() && samples.size() <= sample_capacity && active.order.size() == entries_[0].order.size(),
            "采样或活动参考形状无效");
    for (Node node : samples) require(node < active.order.size(), "采样节点越界");
    Node selected = archive_capacity, largest = 0;
    for (Node i = 0; i < entries_.size(); ++i) {
        const auto& candidate = entries_[i];
        if (duplicate_record(candidate, active)) continue;
        const Node difference = sampled_difference(candidate.view(), active.view(), samples.data(), samples.size());
        if (selected == archive_capacity || difference > largest ||
            (difference == largest && record_less(candidate, entries_[selected]))) {
            selected = i; largest = difference;
        }
    }
    return selected;
}

ControlStaticData make_control_static(const PreparedProblem& problem, std::uint64_t instance_key) {
    require(problem.ready && problem.local_scale.size() == problem.size(), "控制静态数据尚未准备");
    ControlStaticData result{problem.size(), problem.settings.primary_width, flattened(problem.primary),
        std::vector<Node>(std::min(problem.size(), sample_capacity)), problem.local_scale, problem.scale_epsilon};
    sample_nodes(result.n, instance_key, result.samples.data());
    return result;
}

ActionSnapshot make_action_snapshot(const ControlStaticData& problem, const TourRecord& active,
    const ReferenceArchive& archive, const FeedbackState& feedback, const SparsePheromone& pheromone,
    TrailLimits limits, const DistanceFunction& distance, double elapsed_ratio,
    std::uint64_t solve_key, Node batch, std::uint32_t experiment_mask) {
    require(problem.n == active.order.size() && std::isfinite(elapsed_ratio) &&
            std::isfinite(limits.minimum) && std::isfinite(limits.maximum) &&
            limits.minimum <= limits.maximum, "特征输入形状或比例无效");
    ActionSnapshot result;
    result.alternative_slot = archive.alternative(active, problem.samples);
    const bool has_alternative = result.alternative_slot < archive_capacity;
    result.legal_mask = experiment_mask & (has_alternative ? UINT32_MAX : 0xffffu);
    require(result.legal_mask != 0, "没有合法控制动作");
    result.regions.count = std::min(problem.n, region_capacity);
    const auto& entries = archive.entries();
    TourView views[archive_capacity];
    for (Node i = 0; i < entries.size(); ++i) views[i] = entries[i].view();
    for (Node mode = 0; mode < 2; ++mode) {
        const TourRecord* reference = mode == 0 ? &active : has_alternative ? &entries[result.alternative_slot] : nullptr;
        float gap = 0, difference = 0;
        if (reference) {
            gap = static_cast<float>(unit_clip((reference->cost / entries[0].cost - 1) / archive_quality_band));
            if (mode) difference = static_cast<float>(sampled_difference(reference->view(), active.view(),
                problem.samples.data(), problem.samples.size())) / (2 * problem.samples.size());
        }
        for (Node region = 0; region < 4; ++region) {
            float regional[4]{};
            if (reference) {
                auto* nodes = result.regions.nodes[mode][region];
                build_region(reference->view(), problem.primary.data(), problem.primary_width,
                             solve_key, batch, mode, region, nodes);
                region_features(reference->view(), nodes, result.regions.count, views, entries.size(),
                    problem.local_scale.data(), problem.epsilon, limits.minimum, limits.maximum,
                    distance, [&](Node a, Node b) { return pheromone.get(a, b); }, regional);
            }
            for (Node level = 0; level < 4; ++level) {
                const Node action = mode * 16 + region * 4 + level;
                const float values[12]{static_cast<float>(unit_clip(elapsed_ratio)),
                    static_cast<float>(feedback.stagnant_batches < 32 ? feedback.stagnant_batches / 32.0 : 1),
                    feedback.return_rate, feedback.ls_work, static_cast<float>(mode), (level + 1) * 0.25f,
                    gap, difference, regional[0], regional[1], regional[2], regional[3]};
                for (Node feature = 0; feature < 12; ++feature) result.features[feature * 32 + action] = values[feature];
            }
        }
    }
    return result;
}

}  // namespace gp_faco
