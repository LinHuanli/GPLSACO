#pragma once

#include "gp_faco/control_ops.hpp"
#include "gp_faco/prepared_problem.hpp"

namespace gp_faco {

struct TourRecord {
    std::vector<Node> order, positions;
    double cost;
    TourFingerprint identity;
    TourRecord(std::vector<Node> tour, const DistanceFunction& distance);
    TourView view() const { return {order.data(), positions.data(), static_cast<Node>(order.size())}; }
};

bool record_less(const TourRecord& a, const TourRecord& b);
bool duplicate_record(const TourRecord& a, const TourRecord& b);

class ReferenceArchive {
public:
    explicit ReferenceArchive(TourRecord initial) : entries_{std::move(initial)} {}
    const std::vector<TourRecord>& entries() const { return entries_; }
    void update(const TourRecord& global, const std::vector<TourRecord>& completed);
    Node alternative(const TourRecord& active, const std::vector<Node>& samples) const;
private:
    std::vector<TourRecord> entries_;
};

struct ControlStaticData {
    Node n, primary_width;
    std::vector<Node> primary, samples;
    std::vector<double> local_scale;
    double epsilon;
};
ControlStaticData make_control_static(const PreparedProblem& problem, std::uint64_t instance_key);

struct ActionSnapshot {
    std::array<float, 12 * 32> features{};
    StartRegions regions;
    Node alternative_slot = archive_capacity;
    std::uint32_t legal_mask = 0;
};

// CPU语义参考，GPU应在同一批次只算一次这些量，再广播给32个动作。
ActionSnapshot make_action_snapshot(const ControlStaticData& problem, const TourRecord& active,
    const ReferenceArchive& archive, const FeedbackState& feedback, const SparsePheromone& pheromone,
    TrailLimits limits, const DistanceFunction& distance, double elapsed_ratio,
    std::uint64_t solve_key, Node batch, std::uint32_t experiment_mask = UINT32_MAX);

}  // namespace gp_faco
