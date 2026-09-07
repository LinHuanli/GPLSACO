#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace gp_faco {

using Node = std::uint32_t;
using DistanceFunction = std::function<double(Node, Node)>;
using CandidateRows = std::vector<std::vector<Node>>;
using Edge = std::pair<Node, Node>;
using EdgeAllowed = std::function<bool(Node, Node)>;

// 该层是纯 C++ 语义参考；CUDA 将使用设备视图，不携带 std::function。
class DistanceOrderedCandidates {
public:
    DistanceOrderedCandidates(CandidateRows rows, const DistanceFunction& distance);
    const std::vector<Node>& operator[](Node node) const { return rows_.at(node); }
    std::size_t size() const { return rows_.size(); }
private:
    CandidateRows rows_;
};

struct Reconnection {
    bool nonidentity = false;
    std::array<Edge, 3> removed{};
    std::array<Edge, 3> added{};
};

struct LocalSearchStats {
    std::uint64_t processed_nodes = 0;
    std::uint64_t candidate_checks = 0;
    std::uint64_t move_evaluations = 0;
    std::uint64_t accepted_moves = 0;
    std::uint64_t reactivations = 0;
    std::uint64_t constraint_rejections = 0;
    bool evaluation_limit_reached = false;
};

class CpuTour {
public:
    CpuTour(std::vector<Node> order, DistanceFunction distance);
    const std::vector<Node>& order() const { return order_; }
    const std::vector<Node>& positions() const { return positions_; }
    std::size_t size() const { return order_.size(); }
    double cost() const { return cost_; }
    double recomputed_cost() const;
    Node successor(Node node) const;
    Node predecessor(Node node) const;
    bool contains_edge(Node a, Node b) const;
    Reconnection relocation_edges(Node target, Node node) const;
    bool relocate(Node target, Node node);
    void reverse_section(Node first, Node end);
    LocalSearchStats checklist_two_opt(
        const DistanceOrderedCandidates& candidates, std::vector<Node>& checklist,
        std::uint64_t max_evaluations = std::numeric_limits<std::uint64_t>::max(),
        const EdgeAllowed& edge_allowed = {});
private:
    void flip_without_cost(Node first, Node end);
    std::vector<Node> order_;
    std::vector<Node> positions_;
    DistanceFunction distance_;
    double cost_ = 0;
};

struct ConstructionStats {
    std::uint32_t mne = 0;
    std::uint32_t steps = 0;
    std::uint32_t nonidentity_relocations = 0;
    bool legal_exhausted = false;
};

class FocusedConstruction {
public:
    FocusedConstruction(const CpuTour& reference, Node start, std::uint32_t mne_target);
    bool done() const;
    void step(Node selected);
    const CpuTour& tour() const { return work_; }
    CpuTour& tour() { return work_; }
    Node current() const { return current_; }
    const std::vector<std::uint8_t>& visited() const { return visited_; }
    std::vector<Node>& checklist() { return checklist_; }
    const ConstructionStats& stats() const { return stats_; }
private:
    CpuTour reference_;
    CpuTour work_;
    std::vector<std::uint8_t> visited_;
    std::vector<Node> checklist_;
    Node current_;
    std::uint32_t target_;
    ConstructionStats stats_;
};

enum class SelectionStage { Primary, ZeroWeightPrimary, Backup, GlobalFallback, Exhausted };
struct Selection {
    Node node;
    SelectionStage stage;
};

Selection select_next(Node current, const std::vector<Node>& primary,
                      const std::vector<double>& products, const std::vector<Node>& backup,
                      const std::vector<std::uint8_t>& visited, const DistanceFunction& distance,
                      double uniform_01, const std::function<bool(Node)>& node_allowed = {});

struct TrailLimits { double minimum; double maximum; };
TrailLimits candidate_trail_limits(std::uint32_t candidate_count, double p_best,
                                  double retention, double solution_cost);

class SparsePheromone {
public:
    SparsePheromone(CandidateRows rows, double initial, bool symmetric = true);
    double get(Node from, Node to) const;
    const std::vector<double>& stored() const { return trails_; }
    double default_value() const { return default_; }
    void evaporate(double retention, double minimum);
    void deposit(Node from, Node to, double increment, double maximum);
    void reset(double value);
    std::vector<double> product_cache(const DistanceFunction& distance, double beta) const;
private:
    void increase_directed(Node from, Node to, double increment, double maximum);
    CandidateRows rows_;
    std::vector<double> trails_;
    std::size_t width_;
    double default_;
    bool symmetric_;
};

}  // namespace gp_faco
