// Route/LS/pheromone 语义依据 RSkinderowicz Adaptive-Tuning（MIT）。
// 原实现 Copyright (c) 2024 RSkinderowicz；许可全文见
// provenance/licenses/Adaptive-Tuning-MIT.txt。
// 本文件为项目 CPU 移植；来源、原文许可和适配说明见 provenance。
#include "gp_faco/faco_cpu.hpp"
#include "gp_faco/edge_constraints.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gp_faco {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void validate_rows(const CandidateRows& rows) {
    require(rows.size() >= 3, "候选表需要至少三个节点");
    for (std::size_t from = 0; from < rows.size(); ++from) {
        auto sorted = rows[from];
        std::sort(sorted.begin(), sorted.end());
        require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "候选表有重复节点");
        for (Node to : sorted) require(to < rows.size() && to != from, "候选节点越界或自环");
    }
}

}  // namespace

DistanceOrderedCandidates::DistanceOrderedCandidates(CandidateRows rows, const DistanceFunction& distance)
    : rows_(std::move(rows)) {
    validate_rows(rows_);
    for (Node from = 0; from < rows_.size(); ++from) {
        for (Node to : rows_[from]) {
            const double value = distance(from, to);
            require(std::isfinite(value) && value >= 0, "候选距离必须非负且有限");
        }
        // 成员由先验决定；仅为 LS 建立真实距离有序视图，相同距离保持输入次序。
        std::stable_sort(rows_[from].begin(), rows_[from].end(), [&](Node a, Node b) {
            return distance(from, a) < distance(from, b);
        });
    }
}

CpuTour::CpuTour(std::vector<Node> order, DistanceFunction distance)
    : order_(std::move(order)), positions_(order_.size()), distance_(std::move(distance)) {
    require(order_.size() >= 3 && static_cast<bool>(distance_), "无效 tour 或距离函数");
    auto sorted = order_;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) require(sorted[i] == i, "tour 不是完整排列");
    for (Node i = 0; i < order_.size(); ++i) positions_[order_[i]] = i;
    cost_ = recomputed_cost();
    require(std::isfinite(cost_) && cost_ >= 0, "tour 成本必须非负且有限");
}

double CpuTour::recomputed_cost() const {
    double sum = 0;
    Node previous = order_.back();
    for (Node node : order_) {
        const double distance = distance_(previous, node);
        require(std::isfinite(distance) && distance >= 0, "距离必须非负且有限");
        sum += distance;
        previous = node;
    }
    return sum;
}

Node CpuTour::successor(Node node) const {
    return order_[(positions_.at(node) + 1) % size()];
}

Node CpuTour::predecessor(Node node) const {
    return order_[(positions_.at(node) + size() - 1) % size()];
}

bool CpuTour::contains_edge(Node a, Node b) const {
    return successor(a) == b || predecessor(a) == b;
}

Reconnection CpuTour::relocation_edges(Node target, Node node) const {
    require(target < size() && node < size() && target != node, "重定位端点无效");
    if (successor(target) == node) return {};
    const Node before = predecessor(node), after = successor(node), target_after = successor(target);
    return {true, {{{before, node}, {node, after}, {target, target_after}}},
                  {{{before, after}, {target, node}, {node, target_after}}}};
}

bool CpuTour::relocate(Node target, Node node) {
    const auto edges = relocation_edges(target, node);
    if (!edges.nonidentity) return false;
    const auto position = positions_[node], target_position = positions_[target];
    if (target_position < position) {
        for (auto i = position; i > target_position + 1; --i) order_[i] = order_[i - 1];
        order_[target_position + 1] = node;
    } else {
        for (auto i = position; i < target_position; ++i) order_[i] = order_[i + 1];
        order_[target_position] = node;
    }
    for (auto i = std::min(position, target_position); i <= std::max(position, target_position); ++i) {
        positions_[order_[i]] = i;
    }
    // 保留原生六条边的浮点加减顺序，方便比较增量误差。
    cost_ += -distance_(edges.removed[0].first, edges.removed[0].second)
             -distance_(edges.removed[1].first, edges.removed[1].second)
             -distance_(edges.removed[2].first, edges.removed[2].second)
             +distance_(edges.added[0].first, edges.added[0].second)
             +distance_(edges.added[1].first, edges.added[1].second)
             +distance_(edges.added[2].first, edges.added[2].second);
    return true;
}

void CpuTour::flip_without_cost(Node start, Node end) {
    auto first = positions_.at(start), last = positions_.at(end);
    if (first > last) std::swap(first, last);
    const auto n = static_cast<Node>(size());
    const auto length = last - first;
    if (length <= n - length) {
        std::reverse(order_.begin() + first, order_.begin() + last);
        for (auto i = first; i < last; ++i) positions_[order_[i]] = i;
    } else {
        // 保留原生环绕边界的计算方式。first==0 时，原生会多走一圈；
        // 即使无向边集合等价，也不能简写为 count/2 次交换而改变数组方向。
        Node left = last, right = first == 0 ? n - 1 : first - 1;
        std::int64_t i = 0, j = n - left + right + 1;
        while (i++ < j--) {
            std::swap(order_[left], order_[right]);
            positions_[order_[left]] = left;
            positions_[order_[right]] = right;
            left = (left + 1) % n;
            right = right == 0 ? n - 1 : right - 1;
        }
    }
}

void CpuTour::reverse_section(Node start, Node end) {
    require(start < size() && end < size(), "反转端点越界");
    if (start == end) return;
    if (positions_[start] > positions_[end]) std::swap(start, end);
    const auto before = predecessor(start), last = predecessor(end);
    const double delta = -distance_(before, start) - distance_(last, end)
                         + distance_(before, last) + distance_(start, end);
    flip_without_cost(start, end);
    cost_ += delta;
}

LocalSearchStats CpuTour::checklist_two_opt(const DistanceOrderedCandidates& candidates,
                                           std::vector<Node>& checklist,
                                           std::uint64_t max_evaluations,
                                           const EdgeAllowed& edge_allowed) {
    require(candidates.size() == size(), "LS 候选表维数不符");
    for (Node node : checklist) require(node < size(), "checklist 节点越界");
    LocalSearchStats stats;
    double accumulated_gain = 0;
    std::size_t next = 0;
    while (next < checklist.size() && stats.accepted_moves < size()) {
        if (stats.move_evaluations >= max_evaluations) {
            stats.evaluation_limit_reached = true;
            break;
        }
        const Node a = checklist[next++], a_next = successor(a), a_previous = predecessor(a);
        ++stats.processed_nodes;
        const double to_next = distance_(a, a_next), to_previous = distance_(a_previous, a);
        double best_gain = -1;
        std::array<Node, 4> move{};
        bool interrupted = false;
        for (int kind = 0; kind < 2 && !interrupted; ++kind) {
            const double current_distance = kind == 0 ? to_next : to_previous;
            for (Node b : candidates[a]) {
                ++stats.candidate_checks;
                const double ab = distance_(a, b);
                if (!(current_distance > ab)) break;
                if (stats.move_evaluations == max_evaluations) {
                    interrupted = true;
                    break;
                }
                ++stats.move_evaluations;
                const Node neighbor = kind == 0 ? successor(b) : predecessor(b);
                if (edge_allowed && !two_opt_allowed(a, kind == 0 ? a_next : a_previous,
                                                    b, neighbor, edge_allowed)) {
                    ++stats.constraint_rejections;
                    continue;
                }
                const double other = kind == 0 ? distance_(b, neighbor) : distance_(neighbor, b);
                const double closing = distance_(kind == 0 ? a_next : a_previous, neighbor);
                // 相邻边只会原样补回；明确赋数学上的零增益，避免浮点假改善循环。
                const double gain = (neighbor == a || b == (kind == 0 ? a_next : a_previous))
                    ? 0.0 : current_distance + other - ab - closing;
                if (gain > best_gain) {
                    best_gain = gain;
                    move = kind == 0 ? std::array<Node, 4>{a_next, neighbor, a, b}
                                     : std::array<Node, 4>{a, b, a_previous, neighbor};
                }
            }
        }
        // 统一限额截断只提交已经穷尽本节点声明检查的 move；不偏向先扫描的一类。
        if (interrupted) {
            stats.evaluation_limit_reached = true;
            break;
        }
        if (best_gain > 0) {
            flip_without_cost(move[0], move[1]);
            for (Node node : move) {
                if (std::find(checklist.begin() + static_cast<std::ptrdiff_t>(next),
                              checklist.end(), node) == checklist.end()) {
                    checklist.push_back(node);
                    ++stats.reactivations;
                }
            }
            ++stats.accepted_moves;
            accumulated_gain -= best_gain;
        }
    }
    cost_ += accumulated_gain;
    return stats;
}

FocusedConstruction::FocusedConstruction(const CpuTour& reference, Node start, std::uint32_t target)
    : reference_(reference), work_(reference), visited_(reference.size(), 0),
      current_(start), target_(target) {
    require(start < reference.size() && target > 0, "起点或 MNE 阈值无效");
    visited_[start] = 1;
}

bool FocusedConstruction::done() const {
    return stats_.mne >= target_ || stats_.steps + 1 >= work_.size();
}

void FocusedConstruction::step(Node selected) {
    require(!done() && selected < work_.size() && !visited_[selected], "构造已结束或选中已访问节点");
    const Node previous = work_.predecessor(selected);
    visited_[selected] = 1;
    ++stats_.steps;
    if (work_.relocate(current_, selected)) ++stats_.nonidentity_relocations;
    if (!reference_.contains_edge(current_, selected)) {
        ++stats_.mne;
        for (Node node : {current_, selected, previous}) {
            if (std::find(checklist_.begin(), checklist_.end(), node) == checklist_.end()) {
                checklist_.push_back(node);
            }
        }
    }
    current_ = selected;
}

Selection select_next(Node current, const std::vector<Node>& primary,
                      const std::vector<double>& products, const std::vector<Node>& backup,
                      const std::vector<std::uint8_t>& visited, const DistanceFunction& distance,
                      double uniform, const std::function<bool(Node)>& node_allowed) {
    require(current < visited.size() && visited[current] && primary.size() == products.size(),
            "选点状态或产品缓存形状不符");
    require(std::isfinite(uniform) && uniform >= 0 && uniform < 1, "随机数必须在 [0,1)");
    std::vector<Node> available;
    std::vector<double> prefixes;
    double sum = 0;
    for (std::size_t i = 0; i < primary.size(); ++i) {
        const Node node = primary[i];
        require(node < visited.size() && std::isfinite(products[i]) && products[i] >= 0,
                "候选编号或权重无效");
        if (!visited[node] && (!node_allowed || node_allowed(node))) {
            available.push_back(node);
            sum += products[i];
            prefixes.push_back(sum);
        }
    }
    require(std::isfinite(sum), "候选权重之和溢出");
    if (!available.empty()) {
        if (sum == 0) {
            // 原生单候选零权重会返回当前节点；共同底座显式均匀回退到合法候选。
            const auto position = std::min(available.size() - 1,
                static_cast<std::size_t>(uniform * available.size()));
            return {available[position], SelectionStage::ZeroWeightPrimary};
        }
        const double threshold = uniform * sum;
        for (std::size_t i = 0; i < available.size(); ++i) {
            if (threshold < prefixes[i]) return {available[i], SelectionStage::Primary};
        }
        return {available.back(), SelectionStage::Primary};
    }
    for (Node node : backup) {
        require(node < visited.size(), "备用节点越界");
        if (!visited[node] && (!node_allowed || node_allowed(node))) return {node, SelectionStage::Backup};
    }
    Node chosen = current;
    double minimum = std::numeric_limits<double>::infinity();
    for (Node node = 0; node < visited.size(); ++node) {
        if (visited[node] || (node_allowed && !node_allowed(node))) continue;
        const double value = distance(current, node);
        require(std::isfinite(value) && value >= 0, "回退距离必须有限且非负");
        if (value < minimum) { minimum = value; chosen = node; }
    }
    return {chosen, chosen == current ? SelectionStage::Exhausted : SelectionStage::GlobalFallback};
}

TrailLimits candidate_trail_limits(std::uint32_t count, double p_best, double retention, double cost) {
    require(count >= 2 && p_best > 0 && p_best < 1 && retention >= 0 && retention < 1 &&
            std::isfinite(cost) && cost > 0, "信息素边界参数无效");
    const double maximum = 1.0 / (cost * (1.0 - retention));
    const double p = std::pow(p_best, 1.0 / count);
    const double minimum = std::min(maximum, maximum * (1.0 - p) / ((count - 1.0) * p));
    require(std::isfinite(maximum) && minimum > 0, "信息素边界超出有效数值范围");
    return {minimum, maximum};
}

SparsePheromone::SparsePheromone(CandidateRows rows, double initial, bool symmetric)
    : rows_(std::move(rows)), width_(rows_.empty() ? 0 : rows_[0].size()),
      default_(initial), symmetric_(symmetric) {
    validate_rows(rows_);
    require(width_ > 0 && std::isfinite(initial) && initial >= 0, "初始信息素或候选宽度无效");
    for (const auto& row : rows_) require(row.size() == width_, "存储行必须等宽");
    trails_.assign(rows_.size() * width_, initial);
}

double SparsePheromone::get(Node from, Node to) const {
    require(from < rows_.size() && to < rows_.size(), "信息素节点越界");
    const auto& row = rows_[from];
    const auto found = std::find(row.begin(), row.end(), to);
    return found == row.end() ? default_ : trails_[from * width_ + (found - row.begin())];
}

void SparsePheromone::evaporate(double retention, double minimum) {
    require(std::isfinite(retention) && retention >= 0 && retention <= 1 &&
            std::isfinite(minimum) && minimum >= 0, "蒸发参数无效");
    for (double& trail : trails_) trail = std::max(minimum, trail * retention);
    default_ = std::max(minimum, default_ * retention);
}

void SparsePheromone::increase_directed(Node from, Node to, double increment, double maximum) {
    const auto& row = rows_[from];
    const auto found = std::find(row.begin(), row.end(), to);
    if (found != row.end()) {
        auto& trail = trails_[from * width_ + (found - row.begin())];
        trail = std::min(maximum, trail + increment);
    }
}

void SparsePheromone::deposit(Node from, Node to, double increment, double maximum) {
    require(from < rows_.size() && to < rows_.size() && std::isfinite(increment) && increment >= 0 &&
            std::isfinite(maximum) && maximum >= 0, "信息素强化参数无效");
    increase_directed(from, to, increment, maximum);
    if (symmetric_) increase_directed(to, from, increment, maximum);
}

void SparsePheromone::reset(double value) {
    require(std::isfinite(value) && value >= 0, "重置值无效");
    std::fill(trails_.begin(), trails_.end(), value);
    default_ = value;
}

std::vector<double> SparsePheromone::product_cache(const DistanceFunction& distance, double beta) const {
    require(std::isfinite(beta) && beta > 0, "距离启发式 beta 必须正且有限");
    std::vector<double> products(trails_.size());
    for (Node from = 0; from < rows_.size(); ++from) {
        for (std::size_t j = 0; j < width_; ++j) {
            const double value = distance(from, rows_[from][j]);
            require(std::isfinite(value) && value >= 0, "启发式距离无效");
            const double heuristic = value > 0 ? 1.0 / std::pow(value, beta) : 1.0;
            products[from * width_ + j] = trails_[from * width_ + j] * heuristic;
        }
    }
    return products;
}

}  // namespace gp_faco
