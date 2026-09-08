// 共享的无标签准备流程；昂贵准备开始前保留廉价可行解。
#include "gp_faco/prepared_problem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace gp_faco {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
}

FixedFacoSettings normalized_settings(FixedFacoSettings c, Node n) {
    require(n >= 3 && n <= 10000 && c.ants > 0 && c.ants <= 4096 && c.primary_width >= 2 &&
            c.ls_width > 0 && std::isfinite(c.beta) && c.beta > 0 && c.retention >= 0 &&
            c.retention < 1 && c.p_best > 0 && c.p_best < 1 && c.epoch_source_probability >= 0 &&
            c.epoch_source_probability <= 1, "FACO配置或规模无效");
    c.primary_width = std::min(c.primary_width, n - 1);
    c.backup_width = std::min(c.backup_width, n - 1 - c.primary_width);
    c.ls_width = std::min(c.ls_width, n - 1);
    return c;
}

double PreparedProblem::distance(Node a, Node b) const {
    const double x = coordinates[a * 2] - coordinates[b * 2];
    const double y = coordinates[a * 2 + 1] - coordinates[b * 2 + 1];
    return std::sqrt(x*x + y*y);
}

PreparedProblem make_cheap_problem(std::vector<double> coordinates, FixedFacoSettings settings) {
    const auto started = Clock::now();
    require(coordinates.size() % 2 == 0 && coordinates.size() / 2 >= 3 &&
            coordinates.size() / 2 <= 10000, "坐标形状无效");
    for (double coordinate : coordinates) require(std::isfinite(coordinate), "坐标必须有限");
    PreparedProblem p;
    p.coordinates = std::move(coordinates);
    p.settings = normalized_settings(settings, p.size());
    p.cheap_tour.resize(p.size());
    std::iota(p.cheap_tour.begin(), p.cheap_tour.end(), 0);
    CpuTour cheap(p.cheap_tour, [&](Node a, Node b) { return p.distance(a, b); });
    p.cheap_cost = cheap.cost();
    require(std::isfinite(p.cheap_cost) && p.cheap_cost > 0, "廉价可行解成本无效");
    p.cheap_seconds = seconds(started);
    return p;
}

bool prepare_problem(PreparedProblem& p, const std::function<bool()>& stop_requested) {
    require(!p.ready, "实例已完成准备");
    const auto started = Clock::now();
    const auto expired = [&]() { return stop_requested && stop_requested(); };
    const Node n = p.size();
    const auto& c = p.settings;
    p.primary.assign(n, {}); p.backup.assign(n, {}); p.ls.assign(n, {});
    const Node scale_width = std::min<Node>(8, n - 1);
    const Node needed = std::max({c.primary_width + c.backup_width, c.ls_width, scale_width});
    p.local_scale.assign(n, 0);
    double min_x = p.coordinates[0], max_x = min_x, min_y = p.coordinates[1], max_y = min_y;
    for (Node i = 0; i < n; ++i) {
        min_x = std::min(min_x, p.coordinates[i * 2]); max_x = std::max(max_x, p.coordinates[i * 2]);
        min_y = std::min(min_y, p.coordinates[i * 2 + 1]); max_y = std::max(max_y, p.coordinates[i * 2 + 1]);
    }
    p.scale_epsilon = 1e-12 * std::max({max_x - min_x, max_y - min_y, 1.0});
    for (Node a = 0; a < n; ++a) {
        if (expired()) return false;
        // 一次仅保留一行O(n)临时距离，不分配n²矩阵。
        std::vector<std::pair<double, Node>> row;
        row.reserve(n - 1);
        for (Node b = 0; b < n; ++b) if (a != b) {
            const double d = p.distance(a, b);
            require(std::isfinite(d), "坐标距离溢出");
            row.emplace_back(d, b);
        }
        std::partial_sort(row.begin(), row.begin() + needed, row.end());
        // 独立于候选先验的真实最近8邻居摘要；在已有距离行准备中一次计算。
        for (Node j = 0; j < scale_width; ++j) p.local_scale[a] += row[j].first / scale_width;
        for (Node j = 0; j < c.primary_width; ++j) p.primary[a].push_back(row[j].second);
        for (Node j = c.primary_width; j < c.primary_width + c.backup_width; ++j)
            p.backup[a].push_back(row[j].second);
        for (Node j = 0; j < c.ls_width; ++j) p.ls[a].push_back(row[j].second);
    }
    std::vector<Node> initial;
    std::vector<bool> visited(n, false);
    Node current = 0;
    for (Node i = 0; i < n; ++i) {
        if (expired()) return false;
        initial.push_back(current); visited[current] = true;
        Node next = n; double shortest = std::numeric_limits<double>::infinity();
        for (Node b = 0; b < n; ++b) if (!visited[b] && p.distance(current, b) < shortest) {
            shortest = p.distance(current, b); next = b;
        }
        current = next;
    }
    const auto distance = [&](Node a, Node b) { return p.distance(a, b); };
    CpuTour tour(initial, distance);
    std::vector<Node> checklist(initial);
    tour.checklist_two_opt(DistanceOrderedCandidates(p.ls, distance), checklist, c.initial_ls_evaluation_limit);
    if (expired()) return false;
    p.initial_tour = tour.order(); p.initial_cost = tour.recomputed_cost();
    // 初始改进不能丢掉更早完成的廉价解；等成本保留较早的候选。
    if (p.cheap_cost <= p.initial_cost) { p.initial_tour = p.cheap_tour; p.initial_cost = p.cheap_cost; }
    p.preparation_seconds = seconds(started);
    p.ready = true;
    return true;
}

std::vector<Node> flattened(const CandidateRows& rows) {
    std::vector<Node> values;
    for (const auto& row : rows) values.insert(values.end(), row.begin(), row.end());
    return values;
}

}  // namespace gp_faco
