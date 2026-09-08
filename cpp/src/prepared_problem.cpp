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
    const auto candidates_finished = Clock::now();
    p.preparation_profile.candidates_and_scales_seconds =
        std::chrono::duration<double>(candidates_finished - started).count();
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
    const auto nearest_finished = Clock::now();
    p.preparation_profile.nearest_neighbor_seconds =
        std::chrono::duration<double>(nearest_finished - candidates_finished).count();
    const auto distance = [&](Node a, Node b) { return p.distance(a, b); };
    CpuTour tour(initial, distance);
    std::vector<Node> checklist(initial);
    tour.checklist_two_opt(DistanceOrderedCandidates(p.ls, distance), checklist, c.initial_ls_evaluation_limit);
    const auto ls_finished = Clock::now();
    p.preparation_profile.initial_ls_seconds =
        std::chrono::duration<double>(ls_finished - nearest_finished).count();
    if (expired()) return false;
    p.initial_tour = tour.order(); p.initial_cost = tour.recomputed_cost();
    // 初始改进不能丢掉更早完成的廉价解；等成本保留较早的候选。
    if (p.cheap_cost <= p.initial_cost) { p.initial_tour = p.cheap_tour; p.initial_cost = p.cheap_cost; }
    const auto finished = Clock::now();
    p.preparation_profile.finalization_seconds = std::chrono::duration<double>(finished - ls_finished).count();
    p.preparation_seconds = std::chrono::duration<double>(finished - started).count();
    p.ready = true;
    return true;
}

void apply_candidate_graph(PreparedProblem& p, CandidateGraphSpec spec) {
    const auto started = Clock::now();
    require(p.ready && !p.graph, "固定图需要已完成的共同准备，且只能登记一次");
    const Node n = p.size();
    require(spec.common_initial_tour == p.initial_tour, "图输入改变了共同初始tour");
    SparseUndirectedGraph graph(n, spec.edges);
    require(graph.edges() == spec.edges.size() && graph.contains_tour(p.initial_tour),
            "图含重复边或遗漏共同初始tour边");
    require(graph.edges() <= static_cast<std::size_t>(n) * (p.settings.primary_width + 1),
            "实际图超过预登记的固定容量");
    const auto validate = [&](const CandidateRows& rows, Node width, bool in_graph, bool ordered) {
        require(rows.size() == n, "图枚举行节点数不符");
        for (Node a = 0; a < n; ++a) {
            require(rows[a].size() == width, "图枚举行必须使用固定槽位");
            bool padded = false;
            std::vector<Node> seen;
            std::pair<double, Node> previous{-1, 0};
            for (Node b : rows[a]) {
                if (b == n) { padded = true; continue; }
                require(!padded && b < n && b != a &&
                        std::find(seen.begin(), seen.end(), b) == seen.end(),
                        "图枚举含未知/重复节点、自环或非右侧padding");
                require(!in_graph || graph.contains(a, b), "主行或LS行成员不在完整图中");
                const std::pair<double, Node> current{p.distance(a, b), b};
                require(!ordered || previous <= current, "LS行必须按原问题距离和节点ID排序");
                previous = current; seen.push_back(b);
            }
        }
    };
    validate(spec.primary, p.settings.primary_width, true, false);
    validate(spec.backup, p.settings.backup_width, false, false);
    validate(spec.ls, p.settings.ls_width, true, true);
    for (Node a = 0; a < n; ++a) for (Node b : spec.backup[a]) if (b < n)
        require(std::find(spec.primary[a].begin(), spec.primary[a].end(), b) == spec.primary[a].end(),
                "主行和备用行重复了实际成员");
    p.primary = spec.primary; p.backup = spec.backup; p.ls = spec.ls;
    p.graph_spec = std::move(spec); p.graph = std::move(graph);
    p.graph_preparation_seconds = seconds(started);
    p.preparation_seconds += p.graph_preparation_seconds;
}

std::vector<Node> flattened(const CandidateRows& rows) {
    std::vector<Node> values;
    for (const auto& row : rows) values.insert(values.end(), row.begin(), row.end());
    return values;
}

}  // namespace gp_faco
