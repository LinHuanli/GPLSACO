// 对照直接编译进来的作者实现，而非另写一个模仿实现作为 oracle。
// 原生源码（MIT）及源文件指纹见 provenance/THIRD_PARTY_NOTICES.md。
#define main gpfaco_unused_native_main
#include "faco.cpp"
#undef main

#include "gp_faco/faco_cpu.hpp"

#include <numeric>
#include <random>
#include <set>

namespace {

using gp_faco::Node;
nlohmann::json report;
nlohmann::json context;
double maximum_cost_error = 0;

void verify(bool condition, const std::string& message) {
    // Release 构建也执行验收，不能用会被 NDEBUG 删除的 assert。
    if (!condition) throw std::runtime_error(message + " context=" + context.dump());
}

void near(double left, double right, const std::string& message, double tolerance = 1e-8) {
    verify(std::isfinite(left) && std::isfinite(right) && std::abs(left - right) <= tolerance,
           message + " delta=" + std::to_string(left - right));
}

std::unique_ptr<ProblemInstance> fixture(Node n, bool integer = false, bool repeated = false) {
    std::mt19937_64 random(4129 + n);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<std::pair<double, double>> points(n);
    for (auto& point : points) point = {uniform(random), uniform(random)};
    if (repeated) for (Node i = 3; i < n; i += 3) points[i] = points[0];
    std::vector<double> matrix(static_cast<std::size_t>(n) * n);
    for (Node i = 0; i < n; ++i) {
        for (Node j = 0; j < n; ++j) {
            const double value = std::hypot(points[i].first - points[j].first,
                                            points[i].second - points[j].second);
            matrix[i * n + j] = integer ? std::floor(value * 10000 + 0.5) : value;
        }
    }
    auto problem = std::make_unique<ProblemInstance>(n, EXPLICIT, std::vector<Vec2d>{}, matrix, true);
    problem->compute_nn_lists(std::min<Node>(32, n - 1));
    return problem;
}

gp_faco::CandidateRows candidate_rows(const ProblemInstance& problem, Node width) {
    gp_faco::CandidateRows rows(problem.dimension_);
    for (Node i = 0; i < problem.dimension_; ++i) {
        for (Node j : problem.get_nearest_neighbors(i, width)) rows[i].push_back(j);
    }
    return rows;
}

void check_tour(const gp_faco::CpuTour& migrated, const Route& native, const ProblemInstance& problem) {
    verify(migrated.order() == native.route_, "tour 数组与原生不同");
    verify(migrated.positions() == native.positions_, "inverse positions 与原生不同");
    verify(problem.is_route_valid(migrated.order()), "输出不是合法排列");
    const double independent = problem.calculate_route_length(migrated.order());
    maximum_cost_error = std::max(maximum_cost_error, std::abs(independent - migrated.cost()));
    near(migrated.cost(), native.cost_, "增量成本与原生不同");
    near(migrated.cost(), independent, "增量成本与完整重算不同");
}

void relocation_and_reversal() {
    std::uint64_t relocations = 0, reversals = 0;
    for (Node n = 3; n <= 7; ++n) {
        auto problem = fixture(n);
        auto distance = problem->get_distance_fn();
        std::vector<Node> order(n);
        std::iota(order.begin(), order.end(), 0);
        do {
            const double cost = problem->calculate_route_length(order);
            for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) {
                gp_faco::CpuTour migrated(order, distance);
                Route native(order, distance, cost);
                if (a != b) {
                    context = {{"phase", "relocation"}, {"n", n}, {"order", order}, {"a", a}, {"b", b}};
                    const auto reconnection = migrated.relocation_edges(a, b);
                    migrated.relocate(a, b);
                    native.relocate_node(a, b);
                    check_tour(migrated, native, *problem);
                    verify(migrated.successor(a) == b, "重定位后不是目标后继");
                    if (reconnection.nonidentity) {
                        std::multiset<std::pair<Node, Node>> expected, actual;
                        const auto edge = [](Node x, Node y) {
                            return std::pair<Node, Node>{std::min(x, y), std::max(x, y)};
                        };
                        for (Node i = 0; i < n; ++i) expected.insert(edge(order[i], order[(i+1)%n]));
                        for (const auto& removed : reconnection.removed) {
                            const auto found = expected.find(edge(removed.first, removed.second));
                            verify(found != expected.end(), "删除边不在原 tour 中");
                            expected.erase(found);
                        }
                        for (const auto& added : reconnection.added) expected.insert(edge(added.first, added.second));
                        for (Node i = 0; i < n; ++i) actual.insert(edge(migrated.order()[i], migrated.order()[(i+1)%n]));
                        verify(expected == actual, "三条重连边未完整描述重定位");
                    }
                    ++relocations;
                }
                gp_faco::CpuTour reversed(order, distance);
                context = {{"phase", "reversal"}, {"n", n}, {"order", order}, {"a", a}, {"b", b}};
                Route native_reversed(order, distance, cost);
                reversed.reverse_section(a, b);
                native_reversed.flip_route_section(a, b);
                // 原生低层 flip 本身不维护cost，测试公共反转接口时独立重算。
                native_reversed.cost_ = problem->calculate_route_length(native_reversed.route_);
                check_tour(reversed, native_reversed, *problem);
                ++reversals;
            }
        } while (std::next_permutation(order.begin(), order.end()));
    }
    report["exhaustive_relocations_n3_to7"] = relocations;
    report["exhaustive_reversals_n3_to7"] = reversals;
}

void construction_and_local_search() {
    std::mt19937_64 random(7331);
    std::uint64_t cases = 0, transitions = 0, reactivations = 0, evaluations = 0, limited_cases = 0;
    bool mne_differs_from_final_edges = false;
    bool restored_edge_was_previously_changed = false;
    for (Node n : {8u, 17u, 100u, 500u, 1000u}) {
        for (int metric_case = 0; metric_case < 3; ++metric_case) {
            auto problem = fixture(n, metric_case == 1, metric_case == 2);
            const Node width = std::min<Node>(20, n - 1);
            const auto rows = candidate_rows(*problem, width);
            gp_faco::DistanceOrderedCandidates candidates(rows, problem->get_distance_fn());
            for (int repeat = 0; repeat < 32; ++repeat) {
                std::vector<Node> order(n), choices(n);
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), random);
                std::iota(choices.begin(), choices.end(), 0);
                std::shuffle(choices.begin(), choices.end(), random);
                const Node start = choices.front();
                const Node target = std::array<Node, 4>{2, 4, 8, 16}[repeat % 4];
                gp_faco::CpuTour initial(order, problem->get_distance_fn());
                gp_faco::FocusedConstruction migrated(initial, start, target);
                Route reference(order, problem->get_distance_fn(), initial.cost());
                Route native(reference);
                std::vector<Node> checklist;
                Node current = start, mne = 0, position = 1;
                int previous_difference = 0;
                while (!migrated.done()) {
                    const Node selected = choices[position++];
                    context = {{"phase", "construction"}, {"n", n}, {"metric", metric_case},
                               {"repeat", repeat}, {"position", position}, {"selected", selected}};
                    const Node old_previous = native.get_pred(selected);
                    native.relocate_node(current, selected);
                    if (!reference.contains_edge(current, selected)) {
                        ++mne;
                        for (Node node : {current, selected, old_previous}) {
                            if (std::find(checklist.begin(), checklist.end(), node) == checklist.end()) {
                                checklist.push_back(node);
                            }
                        }
                    }
                    migrated.step(selected);
                    check_tour(migrated.tour(), native, *problem);
                    verify(migrated.stats().mne == mne, "MNE 计数与原生条件不同");
                    verify(migrated.checklist() == checklist, "构造 checklist 不同");
                    const int difference = count_diff_edges(native, reference);
                    restored_edge_was_previously_changed |= difference < previous_difference;
                    previous_difference = difference;
                    current = selected;
                    ++transitions;
                }
                verify(mne >= target || position == n, "构造停止条件不符");
                mne_differs_from_final_edges |= static_cast<int>(mne) != count_diff_edges(native, reference);
                native.two_opt_nn(*problem, checklist, width);
                context["phase"] = "local_search";
                const auto stats = migrated.tour().checklist_two_opt(candidates, migrated.checklist());
                check_tour(migrated.tour(), native, *problem);
                verify(migrated.checklist() == checklist, "LS 重新激活顺序不同");
                verify(stats.accepted_moves <= n, "超出原生接受移动上限");
                reactivations += stats.reactivations;
                evaluations += stats.move_evaluations;
                // 成员经先验打乱后应恢复实际距离视图；此检查限无等距的fixture。
                if (metric_case == 0) {
                    auto shuffled_rows = rows;
                    for (auto& row : shuffled_rows) std::reverse(row.begin(), row.end());
                    gp_faco::DistanceOrderedCandidates restored(shuffled_rows, problem->get_distance_fn());
                    for (Node node = 0; node < n; ++node) verify(restored[node] == candidates[node], "LS距离视图未恢复");
                }
                gp_faco::CpuTour limited(initial);
                std::vector<Node> all(order);
                const auto before = limited.order();
                const auto limited_stats = limited.checklist_two_opt(candidates, all, 0);
                verify(limited.order() == before && limited_stats.move_evaluations == 0 &&
                       limited_stats.evaluation_limit_reached, "零评价预算仍改变了 tour");
                for (std::uint64_t cap : {1u, 7u, 127u}) {
                    gp_faco::CpuTour budgeted(initial);
                    std::vector<Node> pending(order);
                    const auto bounded = budgeted.checklist_two_opt(candidates, pending, cap);
                    verify(bounded.move_evaluations <= cap && bounded.accepted_moves <= n,
                           "LS 越过统一检查/接受上限");
                    verify(problem->is_route_valid(budgeted.order()), "LS 截断破坏 tour 排列");
                    const auto cost = problem->calculate_route_length(budgeted.order());
                    near(budgeted.cost(), cost, "LS 截断后增量成本不符");
                    verify(cost <= initial.cost() + 1e-8, "LS 截断接受劣化移动");
                    ++limited_cases;
                }
                ++cases;
            }
        }
    }
    verify(mne_differs_from_final_edges, "缺少 MNE 与真实新边数不同的反例");
    verify(restored_edge_was_previously_changed, "缺少构造撤销早先改边的反例");
    verify(reactivations > 0, "没有覆盖 LS 端点重新激活");
    report["construction_ls_cases"] = cases;
    report["construction_transitions"] = transitions;
    report["ls_reactivations"] = reactivations;
    report["ls_move_evaluations"] = evaluations;
    report["ls_finite_evaluation_limit_cases"] = limited_cases;
    report["mne_final_edge_count_difference_observed"] = mne_differs_from_final_edges;
    report["construction_can_restore_changed_edges"] = restored_edge_was_previously_changed;
}

void selection_tests() {
    context = {{"phase", "selection"}};
    auto problem = fixture(16);
    HeuristicData heuristic(*problem, 1.0);
    const Node current = 0;
    std::vector<Node> primary{1, 2, 3, 4}, backup{5, 6, 7};
    std::vector<double> products{1, 2, 3, 4};
    std::vector<double> cache(16 * primary.size(), 0);
    std::copy(products.begin(), products.end(), cache.begin());
    NodeList native_primary(primary.data(), primary.size()), native_backup(backup.data(), backup.size());
    MatrixPheromone unused(16, 1.0, true);
    std::uint64_t comparisons = 0;
    for (unsigned bits = 0; bits < 128; ++bits) {
        Bitmask mask(16);
        std::vector<std::uint8_t> visited(16, 0);
        mask.set_bit(0); visited[0] = 1;
        for (Node i = 1; i <= 7; ++i) if ((bits >> (i - 1)) & 1u) {
            mask.set_bit(i); visited[i] = 1;
        }
        const auto count = std::count_if(primary.begin(), primary.end(), [&](Node n) { return !visited[n]; });
        get_rng().init(1447 + bits);
        Random mirror; mirror.init(1447 + bits);
        for (int i = 0; i < 100; ++i) {
            const double uniform = count > 1 ? mirror.next_float() : 0.0;
            const Node expected = select_next_node(unused, heuristic, native_primary, cache,
                                                   native_backup, current, mask);
            const auto selected = gp_faco::select_next(current, primary, products, backup, visited,
                                                       problem->get_distance_fn(), uniform);
            verify(selected.node == expected, "正权重 roulette/备用/全回退与原生不同");
            ++comparisons;
        }
    }
    std::vector<std::uint8_t> visited(16, 1);
    visited[3] = 0;
    auto selected = gp_faco::select_next(0, primary, std::vector<double>(4, 0), backup,
                                         visited, problem->get_distance_fn(), 0.5);
    verify(selected.node == 3 && selected.stage == gp_faco::SelectionStage::ZeroWeightPrimary,
           "零权重单候选未返回合法节点");
#ifdef NDEBUG
    Bitmask mask(16);
    for (Node i = 0; i < 16; ++i) if (visited[i]) mask.set_bit(i);
    std::fill(cache.begin(), cache.end(), 0);
    const Node source_zero = select_next_node(unused, heuristic, native_primary, cache,
                                               native_backup, 0, mask);
    verify(source_zero == 0, "锁定原生零权重退化行为发生改变");
    report["native_zero_weight_single_candidate_returns_current"] = true;
#endif
    visited.assign(16, 0); visited[0] = 1;
    // 分层均匀网格检查实际概率，独立于原生 RNG；同时覆盖零权重和极小权重。
    for (const auto& weights : {products, std::vector<double>(4, 0),
                               std::vector<double>{1e-300, 2e-300, 3e-300, 4e-300}}) {
        std::array<int, 4> counts{};
        for (int i = 0; i < 10000; ++i) {
            selected = gp_faco::select_next(0, primary, weights, backup, visited,
                                            problem->get_distance_fn(), (i + 0.5) / 10000);
            verify(selected.node >= 1 && selected.node <= 4, "主候选退化回退离开候选集合");
            ++counts[selected.node - 1];
        }
        const auto expected = weights[0] == 0 ? std::array<int, 4>{2500, 2500, 2500, 2500}
                                              : std::array<int, 4>{1000, 2000, 3000, 4000};
        verify(counts == expected, "roulette/零权重回退的频数不符");
        for (double u : {0.0, std::nextafter(1.0, 0.0)}) {
            selected = gp_faco::select_next(0, primary, weights, backup, visited,
                                            problem->get_distance_fn(), u);
            verify(selected.node == (u == 0 ? 1 : 4), "roulette 端点不符");
        }
    }
    visited.assign(16, 1);
    selected = gp_faco::select_next(0, primary, products, backup, visited, problem->get_distance_fn(), 0);
    verify(selected.stage == gp_faco::SelectionStage::Exhausted, "访问耗尽未终止");
    report["positive_weight_selection_comparisons"] = comparisons;
    report["stratified_roulette_probability_checks"] = 30000;
}

void pheromone_tests() {
    context = {{"phase", "pheromone"}};
    gp_faco::CandidateRows rows{{1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 1}};
    std::mt19937_64 random(1217);
    for (bool symmetric : {false, true}) {
        CandListPheromone native(rows, 1.0, symmetric);
        gp_faco::SparsePheromone migrated(rows, 1.0, symmetric);
        for (int i = 0; i < 128; ++i) {
            migrated.evaporate(0.5, 0.01);
            native.evaporate(0.5, 0.01);
            const Node from = random() % 5, to = random() % 5;
            migrated.deposit(from, to, 0.2, 0.7);
            native.increase(from, to, 0.2, 0.7);
            for (Node a = 0; a < 5; ++a) for (Node b = 0; b < 5; ++b) {
                near(migrated.get(a, b), native.get(a, b), "稀疏信息素与原生不同", 1e-15);
            }
        }
        const auto old_default = native.default_pheromone_value_;
        native.set_all_trails(0.8);
        verify(native.default_pheromone_value_ == old_default, "原生set_all_trails行为变化");
        migrated.reset(0.8);
        near(migrated.default_value(), 0.8, "reset漏掉图外default", 0);
        native.default_pheromone_value_ = 0.8;  // 声明的重启扩展，不冒充原生set_all_trails。
        for (Node a = 0; a < 5; ++a) for (Node b = 0; b < 5; ++b) {
            near(migrated.get(a, b), native.get(a, b), "完整reset后信息素不同", 0);
        }
        auto distance = [](Node a, Node b) { return double(a > b ? a - b : b - a); };
        const auto products = migrated.product_cache(distance, 1.0);
        for (Node a = 0; a < 5; ++a) for (std::size_t j = 0; j < 2; ++j) {
            near(products[a * 2 + j], 0.8 / distance(a, rows[a][j]), "reset产品缓存不符", 1e-15);
        }
    }
    std::uint64_t limits_count = 0;
    for (Node count : {2u, 4u, 16u, 32u}) for (double retention : {0.0, 0.5, 0.9}) {
        for (double p : {0.01, 0.1, 0.5}) {
            const auto native = calc_trail_limits_cl(1000, count, p, retention, 123.4);
            const auto migrated = gp_faco::candidate_trail_limits(count, p, retention, 123.4);
            near(native.min_, migrated.minimum, "tau_min公式不同", 1e-15);
            near(native.max_, migrated.maximum, "tau_max公式不同", 1e-15);
            ++limits_count;
        }
    }
    report["pheromone_update_sequences"] = 256;
    report["trail_limit_comparisons"] = limits_count;
    report["default_reset_extension_checked"] = true;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        omp_set_num_threads(1);
        init_random_number_generators(314159);
        relocation_and_reversal();
        construction_and_local_search();
        selection_tests();
        pheromone_tests();
        report["maximum_absolute_cost_recompute_error"] = maximum_cost_error;
        report["status"] = "passed";
        report["scope"] = "CPU operation-level reference; full CUDA Engine not tested";
        if (argc == 2) {
            std::ofstream stream(argv[1]);
            if (!stream) throw std::runtime_error("无法写入测试结果");
            stream << report.dump(2) << '\n';
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
