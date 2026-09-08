// CPU从实际父tour逐移动重放，核验新增边、有限缓存、LS视图及参考足迹。
#include "gp_faco/batch_engine.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using namespace gp_faco;
void verify_escape_cuda_primitives();
namespace {
std::uint64_t checked_moves = 0, checked_tours = 0, new_edges = 0, ls_new_edges = 0;
std::uint64_t restart_checks = 0, footprint_checks = 0, old_reactivations = 0, full_graph_footprints = 0;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> std::vector<T> part(const std::vector<T>& values, std::size_t at, Node n) {
    check(at + n <= values.size(), "诊断数组越界");
    return {values.begin() + at, values.begin() + at + n};
}
std::set<Edge> edges(const std::vector<Node>& tour) {
    std::set<Edge> result;
    for (Node i = 0; i < tour.size(); ++i) result.insert(std::minmax(tour[i], tour[(i + 1) % tour.size()]));
    return result;
}
PreparedProblem problem(Node n, FixedFacoSettings settings, bool full, Node variant = 0) {
    std::vector<double> xy(n * 2);
    for (Node i = 0; i < n; ++i) {
        xy[i * 2] = full ? (i == 0 ? 1 : 0) : ((i * 7919 + variant * 313) % 104729) / 104729.0;
        xy[i * 2 + 1] = full ? 0 : ((i * i * 13 + variant * 101 + 17) % 130363) / 130363.0;
    }
    auto p = make_cheap_problem(xy, settings); prepare_problem(p);
    auto members = edges(p.initial_tour);
    if (full) for (Node a = 0; a < n; ++a) for (Node b = a + 1; b < n; ++b) members.emplace(a, b);
    else for (Node a = 0; a < n; ++a) for (Node j = 0; j < 4 && j < p.primary[a].size(); ++j)
        members.insert(std::minmax(a, p.primary[a][j]));
    CandidateGraphSpec spec; spec.common_initial_tour = p.initial_tour; spec.edges.assign(members.begin(), members.end());
    for (Node a = 0; a < n; ++a) {
        std::vector<Node> row, backup;
        for (Node b = 0; b < n; ++b) if (a != b && members.count(std::minmax(a, b))) row.push_back(b);
        std::sort(row.begin(), row.end(), [&](Node x, Node y) {
            return std::make_pair(p.distance(a, x), x) < std::make_pair(p.distance(a, y), y);
        });
        auto primary = row, ls = row;
        primary.resize(p.settings.primary_width, n); ls.resize(p.settings.ls_width, n);
        for (Node b = 0; b < n; ++b) if (a != b && std::find(primary.begin(), primary.end(), b) == primary.end())
            backup.push_back(b);
        backup.resize(p.settings.backup_width, n);
        spec.primary.push_back(primary); spec.ls.push_back(ls); spec.backup.push_back(backup);
    }
    apply_candidate_graph(p, spec); return p;
}

void same(const BatchEvaluation& x, const BatchEvaluation& y) {
    check(x.completed_construction_steps == y.completed_construction_steps &&
          x.completed_ls_evaluations == y.completed_ls_evaluations &&
          x.completed_constraint_rejections == y.completed_constraint_rejections, "逐批工作量不一致");
    check(x.control_trace.size() == y.control_trace.size(), "轨迹长度不一致");
    for (Node c = 0; c < x.incumbents.size(); ++c)
        check(x.incumbents[c].tour == y.incumbents[c].tour && x.incumbents[c].cost == y.incumbents[c].cost,
              "重放或关闭Escape后的incumbent不一致");
    for (Node i = 0; i < x.control_trace.size(); ++i) {
        const auto& a = x.control_trace[i]; const auto& b = y.control_trace[i];
        check(a.actions == b.actions && a.masks == b.masks && a.features == b.features &&
            a.after_batch.tours == b.after_batch.tours && a.after_batch.archive == b.after_batch.archive &&
            a.after_batch.parent == b.after_batch.parent && a.after_batch.epoch == b.after_batch.epoch &&
            a.after_batch.trails == b.after_batch.trails && a.after_batch.products == b.after_batch.products,
            "Escape重放改变共同动作/随机流/信息素/完整tour");
        for (Node c = 0; c < a.after_batch.colonies.size(); ++c)
            check(a.after_batch.colonies[c].source_uniform == b.after_batch.colonies[c].source_uniform,
                  "Escape侵入普通随机流");
        if (!a.after_batch.ant_footprints.empty() && !b.after_batch.ant_footprints.empty())
            check(a.after_batch.ant_footprints == b.after_batch.ant_footprints &&
                a.after_batch.parent_footprints == b.after_batch.parent_footprints &&
                a.after_batch.archive_footprints == b.after_batch.archive_footprints &&
                a.after_batch.escape_ls_rows == b.after_batch.escape_ls_rows &&
                a.after_batch.escape_ls_replaced == b.after_batch.escape_ls_replaced, "重放足迹/视图不一致");
    }
}

void references(const ControlBatchTrace& trace, Node c, Node n, Node ants) {
    const auto& old = trace.before; const auto& start = trace.after_restart; const auto& end = trace.after_batch;
    const auto base = static_cast<std::size_t>(c) * n;
    const auto& state = end.colonies[c];
    if (trace.actions[c] >= 16) {
        const auto expected = part(old.archive_footprints, base * archive_capacity + trace.alternatives[c] * n, n);
        check(part(start.parent_footprints, base, n) == expected && part(start.epoch_footprints, base, n) == expected,
              "重启没有携带选中档案足迹"); ++restart_checks;
    } else check(part(start.parent_footprints, base, n) == part(old.parent_footprints, base, n), "保持动作丢失parent足迹");
    const auto best = part(end.ant_footprints, base * ants + state.iteration_best * n, n);
    const auto epoch = state.epoch_cost < start.colonies[c].epoch_cost ? best : part(start.epoch_footprints, base, n);
    const auto global = state.global_cost < start.colonies[c].global_cost ? best : part(start.global_footprints, base, n);
    check(part(end.epoch_footprints, base, n) == epoch && part(end.global_footprints, base, n) == global &&
          part(end.parent_footprints, base, n) == (state.source_is_epoch ? epoch : best), "reference/epoch/global足迹错位");

    // 用CPU重新选择档案来源索引；相同tour但不同足迹时，仍要求实际选中的那一份。
    const auto& ctrl = start.controls[c];
    const auto view = [&](Node entry) -> TourView {
        if (entry < archive_capacity) return {start.archive.data() + base * archive_capacity + entry * n,
            start.archive_positions.data() + base * archive_capacity + entry * n, n};
        return {end.tours.data() + base * ants + (entry - archive_capacity) * n,
            end.positions.data() + base * ants + (entry - archive_capacity) * n, n};
    };
    const auto cost = [&](Node entry) { return entry < archive_capacity ? ctrl.archive_cost[entry]
        : end.info[c * ants + entry - archive_capacity].final_cost; };
    const auto identity = [&](Node entry) { return entry < archive_capacity ? ctrl.archive_identity[entry]
        : end.ant_identities[c * ants + entry - archive_capacity]; };
    const auto less = [&](Node a, Node b) {
        if (cost(a) != cost(b)) return cost(a) < cost(b);
        if (identity(a) != identity(b)) return identity(a) < identity(b);
        return canonical_less(view(a), view(b));
    };
    std::vector<Node> chosen{state.global_cost < ctrl.tracked_global ? archive_capacity + state.iteration_best : 0};
    while (chosen.size() < archive_capacity) {
        Node selected = UINT32_MAX;
        for (Node candidate = 0; candidate < archive_capacity + ants; ++candidate) {
            if (candidate < archive_capacity && candidate >= ctrl.archive_size) continue;
            if (cost(candidate) > state.global_cost * (1 + archive_quality_band)) continue;
            bool duplicate = false;
            for (Node prior : chosen) duplicate |= identity(prior) == identity(candidate) && same_tour(view(prior), view(candidate));
            if (!duplicate && (selected == UINT32_MAX || less(candidate, selected))) selected = candidate;
        }
        if (selected == UINT32_MAX) break;
        chosen.push_back(selected);
    }
    check(chosen.size() == end.controls[c].archive_size, "CPU档案来源选择数不一致");
    for (Node slot = 0; slot < chosen.size(); ++slot) {
        const Node entry = chosen[slot];
        const auto expected = entry < archive_capacity
            ? part(start.archive_footprints, base * archive_capacity + entry * n, n)
            : part(end.ant_footprints, base * ants + (entry - archive_capacity) * n, n);
        check(part(end.archive_footprints, base * archive_capacity + slot * n, n) == expected, "档案scratch复制足迹错位");
        ++footprint_checks;
    }
}

void verify(const BatchEvaluation& result, const PreparedProblem& p, Node colonies, std::uint64_t fe, bool full) {
    const Node n = p.size(), ants = p.settings.ants, width = p.settings.ls_width;
    check(result.constraint_mode == ConstraintMode::Escape && result.count_limited &&
        result.total_tour_evaluations == fe * colonies && result.completed_batches == fe / ants &&
        result.charged_seconds == 0 && result.overrun_seconds == 0 && result.discarded_batches == 0,
        "Escape FE或无时间截止语义错误");
    for (const auto& trace : result.control_trace) for (Node c = 0; c < colonies; ++c) {
        references(trace, c, n, ants);
        const auto& end = trace.after_batch;
        for (Node ant = 0; ant < ants; ++ant) {
            const Node global_ant = c * ants + ant;
            const auto base = static_cast<std::size_t>(global_ant) * n;
            const auto& info = end.info[global_ant];
            CpuTour tour(part(trace.after_restart.parent, static_cast<std::size_t>(c) * n, n),
                         [&](Node a, Node b) { return p.distance(a, b); });
            std::set<Edge> cache;
            std::vector<std::uint8_t> anchors(n);
            const Node count = info.construction.nonidentity_relocations + info.local_search.accepted_moves;
            for (Node index = 0; index < count; ++index) {
                const auto& event = end.escape_moves[base * 2 + index];
                check(event.a < n && event.b < n && event.kind <= 2, "移动日志边界不正确");
                const auto before = edges(tour.order());
                if (event.kind == 0) {
                    check(index < info.construction.nonidentity_relocations && tour.successor(event.a) != event.b,
                          "恒等或LS移动伪装成构造扰动");
                    if (event.permit_novel) {
                        check(event.slot < p.settings.primary_width &&
                            std::find(p.backup[event.a].begin(), p.backup[event.a].end(), event.b) != p.backup[event.a].end() &&
                            std::find(p.primary[event.a].begin(), p.primary[event.a].end(), event.b) == p.primary[event.a].end(),
                            "新例外不是显式备用槽位来源");
                        for (Node node : {event.a, tour.successor(event.a), event.b, tour.predecessor(event.b), tour.successor(event.b)})
                            anchors[node] = 1;
                    }
                    tour.relocate(event.a, event.b);
                } else {
                    check(index >= info.construction.nonidentity_relocations && event.slot < width, "LS事件次序或槽位越界");
                    const auto at = (base + event.a) * width + event.slot;
                    check(end.escape_ls_rows[at] == event.b && end.escape_ls_replaced[at] == event.permit_novel &&
                          (!event.permit_novel || anchors[event.a]), "LS新例外来自未授权槽位/非当前锚点");
                    tour.reverse_section(event.kind == 1 ? tour.successor(event.a) : event.a,
                                         event.kind == 1 ? tour.successor(event.b) : event.b);
                }
                for (const auto& edge : edges(tour.order())) if (!before.count(edge) && !p.graph->contains(edge.first, edge.second) && !cache.count(edge)) {
                    check(event.permit_novel, "未授权移动引入图外新边"); cache.insert(edge); ++new_edges;
                    ls_new_edges += event.kind != 0;
                }
                check(cache.size() <= 64 && event.cache_size == cache.size(), "设备例外登记与独立完整边差集不一致");
                ++checked_moves;
            }
            check(tour.order() == part(end.tours, base, n), "完整移动重放得到不同tour");
            check(std::abs(tour.recomputed_cost() - info.final_cost) <= 1e-8 + 1e-12 * info.final_cost, "Escape设备最终成本错误");
            check(part(end.escape_anchors, base, n) == anchors, "扰动五端点集合错误或LS扩展了锚点");
            const auto& recorded = end.escape_caches[global_ant];
            std::set<Edge> actual_cache;
            for (Node i = 0; i < recorded.size; ++i) {
                const auto edge = recorded.edges[i]; check(edge.a < edge.b && actual_cache.emplace(edge.a, edge.b).second,
                    "缓存未规范化或重复登记");
            }
            check(actual_cache == cache && info.escape.new_edges == cache.size(), "缓存被污染/跨批继承/漏登记");
            const auto pending = part(end.pending, base * 5, info.checklist_size);
            old_reactivations += info.escape.old_view_reactivations;
            for (Node node = 0; node < n; ++node) {
                if (anchors[node] || trace.after_restart.parent_footprints[static_cast<std::size_t>(c) * n + node])
                    check(std::find(pending.begin(), pending.end(), node) != pending.end(), "候选视图变化未重新激活节点");
                const auto row = part(end.escape_ls_rows, (base + node) * width, width);
                const auto flags = part(end.escape_ls_replaced, (base + node) * width, width);
                const auto valid = std::count_if(p.ls[node].begin(), p.ls[node].end(), [&](Node x) { return x < n; });
                Node replacements = 0; std::set<Node> members;
                for (Node j = 0; j < width; ++j) {
                    if (j >= valid) { check(row[j] == n && !flags[j], "LS新增了有效槽位"); continue; }
                    check(row[j] < n && row[j] != node && members.insert(row[j]).second, "LS行重复/自身/无效");
                    const bool replacement = std::find(p.ls[node].begin(), p.ls[node].end(), row[j]) == p.ls[node].end();
                    check(flags[j] == replacement && (!replacement || (anchors[node] &&
                        std::find(p.backup[node].begin(), p.backup[node].end(), row[j]) != p.backup[node].end())), "LS来源标记错误");
                    replacements += replacement;
                    if (j) check(std::make_pair(p.distance(node, row[j - 1]), row[j - 1]) <
                        std::make_pair(p.distance(node, row[j]), row[j]), "LS未按真实距离/node ID排序");
                }
                check(replacements <= (valid + 3) / 4 && end.ant_footprints[base + node] == (replacements != 0), "足迹或替换上限错误");
                if (!anchors[node]) check(row == p.ls[node], "非当前扰动锚点改变了LS行");
                if (full && replacements) ++full_graph_footprints;
            }
            ++checked_tours;
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        verify_escape_cuda_primitives();
        FixedFacoSettings settings; settings.ants = 4;
        Program program; program.feature_spec_id = 2; program.length = 1; program.operand[0] = 4;
        BaselinePolicy baseline; baseline.mne_level = baseline.max_mne_level = 3;
        baseline.restart_mode = StaticRestart::Bernoulli; baseline.restart_probability = 1;
        BatchDiagnosticControls diagnostic; diagnostic.capture_control = true;
        const std::vector<BatchTask> tasks{{1, 17}, {1, 29}};
        const bool small = argc > 2 && std::string(argv[2]) == "--small";
        for (bool full : {true, false}) for (Node n : (full ? std::vector<Node>{31} : std::vector<Node>{7, 31, 500, 1000})) {
            if (small && n > 31) continue;
            const auto p = problem(n, settings, full);
            FacoBatchEngine escape(n, 2, settings, ConstraintMode::Escape), hard(n, 2, settings, ConstraintMode::Hard);
            escape.register_graph_problem(1, p.coordinates, *p.graph_spec);
            hard.register_graph_problem(1, p.coordinates, *p.graph_spec);
            const std::uint64_t fe = n <= 31 ? 64 : 16;
            auto disabled = diagnostic; disabled.disable_escape = true;
            const auto h = hard.evaluate_baseline_evaluations(tasks, fe, baseline, PreparationMode::CachedCharged, UINT32_MAX, diagnostic);
            const auto e = escape.evaluate_baseline_evaluations(tasks, fe, baseline, PreparationMode::CachedCharged, UINT32_MAX, disabled);
            check(h.allocated_device_bytes == e.allocated_device_bytes && h.reserved_escape_device_bytes == e.reserved_escape_device_bytes,
                  "Hard/Escape预留设备内存预算不匹配");
            same(h, e); verify(e, p, 2, fe, full);
            const auto actual = escape.evaluate_baseline_evaluations(tasks, fe, baseline, PreparationMode::CachedCharged, UINT32_MAX, diagnostic);
            verify(actual, p, 2, fe, full);
            escape.evaluate_program_evaluations(tasks, 8, program, PreparationMode::CachedCharged);
            auto delayed = diagnostic; delayed.delay_batch = 1; delayed.completion_delay_ms = 50; delayed.profile = true;
            const auto replay = escape.evaluate_baseline_evaluations(tasks, fe, baseline, PreparationMode::EndToEnd, UINT32_MAX, delayed);
            same(actual, replay); verify(replay, p, 2, fe, full);
            const auto gp = escape.evaluate_program_evaluations(tasks, 8, program, PreparationMode::CachedCharged, UINT32_MAX, diagnostic);
            verify(gp, p, 2, 8, full);
            const auto zero = escape.evaluate_program_evaluations(tasks, 0, program, PreparationMode::CachedCharged);
            check(zero.total_tour_evaluations == 0 && zero.incumbents[0].tour == p.initial_tour, "Escape零FE改变共同初解");
            bool rejected = false;
            try { escape.evaluate_program(tasks, 1, Program{}, PreparationMode::CachedCharged); }
            catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "Escape接受旧墙钟入口");
        }
        check(new_edges > 0 && ls_new_edges > 0 && restart_checks > 0 && old_reactivations > 0 && full_graph_footprints > 0,
              "缺少真实出口/LS重连/重启/失效重激活/无图外边足迹覆盖");
        if (argc >= 2) {
            std::ofstream out(argv[1]);
            out << "{\"status\":\"passed\",\"gpu_transaction_cases\":50,\"random_stream_prefixes\":150,\"verified_tours\":" << checked_tours << ",\"replayed_moves\":" << checked_moves
                << ",\"new_edges\":" << new_edges << ",\"ls_new_edges\":" << ls_new_edges << ",\"restart_checks\":" << restart_checks
                << ",\"archive_footprint_checks\":" << footprint_checks << ",\"old_view_reactivations\":" << old_reactivations
                << ",\"full_graph_footprint_nodes\":" << full_graph_footprints << "}\n";
        }
        std::cout << checked_tours << " Escape tours and " << checked_moves << " accepted moves replayed\n";
    } catch (const std::exception& error) { std::cerr << "FAILED: " << error.what() << '\n'; return 1; }
}
