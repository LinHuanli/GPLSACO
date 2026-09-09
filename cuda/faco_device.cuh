// 共同FACO构造/LS设备实现；诊断与真实选点共享此内核。
// 原始Route/LS语义 Copyright (c) 2024 RSkinderowicz，MIT许可见provenance。
#pragma once

#include "edge_observation.cuh"
#include "gp_faco/faco_cuda_diagnostic.hpp"
#include "gp_faco/edge_constraints.hpp"
#include "gp_faco/profiling.hpp"
#include "escape_device.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace gp_faco::cuda_detail {

#ifndef GPFACO_WARPS_PER_BLOCK
#define GPFACO_WARPS_PER_BLOCK 2
#endif
#ifndef GPFACO_PARALLEL_LS_PREFIX
#define GPFACO_PARALLEL_LS_PREFIX 1
#endif
#ifndef GPFACO_SHARED_ANT_FLAGS
#define GPFACO_SHARED_ANT_FLAGS 1
#endif
constexpr unsigned ant_warps_per_block = GPFACO_WARPS_PER_BLOCK;
constexpr unsigned ant_shared_bytes_per_node = 6 + 2 * GPFACO_SHARED_ANT_FLAGS;

struct MatrixDistance {
    const double* values;
    Node n;
    __device__ MatrixDistance for_ant(Node) const { return *this; }
    __device__ double operator()(Node a, Node b) const {
        return values[static_cast<std::size_t>(a) * n + b];
    }
};

struct ExplicitChoices {
    static constexpr bool shared_parent = false;
    const Node* visits;
    __device__ ExplicitChoices for_ant(Node, Node) const { return *this; }
    __device__ Node ant_index(Node ant) const { return ant; }
    __device__ std::size_t parent_offset(Node ant, Node n) const {
        return static_cast<std::size_t>(ant) * n;
    }
    __device__ std::size_t candidate_offset(Node, Node, Node) const { return 0; }
    __device__ Node start(Node ant, Node n) const { return visits[ant * n]; }
    template<class Distance, class Allowed>
    __device__ Node next(Node ant, Node current, const std::uint8_t*, Node step, Node n, Distance,
                        const Node* tour, const Node* position, const Allowed& allowed) const {
        const Node node = visits[ant * n + step];
        if constexpr (std::is_same_v<Allowed, UnrestrictedEdges>) return node;
        else return relocation_allowed(tour, position, n, current, node, allowed) ? node : n;
    }
};

struct State {
    Node current, selected, old_previous, node_position, target_position;
    Node first, last, a, a_next, a_previous, best_slot, best_kind;
    Node move[4], allowed[2], pending_size, pending_head;
    bool nonidentity, stopped;
    double cost, accumulated_gain, best_gain, current_edge[2];
    ConstructionStats construction;
    LocalSearchStats ls;
};

template<class Distance>
__device__ double distance(const Distance& matrix, Node, Node a, Node b) {
    return matrix(a, b);
}

template<class TourNode>
__device__ inline Node successor(const TourNode* tour, const TourNode* position, Node n, Node node) {
    return tour[(position[node] + 1) % n];
}

template<class TourNode>
__device__ inline Node predecessor(const TourNode* tour, const TourNode* position, Node n, Node node) {
    return tour[(position[node] + n - 1) % n];
}

__device__ inline void append_if_absent(Node* pending, Node from, Node node, Node& length,
                                        std::uint8_t* queued = nullptr, Node capacity = 0) {
    if (queued) {
        if (queued[node]) return;
        queued[node] = 1;
    } else {
        for (Node i = from; i < length; ++i) if (pending[capacity ? i % capacity : i] == node) return;
    }
    // queued 保证同时待处理的节点最多 n 个；环形存储保留原 FIFO 和重新入队顺序。
    pending[capacity ? length % capacity : length] = node;
    ++length;
}

template<bool WarpTours>
__device__ inline void ant_group_sync() {
    if constexpr (WarpTours) __syncwarp();
    else __syncthreads();
}

template<class Distance, class Choices, class Allowed = UnrestrictedEdges, bool Profile = false, bool WarpTours = false,
         bool CompactWorkspace = false>
__global__ void construct_and_search(
    Distance distances, const Node* all_candidates, Node n, Node width,
    const Node* parent_tours, Choices choice_views, const Node* targets,
    std::uint64_t evaluation_limit, Node* all_tours, Node* all_positions,
    Node* all_parent_positions, Node* all_scratch, Node* all_pending,
    double* all_gains, Node* construction_tours, FacoDiagnosticInfo* output,
    std::uint8_t* all_visited, Allowed all_allowed = {}, AntPhaseCycles* profile_cycles = nullptr,
    Node* construction_new_edges = nullptr, std::uint8_t* all_queued = nullptr,
    const double* all_candidate_distances = nullptr, const double* cached_parent_costs = nullptr) {
    const Node lane = WarpTours ? threadIdx.x % 32 : threadIdx.x;
    const Node group_width = WarpTours ? 32 : blockDim.x;
    const Node group = WarpTours ? threadIdx.x / 32 : 0;
    std::uint64_t tick0 = 0, tick1 = 0, tick2 = 0, tick3 = 0;
    if constexpr (Profile) { if (lane == 0) tick0 = clock64(); }
    const auto ant = WarpTours ? blockIdx.x * ant_warps_per_block + group : blockIdx.x;
    const auto base = static_cast<std::size_t>(ant) * n;
    const auto matrix = distances.for_ant(ant);
    const auto choices = choice_views.for_ant(ant, n);
    const auto local_ant = choice_views.ant_index(ant);
    const auto allowed = all_allowed.for_ant(ant, n);
    constexpr bool escape = std::is_same_v<std::decay_t<decltype(allowed)>, EscapeEdges>;
    const Node* candidates = all_candidates + choice_views.candidate_offset(ant, n, width);
    const Node* parent = parent_tours + choice_views.parent_offset(ant, n);
    std::uint8_t* visited = all_visited + base;
    auto* queued = all_queued ? all_queued + base : nullptr;
    const double* candidate_distances = all_candidate_distances ?
        all_candidate_distances + choice_views.candidate_offset(ant, n, width) : nullptr;
    // 主实验 n<=1500 时用 16 位共享路线/位置/交换缓冲，避免每次翻转往返全局显存。
    using TourNode = std::conditional_t<WarpTours, std::uint16_t, Node>;
    extern __shared__ std::uint16_t shared_tours[];
    TourNode* tour;
    TourNode* position;
    TourNode* scratch;
    if constexpr (WarpTours) {
        tour = shared_tours + static_cast<std::size_t>(group) * n * 3;
        position = tour + n;
        scratch = position + n;
        if constexpr (GPFACO_SHARED_ANT_FLAGS) {
            // 每只蚂蚁独有的标记与路线一起驻留；末尾回写，保持诊断快照语义。
            auto* flags = reinterpret_cast<std::uint8_t*>(shared_tours + ant_warps_per_block * n * 3);
            visited = flags + static_cast<std::size_t>(group) * n * 2;
            if (all_queued) queued = visited + n;
        }
    } else {
        tour = all_tours + base;
        position = all_positions + base;
        scratch = all_scratch + base;
    }
    static_assert(!CompactWorkspace || WarpTours, "紧凑工作区依赖 warp 共享路线");
    Node* parent_position = all_parent_positions + (CompactWorkspace ? choice_views.parent_offset(ant, n) : base);
    Node* pending = all_pending + base * (CompactWorkspace ? 1 : 5);
    const Node pending_capacity = CompactWorkspace ? n : 0;
    double* gains = all_gains + static_cast<std::size_t>(ant) * width * 2;
    __shared__ State states[WarpTours ? ant_warps_per_block : 1];
    auto& state = states[group];
    const auto ls_allowed = [&](Node node, Node slot) {
        if constexpr (escape) return choices.ls_allowed(node, slot, width, allowed);
        else return allowed;
    };
    for (Node i = lane; i < n; i += group_width) {
        tour[i] = parent[i];
        position[parent[i]] = i;
        if constexpr (!CompactWorkspace) parent_position[parent[i]] = i;
        visited[i] = 0;
        if (queued) queued[i] = 0;
        if constexpr (escape) choices.anchors[i] = choices.footprint[i] = 0;
    }
    if (lane == 0) {
        state = {};
        output[ant].escape = {};
        if constexpr (escape) allowed.cache->reset();
        state.current = choices.start(local_ant, n);
        output[ant].start_node = state.current;
        if (cached_parent_costs) state.cost = cached_parent_costs[choice_views.parent_offset(ant, n) / n];
        else for (Node i = 0; i < n; ++i)
            state.cost += distance(matrix, n, parent[(i + n - 1) % n], parent[i]);
    }
    ant_group_sync<WarpTours>();

    if (lane == 0) visited[state.current] = 1;
    ant_group_sync<WarpTours>();
    if constexpr (Profile) { if (lane == 0) tick1 = clock64(); }

    // 一蚂蚁一block，移动前捕获端点；scratch隔离并行读取和覆盖。
    while (state.construction.mne < targets[ant] && state.construction.steps + 1 < n) {
        // 所有warp先读完循环条件，线程0才能增加该条件所读取的steps。
        ant_group_sync<WarpTours>();
        if (lane == 0) {
            state.selected = choices.next(local_ant, state.current, visited, state.construction.steps + 1,
                                          n, matrix, tour, position, allowed);
            state.stopped = state.selected >= n;
            if (state.stopped) state.construction.legal_exhausted = true;
            // 同一线程完成选择与端点捕获，随后只需一次block发布边界。
            else {
                if constexpr (escape) {
                    const auto proposal = prepare_escape_relocation(tour, position, n, state.current,
                        state.selected, allowed.with_permission(choices.selected_novel));
                    if (!commit_escape_edges(*allowed.cache, proposal)) asm("trap;");
                    choices.stats->new_edges += proposal.count;
                }
                ++state.construction.steps;
                visited[state.selected] = 1;
                state.old_previous = predecessor(tour, position, n, state.selected);
                state.node_position = position[state.selected];
                state.target_position = position[state.current];
                const Node after = successor(tour, position, n, state.selected);
                const Node target_after = successor(tour, position, n, state.current);
                state.nonidentity = target_after != state.selected;
                if (state.nonidentity) {
                    if constexpr (escape) choices.record_move(state.construction.nonidentity_relocations, n,
                        {0, state.current, state.selected, choices.selected_slot, allowed.cache->size, choices.selected_novel});
                    if constexpr (escape) if (choices.selected_novel) {
                        // 只登记当前构造扰动的五个端点；后续LS不扩展许可锚点集合。
                        choices.anchors[state.current] = choices.anchors[target_after] = 1;
                        choices.anchors[state.selected] = choices.anchors[state.old_previous] = choices.anchors[after] = 1;
                        ++choices.stats->escape_relocations;
                    }
                    ++state.construction.nonidentity_relocations;
                    state.cost += -distance(matrix, n, state.old_previous, state.selected)
                                  -distance(matrix, n, state.selected, after)
                                  -distance(matrix, n, state.current, target_after)
                                  +distance(matrix, n, state.old_previous, after)
                                  +distance(matrix, n, state.current, state.selected)
                                  +distance(matrix, n, state.selected, target_after);
                }
            }
        }
        ant_group_sync<WarpTours>();
        // 全声明枚举均无合法移动时共同退出；没有visited[n]或隐式图外回退。
        if (state.stopped) break;
        if (state.nonidentity) {
            const Node a = state.target_position, b = state.node_position;
            const Node first = a < b ? a + 1 : b, last = a < b ? b : a;
            // 重定位只改变闭区间[first,last]，区间外路线和位置都无需读写。
            for (Node i = first + lane; i <= last; i += group_width) {
                Node source = i;
                if (a < b && i > a && i <= b) source = i == a + 1 ? b : i - 1;
                if (a > b && i >= b && i <= a) source = i == a ? b : i + 1;
                scratch[i] = tour[source];
            }
            ant_group_sync<WarpTours>();
            for (Node i = first + lane; i <= last; i += group_width) {
                tour[i] = scratch[i];
                position[scratch[i]] = i;
            }
            ant_group_sync<WarpTours>();
        }
        if (lane == 0) {
            if (successor(parent, parent_position, n, state.current) != state.selected &&
                predecessor(parent, parent_position, n, state.current) != state.selected) {
                ++state.construction.mne;
                append_if_absent(pending, 0, state.current, state.pending_size, queued, pending_capacity);
                append_if_absent(pending, 0, state.selected, state.pending_size, queued, pending_capacity);
                append_if_absent(pending, 0, state.old_previous, state.pending_size, queued, pending_capacity);
            }
            state.current = state.selected;
        }
        ant_group_sync<WarpTours>();
    }
    if (construction_tours) {
        for (Node i = lane; i < n; i += group_width) construction_tours[base + i] = tour[i];
    }
    if constexpr (!WarpTours) if (construction_new_edges) {
        const Node count = observed_new_edges(tour, parent_position, n);
        if (lane == 0) construction_new_edges[ant] = count;
    }
    if (lane == 0) output[ant].construction_cost = state.cost;
    ant_group_sync<WarpTours>();
    if constexpr (escape) {
        for (Node node = lane; node < n; node += group_width)
            choices.build_ls_row(local_ant, node, width, n, candidates, matrix);
        ant_group_sync<WarpTours>();
        candidates = choices.ls_rows;
        if (lane == 0) for (Node node = 0; node < n; ++node) {
            // 足迹按node ID继承；即使父tour未留下图外边，也检查失效后恢复的普通行。
            if (choices.parent_footprint[node]) {
                const Node before = state.pending_size;
                append_if_absent(pending, 0, node, state.pending_size, queued, pending_capacity);
                choices.stats->old_view_reactivations += state.pending_size != before;
            }
            if (choices.anchors[node]) {
                ++choices.stats->ls_anchor_nodes;
                const Node before = state.pending_size;
                append_if_absent(pending, 0, node, state.pending_size, queued, pending_capacity);
                choices.stats->anchor_reactivations += state.pending_size != before;
                for (Node j = 0; j < width; ++j)
                    choices.stats->ls_replaced_slots += choices.ls_replaced[static_cast<std::size_t>(node) * width + j];
            }
        }
        ant_group_sync<WarpTours>();
    }
    if constexpr (Profile) { if (lane == 0) tick2 = clock64(); }

    for (;;) {
        if (lane == 0) {
            state.stopped = state.pending_head == state.pending_size || state.ls.accepted_moves >= n;
            if (!state.stopped && state.ls.move_evaluations >= evaluation_limit) {
                state.ls.evaluation_limit_reached = true;
                state.stopped = true;
            }
            if (!state.stopped) {
                state.a = pending[CompactWorkspace ? state.pending_head % n : state.pending_head];
                ++state.pending_head;
                if (queued) queued[state.a] = 0;
                state.a_next = successor(tour, position, n, state.a);
                state.a_previous = predecessor(tour, position, n, state.a);
                ++state.ls.processed_nodes;
                state.current_edge[0] = distance(matrix, n, state.a, state.a_next);
                state.current_edge[1] = distance(matrix, n, state.a_previous, state.a);
                state.allowed[0] = state.allowed[1] = 0;
                // 图约束、宽候选行和诊断布局继续按原顺序确定前缀。
                if (!(WarpTours && GPFACO_PARALLEL_LS_PREFIX &&
                      std::is_same_v<Allowed, UnrestrictedEdges> && width <= 32)) {
                for (int kind = 0; kind < 2 && !state.stopped; ++kind) {
                    const double current_distance = state.current_edge[kind];
                    for (Node j = 0; j < width; ++j) {
                        const Node b = candidates[state.a * width + j];
                        ++state.ls.candidate_checks;
                        if (b >= n) break;  // 距离有序行的右侧padding，不访问哨兵坐标。
                        if (!(current_distance > (candidate_distances ? candidate_distances[state.a * width + j] :
                                                  distance(matrix, n, state.a, b)))) break;
                        if (state.ls.move_evaluations == evaluation_limit) {
                            state.ls.evaluation_limit_reached = true;
                            state.stopped = true;
                            break;
                        }
                        ++state.ls.move_evaluations;
                        ++state.allowed[kind];
                        const Node neighbor = kind == 0 ? successor(tour, position, n, b)
                                                        : predecessor(tour, position, n, b);
                        const auto view = ls_allowed(state.a, j);
                        if constexpr (escape) {
                            const auto proposal = prepare_escape_two_opt(state.a,
                                kind == 0 ? state.a_next : state.a_previous, b, neighbor, view);
                            choices.stats->ls_capacity_rejections += proposal.status == EscapeProposalStatus::Capacity;
                            state.ls.constraint_rejections += !proposal;
                        } else if (!two_opt_allowed(state.a, kind == 0 ? state.a_next : state.a_previous,
                                             b, neighbor, view)) ++state.ls.constraint_rejections;
                    }
                }
                }
            }
        }
        ant_group_sync<WarpTours>();
        if (state.stopped) break;
        if constexpr (WarpTours && GPFACO_PARALLEL_LS_PREFIX &&
                      std::is_same_v<Allowed, UnrestrictedEdges>) {
            if (width <= 32) {
                // 只并行求首个不合格位置；按kind顺序记账，保留预算正好耗尽的行为。
                for (Node kind = 0; kind < 2; ++kind) {
                    bool bad = false;
                    if (lane < width) {
                        const Node b = candidates[state.a * width + lane];
                        bad = b >= n;
                        if (!bad) bad = !(state.current_edge[kind] > (candidate_distances ?
                            candidate_distances[state.a * width + lane] : distance(matrix, n, state.a, b)));
                    }
                    const unsigned failures = __ballot_sync(0xffffffffu, bad);
                    const Node prefix = failures ? __ffs(failures) - 1 : width;
                    if (lane == 0) {
                        const auto remaining = evaluation_limit - state.ls.move_evaluations;
                        const Node used = remaining < prefix ? static_cast<Node>(remaining) : prefix;
                        state.allowed[kind] = used;
                        state.ls.move_evaluations += used;
                        state.ls.candidate_checks += used + (used < width);
                        if (used < prefix) {
                            state.ls.evaluation_limit_reached = true;
                            state.stopped = true;
                        }
                    }
                    ant_group_sync<WarpTours>();
                    if (state.stopped) break;
                }
                if (state.stopped) break;
            }
        }
        double local_gain = -1;
        Node local_index = UINT32_MAX;
        for (Node index = lane; index < width * 2; index += group_width) {
            const Node kind = index / width, j = index % width;
            if (j >= state.allowed[kind]) continue;
            const Node b = candidates[state.a * width + j];
            const Node neighbor = kind == 0 ? successor(tour, position, n, b)
                                            : predecessor(tour, position, n, b);
            const double current_distance = state.current_edge[kind];
            const double other = kind == 0 ? distance(matrix, n, b, neighbor)
                                           : distance(matrix, n, neighbor, b);
            const Node closing = kind == 0 ? state.a_next : state.a_previous;
            // 恒等 2-opt 的真增益为零；不引入 epsilon，也不改变其他移动的顺序。
            const double gain = (neighbor == state.a || b == closing) ? 0.0 :
                two_opt_allowed(state.a, closing, b, neighbor, ls_allowed(state.a, j))
                ? current_distance + other - (candidate_distances ? candidate_distances[state.a * width + j] :
                                              distance(matrix, n, state.a, b))
                  -distance(matrix, n, closing, neighbor)
                : -CUDART_INF;
            if constexpr (WarpTours) {
                if (gain > local_gain) { local_gain = gain; local_index = index; }
            } else gains[index] = gain;
        }
        if constexpr (WarpTours) {
            // 只比较已经按原表达式得到的gain；同分保留原kind/j顺序，不重新结合浮点运算。
            for (Node offset = 16; offset; offset /= 2) {
                const double other_gain = __shfl_down_sync(0xffffffffu, local_gain, offset);
                const Node other_index = __shfl_down_sync(0xffffffffu, local_index, offset);
                if (lane + offset < 32 && (other_gain > local_gain ||
                    (other_gain == local_gain && other_index < local_index))) {
                    local_gain = other_gain; local_index = other_index;
                }
            }
        }
        ant_group_sync<WarpTours>();
        if (lane == 0) {
            state.best_gain = -1;
            if constexpr (WarpTours) {
                state.best_gain = local_gain;
                if (local_index != UINT32_MAX) {
                    state.best_slot = local_index % width; state.best_kind = local_index / width;
                }
            } else for (Node kind = 0; kind < 2; ++kind) for (Node j = 0; j < state.allowed[kind]; ++j) {
                const double gain = gains[kind * width + j];
                if (gain > state.best_gain) {
                    state.best_gain = gain;
                    state.best_slot = j; state.best_kind = kind;
                }
            }
            if (state.best_gain > 0) {
                const Node kind = state.best_kind, b_node = candidates[state.a * width + state.best_slot];
                const Node neighbor_node = kind == 0 ? successor(tour, position, n, b_node)
                                                     : predecessor(tour, position, n, b_node);
                state.move[0] = kind == 0 ? state.a_next : state.a;
                state.move[1] = kind == 0 ? neighbor_node : b_node;
                state.move[2] = kind == 0 ? state.a : state.a_previous;
                state.move[3] = kind == 0 ? b_node : neighbor_node;
                if constexpr (escape) {
                    const Node b = candidates[state.a * width + state.best_slot];
                    const Node neighbor = state.best_kind == 0 ? successor(tour, position, n, b)
                                                               : predecessor(tour, position, n, b);
                    const auto proposal = prepare_escape_two_opt(state.a,
                        state.best_kind == 0 ? state.a_next : state.a_previous, b, neighbor,
                        ls_allowed(state.a, state.best_slot));
                    if (!commit_escape_edges(*allowed.cache, proposal)) asm("trap;");
                    choices.stats->new_edges += proposal.count;
                    choices.record_move(state.construction.nonidentity_relocations + state.ls.accepted_moves, n,
                        {state.best_kind + 1, state.a, b, state.best_slot, allowed.cache->size,
                         choices.ls_replaced[static_cast<std::size_t>(state.a) * width + state.best_slot] != 0});
                }
                const Node a = position[state.move[0]], b = position[state.move[1]];
                state.first = a < b ? a : b;
                state.last = a < b ? b : a;
                ++state.ls.accepted_moves;
                state.accumulated_gain -= state.best_gain;
            }
        }
        ant_group_sync<WarpTours>();
        if (state.best_gain > 0) {
            const Node first = state.first, last = state.last, length = last - first;
            const bool straight = length <= n - length || first == 0;
            const Node count = straight ? length : n - length, start = straight ? first : last;
            // 反转由互不相交的交换对组成，原地交换无需全路线scratch往返。
            // first==0仍遵循原生数组布局；补片段允许跨越数组尾部。
            for (Node offset = lane; offset < count / 2; offset += group_width) {
                const Node left = (start + offset) % n, right = (start + count - 1 - offset) % n;
                const TourNode a = tour[left], b = tour[right];
                tour[left] = b; tour[right] = a;
                position[b] = left; position[a] = right;
            }
            ant_group_sync<WarpTours>();
            if (lane == 0) {
                for (Node node : state.move) {
                    const Node before = state.pending_size;
                    append_if_absent(pending, state.pending_head, node, state.pending_size, queued, pending_capacity);
                    state.ls.reactivations += state.pending_size != before;
                }
            }
        }
        ant_group_sync<WarpTours>();
    }
    if constexpr (WarpTours) {
        for (Node i = lane; i < n; i += group_width) {
            all_tours[base + i] = tour[i];
            all_positions[base + i] = position[i];
        }
        ant_group_sync<WarpTours>();
    }
    if (lane == 0) {
        if constexpr (Profile) tick3 = clock64();
        output[ant].final_cost = state.cost + state.accumulated_gain;
        output[ant].construction = state.construction;
        output[ant].local_search = state.ls;
        output[ant].checklist_size = state.pending_size;
        if constexpr (Profile) profile_cycles[ant] = {
            tick1 - tick0, tick2 - tick1, tick3 - tick2, clock64() - tick3};
    }
    if constexpr (WarpTours && GPFACO_SHARED_ANT_FLAGS) {
        for (Node i = lane; i < n; i += group_width) {
            all_visited[base + i] = visited[i];
            if (all_queued) all_queued[base + i] = queued[i];
        }
    }
}

}  // namespace gp_faco::cuda_detail
