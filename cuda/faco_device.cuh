// 共同FACO构造/LS设备实现；诊断与真实选点共享此内核。
// 原始Route/LS语义 Copyright (c) 2024 RSkinderowicz，MIT许可见provenance。
#pragma once

#include "edge_observation.cuh"
#include "gp_faco/faco_cuda_diagnostic.hpp"
#include "gp_faco/edge_constraints.hpp"
#include "gp_faco/profiling.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace gp_faco::cuda_detail {

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
    Node first, last, a, a_next, a_previous;
    Node move[4], allowed[2], pending_size, pending_head;
    bool nonidentity, stopped;
    double cost, accumulated_gain, best_gain;
    ConstructionStats construction;
    LocalSearchStats ls;
};

template<class Distance>
__device__ double distance(const Distance& matrix, Node, Node a, Node b) {
    return matrix(a, b);
}

__device__ inline Node successor(const Node* tour, const Node* position, Node n, Node node) {
    return tour[(position[node] + 1) % n];
}

__device__ inline Node predecessor(const Node* tour, const Node* position, Node n, Node node) {
    return tour[(position[node] + n - 1) % n];
}

__device__ inline void append_if_absent(Node* pending, Node from, Node node, Node& length) {
    for (Node i = from; i < length; ++i) if (pending[i] == node) return;
    pending[length++] = node;
}

template<class Distance, class Choices, class Allowed = UnrestrictedEdges, bool Profile = false>
__global__ void construct_and_search(
    Distance distances, const Node* all_candidates, Node n, Node width,
    const Node* parent_tours, Choices choice_views, const Node* targets,
    std::uint64_t evaluation_limit, Node* all_tours, Node* all_positions,
    Node* all_parent_positions, Node* all_scratch, Node* all_pending,
    double* all_gains, Node* construction_tours, FacoDiagnosticInfo* output,
    std::uint8_t* all_visited, Allowed all_allowed = {}, AntPhaseCycles* profile_cycles = nullptr,
    Node* construction_new_edges = nullptr) {
    std::uint64_t tick0 = 0, tick1 = 0, tick2 = 0, tick3 = 0;
    if constexpr (Profile) { if (threadIdx.x == 0) tick0 = clock64(); }
    const auto ant = blockIdx.x;
    const auto base = static_cast<std::size_t>(ant) * n;
    const auto matrix = distances.for_ant(ant);
    const auto choices = choice_views.for_ant(ant, n);
    const auto local_ant = choice_views.ant_index(ant);
    const auto allowed = all_allowed.for_ant(ant, n);
    const Node* candidates = all_candidates + choice_views.candidate_offset(ant, n, width);
    const Node* parent = parent_tours + choice_views.parent_offset(ant, n);
    std::uint8_t* visited = all_visited + base;
    Node* tour = all_tours + base;
    Node* position = all_positions + base;
    Node* parent_position = all_parent_positions + base;
    Node* scratch = all_scratch + base;
    Node* pending = all_pending + base * 5;
    double* gains = all_gains + static_cast<std::size_t>(ant) * width * 2;
    __shared__ State state;
    for (Node i = threadIdx.x; i < n; i += blockDim.x) {
        tour[i] = parent[i];
        position[parent[i]] = i;
        parent_position[parent[i]] = i;
        visited[i] = 0;
    }
    if (threadIdx.x == 0) {
        state = {};
        state.current = choices.start(local_ant, n);
        output[ant].start_node = state.current;
        for (Node i = 0; i < n; ++i) {
            state.cost += distance(matrix, n, parent[(i + n - 1) % n], parent[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) visited[state.current] = 1;
    __syncthreads();
    if constexpr (Profile) { if (threadIdx.x == 0) tick1 = clock64(); }

    // 一蚂蚁一block，移动前捕获端点；scratch隔离并行读取和覆盖。
    while (state.construction.mne < targets[ant] && state.construction.steps + 1 < n) {
        // 所有warp先读完循环条件，线程0才能增加该条件所读取的steps。
        __syncthreads();
        if (threadIdx.x == 0) {
            state.selected = choices.next(local_ant, state.current, visited, state.construction.steps + 1,
                                          n, matrix, tour, position, allowed);
            state.stopped = state.selected >= n;
            if (state.stopped) state.construction.legal_exhausted = true;
            // 同一线程完成选择与端点捕获，随后只需一次block发布边界。
            else {
                ++state.construction.steps;
                visited[state.selected] = 1;
                state.old_previous = predecessor(tour, position, n, state.selected);
                state.node_position = position[state.selected];
                state.target_position = position[state.current];
                const Node after = successor(tour, position, n, state.selected);
                const Node target_after = successor(tour, position, n, state.current);
                state.nonidentity = target_after != state.selected;
                if (state.nonidentity) {
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
        __syncthreads();
        // 全声明枚举均无合法移动时共同退出；没有visited[n]或隐式图外回退。
        if (state.stopped) break;
        if (state.nonidentity) {
            for (Node i = threadIdx.x; i < n; i += blockDim.x) {
                Node source = i;
                const Node a = state.target_position, b = state.node_position;
                if (a < b && i > a && i <= b) source = i == a + 1 ? b : i - 1;
                if (a > b && i >= b && i <= a) source = i == a ? b : i + 1;
                scratch[i] = tour[source];
            }
            __syncthreads();
            for (Node i = threadIdx.x; i < n; i += blockDim.x) {
                tour[i] = scratch[i];
                position[scratch[i]] = i;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (successor(parent, parent_position, n, state.current) != state.selected &&
                predecessor(parent, parent_position, n, state.current) != state.selected) {
                ++state.construction.mne;
                append_if_absent(pending, 0, state.current, state.pending_size);
                append_if_absent(pending, 0, state.selected, state.pending_size);
                append_if_absent(pending, 0, state.old_previous, state.pending_size);
            }
            state.current = state.selected;
        }
        __syncthreads();
    }
    if (construction_tours) {
        for (Node i = threadIdx.x; i < n; i += blockDim.x) construction_tours[base + i] = tour[i];
    }
    if (construction_new_edges) {
        const Node count = observed_new_edges(tour, parent_position, n);
        if (threadIdx.x == 0) construction_new_edges[ant] = count;
    }
    if (threadIdx.x == 0) output[ant].construction_cost = state.cost;
    __syncthreads();
    if constexpr (Profile) { if (threadIdx.x == 0) tick2 = clock64(); }

    for (;;) {
        if (threadIdx.x == 0) {
            state.stopped = state.pending_head == state.pending_size || state.ls.accepted_moves >= n;
            if (!state.stopped && state.ls.move_evaluations >= evaluation_limit) {
                state.ls.evaluation_limit_reached = true;
                state.stopped = true;
            }
            if (!state.stopped) {
                state.a = pending[state.pending_head++];
                state.a_next = successor(tour, position, n, state.a);
                state.a_previous = predecessor(tour, position, n, state.a);
                ++state.ls.processed_nodes;
                state.allowed[0] = state.allowed[1] = 0;
                // 先按原生顺序确定可检查前缀与精确计数，再并行计算gain。
                for (int kind = 0; kind < 2 && !state.stopped; ++kind) {
                    const double current_distance = kind == 0
                        ? distance(matrix, n, state.a, state.a_next)
                        : distance(matrix, n, state.a_previous, state.a);
                    for (Node j = 0; j < width; ++j) {
                        const Node b = candidates[state.a * width + j];
                        ++state.ls.candidate_checks;
                        if (!(current_distance > distance(matrix, n, state.a, b))) break;
                        if (state.ls.move_evaluations == evaluation_limit) {
                            state.ls.evaluation_limit_reached = true;
                            state.stopped = true;
                            break;
                        }
                        ++state.ls.move_evaluations;
                        ++state.allowed[kind];
                        const Node neighbor = kind == 0 ? successor(tour, position, n, b)
                                                        : predecessor(tour, position, n, b);
                        if (!two_opt_allowed(state.a, kind == 0 ? state.a_next : state.a_previous,
                                             b, neighbor, allowed)) ++state.ls.constraint_rejections;
                    }
                }
            }
        }
        __syncthreads();
        if (state.stopped) break;
        for (Node index = threadIdx.x; index < width * 2; index += blockDim.x) {
            const Node kind = index / width, j = index % width;
            if (j >= state.allowed[kind]) continue;
            const Node b = candidates[state.a * width + j];
            const Node neighbor = kind == 0 ? successor(tour, position, n, b)
                                            : predecessor(tour, position, n, b);
            const double current_distance = kind == 0
                ? distance(matrix, n, state.a, state.a_next)
                : distance(matrix, n, state.a_previous, state.a);
            const double other = kind == 0 ? distance(matrix, n, b, neighbor)
                                           : distance(matrix, n, neighbor, b);
            const Node closing = kind == 0 ? state.a_next : state.a_previous;
            gains[index] = two_opt_allowed(state.a, closing, b, neighbor, allowed)
                ? current_distance + other - distance(matrix, n, state.a, b)
                  -distance(matrix, n, closing, neighbor)
                : -CUDART_INF;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            state.best_gain = -1;
            for (Node kind = 0; kind < 2; ++kind) for (Node j = 0; j < state.allowed[kind]; ++j) {
                const double gain = gains[kind * width + j];
                if (gain > state.best_gain) {
                    state.best_gain = gain;
                    const Node b = candidates[state.a * width + j];
                    const Node neighbor = kind == 0 ? successor(tour, position, n, b)
                                                    : predecessor(tour, position, n, b);
                    state.move[0] = kind == 0 ? state.a_next : state.a;
                    state.move[1] = kind == 0 ? neighbor : b;
                    state.move[2] = kind == 0 ? state.a : state.a_previous;
                    state.move[3] = kind == 0 ? b : neighbor;
                }
            }
            if (state.best_gain > 0) {
                const Node a = position[state.move[0]], b = position[state.move[1]];
                state.first = a < b ? a : b;
                state.last = a < b ? b : a;
                ++state.ls.accepted_moves;
                state.accumulated_gain -= state.best_gain;
            }
        }
        __syncthreads();
        if (state.best_gain > 0) {
            for (Node i = threadIdx.x; i < n; i += blockDim.x) {
                const Node first = state.first, last = state.last, length = last - first;
                Node source = i;
                // 原生first==0的补片段交换最终等于反转指定片段；保持其数组结果。
                if (length <= n - length || first == 0) {
                    if (i >= first && i < last) source = first + last - 1 - i;
                } else {
                    const Node offset = (i + n - last) % n, count = n - length;
                    if (offset < count) source = (last + count - 1 - offset) % n;
                }
                scratch[i] = tour[source];
            }
            __syncthreads();
            for (Node i = threadIdx.x; i < n; i += blockDim.x) {
                tour[i] = scratch[i];
                position[scratch[i]] = i;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                for (Node node : state.move) {
                    const Node before = state.pending_size;
                    append_if_absent(pending, state.pending_head, node, state.pending_size);
                    state.ls.reactivations += state.pending_size != before;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if constexpr (Profile) tick3 = clock64();
        output[ant].final_cost = state.cost + state.accumulated_gain;
        output[ant].construction = state.construction;
        output[ant].local_search = state.ls;
        output[ant].checklist_size = state.pending_size;
        if constexpr (Profile) profile_cycles[ant] = {
            tick1 - tick0, tick2 - tick1, tick3 - tick2, clock64() - tick3};
    }
}

}  // namespace gp_faco::cuda_detail
