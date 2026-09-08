// 设备距离与独立Philox随机流。cuRAND头文件遵循CUDA Toolkit自带许可。
#pragma once
#include "gp_faco/faco_cpu.hpp"
#include "gp_faco/edge_constraints.hpp"
#include "gp_faco/control_ops.hpp"
#include <curand_kernel.h>
#include <math_constants.h>
#include <type_traits>

namespace gp_faco::cuda_detail {

struct CoordinateDistance {
    const double* xy;
    __device__ CoordinateDistance for_ant(Node) const { return *this; }
    __host__ __device__ double operator()(Node a, Node b) const {
        const double x = xy[a * 2] - xy[b * 2], y = xy[a * 2 + 1] - xy[b * 2 + 1];
        return sqrt(x * x + y * y);
    }
};

// batch/ant组成Philox的子序列；step占据独立的4-word counter槽。
// 保留ant=UINT32_MAX给colony级强化决策，不与任何蚂蚁冲突。
__device__ inline curandStatePhilox4_32_10_t random_state(
    std::uint64_t seed, Node batch, Node ant, Node step) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (static_cast<std::uint64_t>(batch) << 32) | ant,
                static_cast<std::uint64_t>(step) * 4, &state);
    return state;
}

__device__ inline double uniform53(curandStatePhilox4_32_10_t& state) {
    const auto hi = static_cast<std::uint64_t>(curand(&state));
    const auto lo = static_cast<std::uint64_t>(curand(&state));
    return static_cast<double>(((hi << 32) | lo) >> 11) * 0x1.0p-53;
}

template<class Allowed>
__device__ bool available_node(const std::uint8_t* visited, const Node* tour,
    const Node* position, Node n, Node current, Node node, const Allowed& allowed) {
    if (visited[node]) return false;
    // 无约束的主实验路径保留原有visited筛选，不额外执行图/重连判断。
    if constexpr (std::is_same_v<Allowed, UnrestrictedEdges>) return true;
    else return relocation_allowed(tour, position, n, current, node, allowed);
}

struct StochasticChoices {
    static constexpr bool shared_parent = true;
    const Node* primary;
    const Node* backup;
    const double* products;
    Node primary_width, backup_width, batch;
    std::uint64_t seed;
    Node* selected_trace;
    double* uniform_trace;
    const Node* start_nodes = nullptr;
    Node start_count = 0;
    __device__ StochasticChoices for_ant(Node, Node) const { return *this; }
    __device__ Node ant_index(Node ant) const { return ant; }
    __device__ std::size_t parent_offset(Node, Node) const { return 0; }
    __device__ std::size_t candidate_offset(Node, Node, Node) const { return 0; }

    __device__ Node start(Node ant, Node n) const {
        auto random = random_state(seed, batch, ant, 0xffffffffu);
        // 拒绝采样避免模偏差。起点占用远离构造step的保留counter区。
        const Node count = start_nodes ? start_count : n;
        const Node threshold = -count % count;
        Node value;
        do { value = curand(&random); } while (value < threshold);
        const Node node = start_nodes ? start_nodes[value % count] : value % count;
        if (selected_trace) {
            selected_trace[static_cast<std::size_t>(ant) * n] = node;
            uniform_trace[static_cast<std::size_t>(ant) * n] = 0;
        }
        return node;
    }

    template<class Distance, class Allowed>
    __device__ Node next(Node ant, Node current, const std::uint8_t* visited,
                         Node step, Node n, Distance distance, const Node* tour,
                         const Node* position, const Allowed& allowed) const {
        auto random = random_state(seed, batch, ant, step);
        const double uniform = uniform53(random);
        double total = 0;
        Node available = 0, chosen = n;
        const auto offset = static_cast<std::size_t>(current) * primary_width;
        for (Node j = 0; j < primary_width; ++j) {
            const Node node = primary[offset + j];
            if (available_node(visited, tour, position, n, current, node, allowed)) {
                total += products[offset + j]; ++available; chosen = node;
            }
        }
        if (available) {
            if (total > 0) {
                double prefix = 0;
                const double threshold = uniform * total;
                for (Node j = 0; j < primary_width; ++j) {
                    const Node node = primary[offset + j];
                    if (available_node(visited, tour, position, n, current, node, allowed)) {
                        prefix += products[offset + j];
                        if (threshold < prefix) { chosen = node; break; }
                    }
                }
            } else {
                Node target = static_cast<Node>(uniform * available);
                if (target >= available) target = available - 1;
                for (Node j = 0; j < primary_width; ++j) {
                    const Node node = primary[offset + j];
                    if (available_node(visited, tour, position, n, current, node, allowed)
                        && target-- == 0) { chosen = node; break; }
                }
            }
        } else {
            for (Node j = 0; j < backup_width; ++j) {
                const Node node = backup[static_cast<std::size_t>(current) * backup_width + j];
                if (available_node(visited, tour, position, n, current, node, allowed)) {
                    chosen = node; break;
                }
            }
            if (chosen == n) {
                double minimum = CUDART_INF;
                for (Node node = 0; node < n; ++node) {
                    if (!available_node(visited, tour, position, n, current, node, allowed)) continue;
                    const double d = distance(current, node);
                    if (d < minimum) { minimum = d; chosen = node; }
                }
            }
        }
        if (selected_trace) {
            selected_trace[static_cast<std::size_t>(ant) * n + step] = chosen;
            uniform_trace[static_cast<std::size_t>(ant) * n + step] = uniform;
        }
        return chosen;
    }
};

struct BatchCoordinateDistance {
    const double* xy;
    Node n, ants;
    __device__ CoordinateDistance for_ant(Node ant) const {
        return {xy + static_cast<std::size_t>(ant / ants) * n * 2};
    }
};

struct BatchStochasticChoices {
    const Node* primary;
    const Node* backup;
    const double* products;
    const std::uint64_t* seeds;
    Node primary_width, backup_width, batch, ants;
    __device__ StochasticChoices for_ant(Node ant, Node n) const {
        const auto colony = ant / ants;
        const auto base = static_cast<std::size_t>(colony) * n;
        return {primary + base * primary_width,
                backup_width ? backup + base * backup_width : nullptr,
                products + base * primary_width, primary_width, backup_width, batch,
                seeds[colony], nullptr, nullptr};
    }
    __device__ Node ant_index(Node ant) const { return ant % ants; }
    __device__ std::size_t parent_offset(Node ant, Node n) const {
        return static_cast<std::size_t>(ant / ants) * n;
    }
    __device__ std::size_t candidate_offset(Node ant, Node n, Node width) const {
        return parent_offset(ant, n) * width;
    }
};

struct BatchRegionChoices : BatchStochasticChoices {
    const StartRegions* regions;
    const std::int32_t* actions;
    __device__ StochasticChoices for_ant(Node ant, Node n) const {
        auto choices = BatchStochasticChoices::for_ant(ant, n);
        const Node colony = ant / ants, action = actions[colony];
        choices.start_nodes = regions[colony].nodes[action / 16][(action / 4) % 4];
        choices.start_count = regions[colony].count;
        return choices;
    }
};

}  // namespace gp_faco::cuda_detail
