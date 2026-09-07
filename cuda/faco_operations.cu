// FACO Route/LS语义依据RSkinderowicz（Copyright (c) 2024，MIT）。
// 完整许可见 provenance/licenses/Adaptive-Tuning-MIT.txt。
// 本阶段只实现显式选点的操作诊断，不作为完整搜索/性能基准。
#include "gp_faco/faco_cuda_diagnostic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gp_faco {
namespace {

void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> class Buffer {
public:
    explicit Buffer(std::size_t size) : size_(size) {
        checked(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }
    ~Buffer() { cudaFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() const { return data_; }
    void upload(const std::vector<T>& values) {
        if (values.size() != size_) throw std::invalid_argument("诊断上传形状不符");
        checked(cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(size_);
        checked(cudaMemcpy(values.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
private:
    std::size_t size_;
    T* data_ = nullptr;
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

__device__ double distance(const double* matrix, Node n, Node a, Node b) {
    return matrix[static_cast<std::size_t>(a) * n + b];
}

__device__ Node successor(const Node* tour, const Node* position, Node n, Node node) {
    return tour[(position[node] + 1) % n];
}

__device__ Node predecessor(const Node* tour, const Node* position, Node n, Node node) {
    return tour[(position[node] + n - 1) % n];
}

__device__ void append_if_absent(Node* pending, Node from, Node node, Node& length) {
    for (Node i = from; i < length; ++i) if (pending[i] == node) return;
    pending[length++] = node;
}

__global__ void construct_and_search(
    const double* matrix, const Node* candidates, Node n, Node width,
    const Node* parent_tours, const Node* visits, const Node* targets,
    std::uint64_t evaluation_limit, Node* all_tours, Node* all_positions,
    Node* all_parent_positions, Node* all_scratch, Node* all_pending,
    double* all_gains, Node* construction_tours, FacoDiagnosticInfo* output) {
    const auto ant = blockIdx.x;
    const auto base = static_cast<std::size_t>(ant) * n;
    const Node* parent = parent_tours + base;
    const Node* visit = visits + base;
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
    }
    if (threadIdx.x == 0) {
        state = {};
        state.current = visit[0];
        for (Node i = 0; i < n; ++i) {
            state.cost += distance(matrix, n, parent[(i + n - 1) % n], parent[i]);
        }
    }
    __syncthreads();

    // 一蚂蚁一block，移动前捕获端点；scratch隔离并行读取和覆盖。
    while (state.construction.mne < targets[ant] && state.construction.steps + 1 < n) {
        // 所有warp先读完循环条件，线程0才能增加该条件所读取的steps。
        __syncthreads();
        if (threadIdx.x == 0) {
            state.selected = visit[++state.construction.steps];
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
        __syncthreads();
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
    for (Node i = threadIdx.x; i < n; i += blockDim.x) construction_tours[base + i] = tour[i];
    if (threadIdx.x == 0) output[ant].construction_cost = state.cost;
    __syncthreads();

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
            gains[index] = current_distance + other - distance(matrix, n, state.a, b)
                           -distance(matrix, n, closing, neighbor);
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
        output[ant].final_cost = state.cost + state.accumulated_gain;
        output[ant].construction = state.construction;
        output[ant].local_search = state.ls;
        output[ant].checklist_size = state.pending_size;
    }
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void permutation(const std::vector<Node>& values, Node n) {
    require(values.size() == n, "诊断tour/选点序列的维数不符");
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    for (Node i = 0; i < n; ++i) require(sorted[i] == i, "诊断tour/选点序列不是完整排列");
}

}  // namespace

std::vector<FacoDiagnosticResult> cuda_faco_diagnostic(
    const std::vector<double>& distances, const CandidateRows& ls_candidates,
    const std::vector<FacoDiagnosticTask>& tasks, std::uint64_t evaluation_limit) {
    const auto n = static_cast<Node>(ls_candidates.size());
    // 限定显式矩阵诊断的规模，避免它意外成为10K Engine的表示。
    require(n >= 3 && n <= 1024 && !tasks.empty() && tasks.size() <= 256,
            "诊断仅支持3..1024节点、1..256蚂蚁");
    require(distances.size() == static_cast<std::size_t>(n) * n, "诊断距离矩阵形状不符");
    const Node width = ls_candidates[0].size();
    require(width > 0 && width < n, "诊断候选宽度无效");
    for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) {
        const double d = distances[a * n + b];
        require(std::isfinite(d) && d >= 0 && d == distances[b * n + a], "距离必须有限、非负且对称");
    }
    const auto distance_fn = [&](Node a, Node b) { return distances[a * n + b]; };
    DistanceOrderedCandidates sorted(ls_candidates, distance_fn);
    std::vector<Node> candidates, tours, visits, targets;
    for (Node a = 0; a < n; ++a) {
        require(sorted[a].size() == width, "诊断候选行必须等宽");
        candidates.insert(candidates.end(), sorted[a].begin(), sorted[a].end());
    }
    for (const auto& task : tasks) {
        permutation(task.tour, n);
        permutation(task.visit_order, n);
        require(task.mne_target > 0, "诊断MNE必须为正");
        tours.insert(tours.end(), task.tour.begin(), task.tour.end());
        visits.insert(visits.end(), task.visit_order.begin(), task.visit_order.end());
        targets.push_back(task.mne_target);
    }
    const std::size_t cells = n * tasks.size();
    Buffer<double> d_distances(distances.size()), d_gains(tasks.size() * width * 2);
    Buffer<Node> d_candidates(candidates.size()), d_parents(cells), d_visits(cells),
        d_targets(tasks.size()), d_tours(cells), d_positions(cells), d_parent_positions(cells),
        d_scratch(cells), d_pending(cells * 5), d_constructed(cells);
    Buffer<FacoDiagnosticInfo> d_info(tasks.size());
    d_distances.upload(distances); d_candidates.upload(candidates);
    d_parents.upload(tours); d_visits.upload(visits); d_targets.upload(targets);
    construct_and_search<<<tasks.size(), 128>>>(
        d_distances.data(), d_candidates.data(), n, width, d_parents.data(), d_visits.data(),
        d_targets.data(), evaluation_limit, d_tours.data(), d_positions.data(),
        d_parent_positions.data(), d_scratch.data(), d_pending.data(), d_gains.data(),
        d_constructed.data(), d_info.data());
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    const auto final_tours = d_tours.download(), final_positions = d_positions.download(),
        constructed = d_constructed.download(), pending = d_pending.download();
    const auto info = d_info.download();
    std::vector<FacoDiagnosticResult> result;
    for (std::size_t ant = 0; ant < tasks.size(); ++ant) {
        const auto base = ant * n;
        require(info[ant].checklist_size <= 5 * n, "设备checklist长度无效");
        result.push_back({
            {constructed.begin() + base, constructed.begin() + base + n},
            {final_tours.begin() + base, final_tours.begin() + base + n},
            {final_positions.begin() + base, final_positions.begin() + base + n},
            {pending.begin() + base * 5, pending.begin() + base * 5 + info[ant].checklist_size},
            info[ant]});
    }
    return result;
}

}  // namespace gp_faco
