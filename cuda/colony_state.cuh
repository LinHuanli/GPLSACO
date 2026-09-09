// 共同colony状态与信息素内核；单实例和多实例Engine复用。
// 原始FACO规则 Copyright (c) 2024 RSkinderowicz，MIT许可见provenance。
#pragma once
#include "faco_choices.cuh"
#include "gp_faco/faco_cuda_diagnostic.hpp"
#include <stdexcept>
#include <vector>

namespace gp_faco::cuda_detail {
inline void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class T> class DeviceArray {
public:
    ~DeviceArray() { if (pointer_) cudaFree(pointer_); }
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void allocate(std::size_t size) {
        require(!pointer_ && count_ == 0, "持久数组不得重复分配");
        count_ = size;
        if (size) checked(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes()));
    }
    T* data() const { return pointer_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
    void upload(const std::vector<T>& input) {
        require(input.size() == count_, "持久数组上传形状不符");
        if (count_) checked(cudaMemcpy(pointer_, input.data(), bytes(), cudaMemcpyHostToDevice));
    }
    void upload_prefix(const std::vector<T>& input) {
        require(input.size() <= count_, "活动种群超过持久数组容量");
        if (!input.empty()) checked(cudaMemcpy(pointer_, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download_prefix(std::size_t count) const {
        require(count <= count_, "活动种群超过持久数组容量");
        std::vector<T> result(count);
        if (count) checked(cudaMemcpy(result.data(), pointer_, count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
    std::vector<T> download() const {
        std::vector<T> result(count_);
        if (count_) checked(cudaMemcpy(result.data(), pointer_, bytes(), cudaMemcpyDeviceToHost));
        return result;
    }
    void zero() { if (count_) checked(cudaMemset(pointer_, 0, bytes())); }
private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

struct WorkTotals {
    std::uint64_t construction = 0, local_search = 0, constraints = 0;
    EscapeStats escape;
};

struct Colony {
    double global_cost, epoch_cost, parent_cost, minimum, maximum, default_trail, source_uniform;
    Node iteration_best;
    bool source_is_epoch;
    WorkTotals work;
};

__device__ inline void bounds(Colony& state, Node width, double retention, double p_best) {
    const double p = pow(p_best, 1.0 / width);
    state.maximum = 1.0 / (state.epoch_cost * (1.0 - retention));
    state.minimum = fmin(state.maximum, state.maximum * (1.0 - p) / ((width - 1.0) * p));
}

static __global__ void reset_colony(Node n, const Node* all_initial, const double* costs, Node* all_parent,
    Node* all_epoch, Node* all_global, Colony* all_state, Node width, double retention, double p_best,
    Node geometry_count = 0) {
    const auto colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const auto geometry = geometry_count ? colony % geometry_count : colony;
    const auto* initial = all_initial + static_cast<std::size_t>(geometry) * n;
    auto* parent = all_parent + base; auto* epoch = all_epoch + base;
    auto* global = all_global + base; auto* state = all_state + colony;
    const double cost = costs[geometry];
    for (Node i = threadIdx.x; i < n; i += blockDim.x) parent[i] = epoch[i] = global[i] = initial[i];
    if (threadIdx.x == 0) {
        *state = {};
        state->global_cost = state->epoch_cost = state->parent_cost = cost;
        bounds(*state, width, retention, p_best);
        state->default_trail = state->maximum;
    }
}

static __global__ void initialize_products(cuda_detail::CoordinateDistance distance, Node cells, Node n,
    Node width, const Node* primary, double beta, const Colony* state,
    double* heuristic, double* trails, double* products, Node geometry_count = 0) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) return;
    const Node colony = i / (n * width);
    const Node geometry_i = geometry_count ? i % (geometry_count * n * width) : i;
    const Node geometry_colony = geometry_count ? colony % geometry_count : colony;
    if (primary[geometry_i] >= n) {
        if (!geometry_count || colony < geometry_count) heuristic[geometry_i] = 0;
        products[i] = 0;
        trails[i] = state[colony].maximum;
        return;
    }
    const double d = distance(geometry_i / width, primary[geometry_i] + geometry_colony * n);
    const double value = d > 0 ? 1.0 / pow(d, beta) : 1.0;
    if (!geometry_count || colony < geometry_count) heuristic[geometry_i] = value;
    trails[i] = state[colony].maximum;
    products[i] = trails[i] * value;
}

static __global__ void cache_candidate_distances(const double* xy, Node cells, Node n,
    Node width, const Node* candidates, double* values) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) return;
    const Node colony = i / (n * width);
    values[i] = candidates[i] < n ? CoordinateDistance{xy}(i / width, candidates[i] + colony * n) : 0;
}

static __global__ void cache_pair_distances(const double* xy, Node n, Node colonies, double* values) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto cells = static_cast<std::size_t>(n) * n;
    if (i >= cells * colonies) return;
    const auto colony = i / cells, edge = i % cells;
    // 当前小规模一次性复用原数值距离；大规模继续保留按需距离路径。
    values[i] = CoordinateDistance{xy + colony * n * 2}(edge / n, edge % n);
}

static __global__ void cache_parent_costs(const double* xy, Node n, const Node* all_parent,
    double* costs, const double* cached = nullptr, Node geometry_count = 0) {
    const Node colony = blockIdx.x;
    const auto* parent = all_parent + static_cast<std::size_t>(colony) * n;
    const auto offset = static_cast<std::size_t>(geometry_count ? colony % geometry_count : colony);
    const CoordinateDistance distance{xy + offset * n * 2, cached ? cached + offset * n * n : nullptr, n};
    extern __shared__ double edges[];
    for (Node i = threadIdx.x; i < n; i += blockDim.x)
        edges[i] = distance(parent[(i + n - 1) % n], parent[i]);
    __syncthreads();
    if (threadIdx.x == 0) {
        // 每个 colony 的蚂蚁共享同一父路线。边长并行计算，仍按原来的 i 顺序
        // 累加 double，避免改变舍入、平局和后续轨迹；不使用增量近似 parent_cost。
        double cost = 0;
        for (Node i = 0; i < n; ++i) cost += edges[i];
        costs[colony] = cost;
    }
}

static __global__ void reduce_and_select(Node n, Node ants, const Node* all_tours,
    const FacoDiagnosticInfo* all_info, Node* all_parent, Node* all_parent_position, Node* all_epoch, Node* all_global,
    Colony* all_state, Node width, double retention, double p_best, double epoch_probability,
    const std::uint64_t* seeds, Node batch, EscapeFootprints footprints = {}) {
    const auto colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const Node* tours = all_tours + base * ants;
    const auto* info = all_info + colony * ants;
    auto* parent = all_parent + base; auto* parent_position = all_parent_position + base;
    auto* epoch = all_epoch + base; auto* global = all_global + base;
    auto* state = all_state + colony;
    const auto seed = seeds[colony];
    __shared__ bool improve_epoch, improve_global;
    if (threadIdx.x == 0) {
        // 完整批次计数在设备累计，生产入口只在结束时取回。
        for (Node ant = 0; ant < ants; ++ant) {
            state->work.construction += info[ant].construction.steps;
            state->work.local_search += info[ant].local_search.move_evaluations;
            state->work.constraints += info[ant].local_search.constraint_rejections;
            state->work.escape.construction_opportunities += info[ant].escape.construction_opportunities;
            state->work.escape.construction_gates += info[ant].escape.construction_gates;
            state->work.escape.construction_replaced_slots += info[ant].escape.construction_replaced_slots;
            state->work.escape.escape_relocations += info[ant].escape.escape_relocations;
            state->work.escape.ls_anchor_nodes += info[ant].escape.ls_anchor_nodes;
            state->work.escape.ls_replaced_slots += info[ant].escape.ls_replaced_slots;
            state->work.escape.old_view_reactivations += info[ant].escape.old_view_reactivations;
            state->work.escape.anchor_reactivations += info[ant].escape.anchor_reactivations;
            state->work.escape.new_edges += info[ant].escape.new_edges;
            state->work.escape.construction_capacity_rejections += info[ant].escape.construction_capacity_rejections;
            state->work.escape.ls_capacity_rejections += info[ant].escape.ls_capacity_rejections;
        }
        Node best = 0;
        for (Node ant = 1; ant < ants; ++ant) if (info[ant].final_cost < info[best].final_cost) best = ant;
        state->iteration_best = best;
        const double cost = info[best].final_cost;
        improve_epoch = cost < state->epoch_cost;
        improve_global = cost < state->global_cost;
        if (improve_epoch) state->epoch_cost = cost;
        if (improve_global) state->global_cost = cost;
        bounds(*state, width, retention, p_best);
        auto random = cuda_detail::random_state(seed, batch, 0xffffffffu, 0);
        state->source_uniform = cuda_detail::uniform53(random);
        state->source_is_epoch = state->source_uniform < epoch_probability;
        state->parent_cost = state->source_is_epoch ? state->epoch_cost : cost;
    }
    __syncthreads();
    const Node* best = tours + static_cast<std::size_t>(state->iteration_best) * n;
    for (Node i = threadIdx.x; i < n; i += blockDim.x) {
        if (improve_epoch) epoch[i] = best[i];
        if (improve_global) global[i] = best[i];
        const Node node = state->source_is_epoch ? epoch[i] : best[i];
        parent[i] = node;
        parent_position[node] = i;
        if (footprints.parent) {
            const auto value = footprints.ants[base * ants + static_cast<std::size_t>(state->iteration_best) * n + i];
            if (improve_epoch) footprints.epoch[base + i] = value;
            if (improve_global) footprints.global[base + i] = value;
            footprints.parent[base + i] = state->source_is_epoch ? footprints.epoch[base + i] : value;
        }
    }
}

static __global__ void update_pheromone(Node n, Node width, Node colonies, const Node* primary,
    const Node* all_parent, const Node* all_position, double retention, Colony* all_state,
    const double* heuristic, double* trails, double* products, Node geometry_count = 0) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= colonies * n * width) return;
    const Node colony = i / (n * width), local_i = i % (n * width);
    const auto base = static_cast<std::size_t>(colony) * n;
    const auto* parent = all_parent + base; const auto* position = all_position + base;
    auto* state = all_state + colony;
    const Node geometry_i = geometry_count ? i % (geometry_count * n * width) : i;
    const Node from = local_i / width, to = primary[geometry_i], pos = position[from];
    double value = fmax(state->minimum, trails[i] * retention);
    // 每个有向条目独占写入；对称强化匹配tour的前驱或后继，避免原子竞争。
    if (parent[(pos + 1) % n] == to || parent[(pos + n - 1) % n] == to) {
        value = fmin(state->maximum, value + 1.0 / state->parent_cost);
    }
    trails[i] = value;
    products[i] = value * heuristic[geometry_i];
    if (local_i == 0) state->default_trail = fmax(state->minimum, state->default_trail * retention);
}

}  // namespace gp_faco::cuda_detail
