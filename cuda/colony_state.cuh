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

struct Colony {
    double global_cost, epoch_cost, parent_cost, minimum, maximum, default_trail, source_uniform;
    Node iteration_best;
    bool source_is_epoch;
};

__device__ inline void bounds(Colony& state, Node width, double retention, double p_best) {
    const double p = pow(p_best, 1.0 / width);
    state.maximum = 1.0 / (state.epoch_cost * (1.0 - retention));
    state.minimum = fmin(state.maximum, state.maximum * (1.0 - p) / ((width - 1.0) * p));
}

static __global__ void reset_colony(Node n, const Node* all_initial, const double* costs, Node* all_parent,
    Node* all_epoch, Node* all_global, Colony* all_state, Node width, double retention, double p_best) {
    const auto colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const auto* initial = all_initial + base;
    auto* parent = all_parent + base; auto* epoch = all_epoch + base;
    auto* global = all_global + base; auto* state = all_state + colony;
    const double cost = costs[colony];
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
    double* heuristic, double* trails, double* products) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) return;
    const Node colony = i / (n * width);
    if (primary[i] >= n) {
        heuristic[i] = products[i] = 0;
        trails[i] = state[colony].maximum;
        return;
    }
    const double d = distance(i / width, primary[i] + colony * n);
    heuristic[i] = d > 0 ? 1.0 / pow(d, beta) : 1.0;
    trails[i] = state[colony].maximum;
    products[i] = trails[i] * heuristic[i];
}

static __global__ void reduce_and_select(Node n, Node ants, const Node* all_tours,
    const FacoDiagnosticInfo* all_info, Node* all_parent, Node* all_parent_position, Node* all_epoch, Node* all_global,
    Colony* all_state, Node width, double retention, double p_best, double epoch_probability,
    const std::uint64_t* seeds, Node batch) {
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
    }
}

static __global__ void update_pheromone(Node n, Node width, Node colonies, const Node* primary,
    const Node* all_parent, const Node* all_position, double retention, Colony* all_state,
    const double* heuristic, double* trails, double* products) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= colonies * n * width) return;
    const Node colony = i / (n * width), local_i = i % (n * width);
    const auto base = static_cast<std::size_t>(colony) * n;
    const auto* parent = all_parent + base; const auto* position = all_position + base;
    auto* state = all_state + colony;
    const Node from = local_i / width, to = primary[i], pos = position[from];
    double value = fmax(state->minimum, trails[i] * retention);
    // 每个有向条目独占写入；对称强化匹配tour的前驱或后继，避免原子竞争。
    if (parent[(pos + 1) % n] == to || parent[(pos + n - 1) % n] == to) {
        value = fmin(state->maximum, value + 1.0 / state->parent_cost);
    }
    trails[i] = value;
    products[i] = value * heuristic[i];
    if (local_i == 0) state->default_trail = fmax(state->minimum, state->default_trail * retention);
}

}  // namespace gp_faco::cuda_detail
