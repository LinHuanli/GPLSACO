// 主底座控制的设备状态；准备、评分、事务和批后更新都在同一CUDA流中。
#pragma once

#include "gp_faco/control_ops.hpp"
#include "colony_state.cuh"

namespace gp_faco::cuda_detail {

struct DevicePheromoneView {
    const Node* primary;
    const double* trails;
    Node width;
    double default_value;
    __host__ __device__ double operator()(Node a, Node b) const {
        for (Node j = 0; j < width; ++j) if (primary[a * width + j] == b) return trails[a * width + j];
        return default_value;
    }
};

static __global__ void initialize_control(Node n, const Node* all_initial, const double* costs,
    Node* all_parent_positions, Node* all_archive, Node* all_archive_positions, ControllerState* controls,
    bool force_collisions = false) {
    const Node colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const auto* initial = all_initial + base;
    auto* tours = all_archive + base * archive_capacity;
    auto* positions = all_archive_positions + base * archive_capacity;
    for (Node i = threadIdx.x; i < n * archive_capacity; i += blockDim.x) {
        tours[i] = i < n ? initial[i] : 0;
        if (i < n) {
            positions[initial[i]] = i;
            all_parent_positions[base + initial[i]] = i;
        } else positions[i] = 0;
    }
    if (threadIdx.x == 0) {
        auto& state = controls[colony]; state = {};
        state.archive_size = 1;
        state.archive_cost[0] = state.tracked_global = state.tracked_epoch = costs[colony];
        state.archive_identity[0] = force_collisions ? TourFingerprint{} : fingerprint({initial, nullptr, n});
        state.active_identity = state.epoch_identity = state.archive_identity[0];
    }
}

static __global__ void fingerprint_ants(Node n, const Node* all_tours, TourFingerprint* result,
                                       bool force_collisions = false) {
    const auto* tour = all_tours + static_cast<std::size_t>(blockIdx.x) * n;
    TourFingerprint local;
    for (Node i = threadIdx.x; i < n; i += blockDim.x) {
        const auto edge = fingerprint_edge(tour[i], tour[(i + 1) % n]);
        local.first ^= edge.first; local.second ^= edge.second;
    }
    __shared__ TourFingerprint reduced[128];
    reduced[threadIdx.x] = local;
    __syncthreads();
    for (Node stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            reduced[threadIdx.x].first ^= reduced[threadIdx.x + stride].first;
            reduced[threadIdx.x].second ^= reduced[threadIdx.x + stride].second;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) result[blockIdx.x] = force_collisions ? TourFingerprint{} : reduced[0];
}

struct ArchivePool {
    Node n;
    const Node *archive, *archive_positions, *tours, *positions;
    const ControllerState* control;
    const FacoDiagnosticInfo* info;
    const TourFingerprint* identities;
    __device__ TourView view(Node entry) const {
        if (entry < archive_capacity) return {archive + static_cast<std::size_t>(entry) * n,
            archive_positions + static_cast<std::size_t>(entry) * n, n};
        const auto base = static_cast<std::size_t>(entry - archive_capacity) * n;
        return {tours + base, positions + base, n};
    }
    __device__ double cost(Node entry) const {
        return entry < archive_capacity ? control->archive_cost[entry] : info[entry - archive_capacity].final_cost;
    }
    __device__ TourFingerprint identity(Node entry) const {
        return entry < archive_capacity ? control->archive_identity[entry] : identities[entry - archive_capacity];
    }
    __device__ bool less(Node a, Node b) const {
        if (cost(a) != cost(b)) return cost(a) < cost(b);
        if (identity(a) != identity(b)) return identity(a) < identity(b);
        return canonical_less(view(a), view(b));
    }
    __device__ bool duplicate(Node a, Node b) const {
        return identity(a) == identity(b) && same_tour(view(a), view(b));
    }
};

static __global__ void update_control(Node n, Node ants, const Node* all_tours, const Node* all_positions,
    const FacoDiagnosticInfo* all_info, const TourFingerprint* all_identities, const Colony* colonies,
    Node* all_archive, Node* all_archive_positions, Node* all_scratch, Node* all_scratch_positions,
    ControllerState* controls, std::uint64_t evaluation_limit) {
    const Node colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    auto* archive = all_archive + base * archive_capacity;
    auto* positions = all_archive_positions + base * archive_capacity;
    auto* scratch = all_scratch + base * archive_capacity;
    auto* scratch_positions = all_scratch_positions + base * archive_capacity;
    auto& control = controls[colony]; const auto& state = colonies[colony];
    const auto* info = all_info + colony * ants;
    const auto* identities = all_identities + colony * ants;
    const ArchivePool pool{n, archive, positions, all_tours + base * ants,
        all_positions + base * ants, &control, info, identities};
    __shared__ Node chosen[archive_capacity], count;
    __shared__ double costs[archive_capacity];
    __shared__ TourFingerprint selected_identities[archive_capacity];
    if (threadIdx.x == 0) {
        const bool improved = state.global_cost < control.tracked_global;
        chosen[0] = improved ? archive_capacity + state.iteration_best : 0;
        count = 1;
        for (Node slot = 1; slot < archive_capacity; ++slot) {
            Node selected = UINT32_MAX;
            for (Node candidate = 0; candidate < archive_capacity + ants; ++candidate) {
                if (candidate < archive_capacity && candidate >= control.archive_size) continue;
                if (pool.cost(candidate) > state.global_cost * (1 + archive_quality_band)) continue;
                bool duplicate = false;
                for (Node j = 0; j < count; ++j) if (pool.duplicate(candidate, chosen[j])) { duplicate = true; break; }
                if (!duplicate && (selected == UINT32_MAX || pool.less(candidate, selected))) selected = candidate;
            }
            if (selected == UINT32_MAX) break;
            chosen[count++] = selected;
        }
        for (Node i = 0; i < count; ++i) { costs[i] = pool.cost(chosen[i]); selected_identities[i] = pool.identity(chosen[i]); }
        double returned = 0, work = 0;
        for (Node ant = 0; ant < ants; ++ant) {
            returned += identities[ant] == control.active_identity;
            if (evaluation_limit) work += static_cast<double>(info[ant].local_search.move_evaluations) / evaluation_limit;
        }
        update_feedback(control.feedback, improved, returned / ants, work / ants, ants);
        if (state.epoch_cost < control.tracked_epoch) control.epoch_identity = identities[state.iteration_best];
        control.active_identity = state.source_is_epoch ? control.epoch_identity : identities[state.iteration_best];
        control.tracked_global = state.global_cost; control.tracked_epoch = state.epoch_cost;
    }
    __syncthreads();
    // 先复制到独立scratch，再替换旧档案，不能一边选用旧slot一边覆盖它。
    for (Node i = threadIdx.x; i < count * n; i += blockDim.x) {
        const Node slot = i / n, position = i % n;
        const Node node = pool.view(chosen[slot]).tour[position];
        scratch[i] = node; scratch_positions[slot * n + node] = position;
    }
    __syncthreads();
    for (Node i = threadIdx.x; i < archive_capacity * n; i += blockDim.x) {
        archive[i] = i < count * n ? scratch[i] : 0;
        positions[i] = i < count * n ? scratch_positions[i] : 0;
    }
    if (threadIdx.x == 0) {
        control.archive_size = count;
        for (Node i = 0; i < archive_capacity; ++i) {
            control.archive_cost[i] = i < count ? costs[i] : 0;
            control.archive_identity[i] = i < count ? selected_identities[i] : TourFingerprint{};
        }
    }
}

static __global__ void build_control_features(Node n, Node width, Node colony_count,
    const double* all_xy, const Node* all_primary, const double* all_scale, const double* epsilons,
    const Node* all_samples, const std::uint64_t* keys, Node batch, double elapsed_ratio,
    const Node* all_parent, const Node* all_parent_positions, const Node* all_archive,
    const Node* all_archive_positions, const ControllerState* controls, const Colony* colonies,
    const double* all_trails, const std::uint32_t* experiment_masks, StartRegions* regions,
    Node* alternatives, std::uint32_t* masks, float* features) {
    const Node colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const TourView active{all_parent + base, all_parent_positions + base, n};
    const auto* archive = all_archive + base * archive_capacity;
    const auto* positions = all_archive_positions + base * archive_capacity;
    const auto& control = controls[colony]; const auto& state = colonies[colony];
    const auto* samples = all_samples + colony * sample_capacity;
    const Node sample_count = n < sample_capacity ? n : sample_capacity;
    __shared__ Node alternative;
    __shared__ float gap[2], difference;
    if (threadIdx.x == 0) {
        alternative = archive_capacity; Node largest = 0;
        for (Node i = 0; i < control.archive_size; ++i) {
            const TourView candidate{archive + i * n, positions + i * n, n};
            if (control.archive_identity[i] == control.active_identity && same_tour(candidate, active)) continue;
            const Node missing = sampled_difference(candidate, active, samples, sample_count);
            bool better_tie = false;
            if (alternative != archive_capacity && missing == largest) {
                const auto left = control.archive_identity[i], right = control.archive_identity[alternative];
                better_tie = control.archive_cost[i] < control.archive_cost[alternative] ||
                    (control.archive_cost[i] == control.archive_cost[alternative] &&
                     (left < right || (left == right && canonical_less(candidate,
                       {archive + alternative * n, positions + alternative * n, n}))));
            }
            if (alternative == archive_capacity || missing > largest || better_tie) {
                alternative = i; largest = missing;
            }
        }
        gap[0] = static_cast<float>(unit_clip((state.parent_cost / state.global_cost - 1) / archive_quality_band));
        gap[1] = alternative < archive_capacity ? static_cast<float>(unit_clip(
            (control.archive_cost[alternative] / state.global_cost - 1) / archive_quality_band)) : 0;
        difference = alternative < archive_capacity ? static_cast<float>(largest) / (2 * sample_count) : 0;
        alternatives[colony] = alternative;
        masks[colony] = experiment_masks[colony] & (alternative < archive_capacity ? UINT32_MAX : 0xffffu);
        regions[colony].count = n < region_capacity ? n : region_capacity;
    }
    __syncthreads();
    if (threadIdx.x >= 8) return;
    const Node mode = threadIdx.x / 4, region = threadIdx.x % 4;
    const bool valid = mode == 0 || alternative < archive_capacity;
    auto* nodes = regions[colony].nodes[mode][region];
    for (Node i = 0; i < region_capacity; ++i) nodes[i] = 0;
    float regional[4]{};
    if (valid) {
        const TourView reference = mode == 0 ? active : TourView{archive + alternative * n, positions + alternative * n, n};
        TourView views[archive_capacity];
        for (Node i = 0; i < control.archive_size; ++i) views[i] = {archive + i * n, positions + i * n, n};
        const auto* primary = all_primary + base * width;
        build_region(reference, primary, width, keys[colony], batch, mode, region, nodes);
        region_features(reference, nodes, regions[colony].count, views, control.archive_size,
            all_scale + base, epsilons[colony], state.minimum, state.maximum,
            CoordinateDistance{all_xy + base * 2},
            DevicePheromoneView{primary, all_trails + base * width, width, state.default_trail}, regional);
    }
    for (Node level = 0; level < 4; ++level) {
        const Node action = mode * 16 + region * 4 + level;
        const float values[12]{static_cast<float>(unit_clip(elapsed_ratio)),
            static_cast<float>(control.feedback.stagnant_batches < 32 ? control.feedback.stagnant_batches / 32.0 : 1),
            control.feedback.return_rate, control.feedback.ls_work, static_cast<float>(mode), (level + 1) * 0.25f,
            gap[mode], mode ? difference : 0, regional[0], regional[1], regional[2], regional[3]};
        for (Node feature = 0; feature < 12; ++feature) features[(feature * colony_count + colony) * 32 + action] = values[feature];
    }
}

static __global__ void apply_control_action(Node n, Node ants, Node primary_width, Node ls_width,
    const std::int32_t* actions, const Node* alternatives, const Node* all_archive,
    const Node* all_archive_positions, Node* all_parent, Node* all_parent_positions, Node* all_epoch,
    Colony* colonies, ControllerState* controls, double retention, double p_best,
    double* all_trails, const double* all_heuristic, double* all_products, Node* all_targets,
    Node* all_tours, Node* all_positions, Node* all_parent_positions_per_ant, Node* all_scratch,
    Node* all_pending, std::uint8_t* all_visited, double* all_gains, FacoDiagnosticInfo* all_info,
    TourFingerprint* all_identities, bool initializing = false) {
    const Node colony = blockIdx.x, action = actions[colony];
    const auto base = static_cast<std::size_t>(colony) * n;
    const Node target = 2u << (action % 4);
    for (Node ant = threadIdx.x; ant < ants; ant += blockDim.x) all_targets[colony * ants + ant] = target;
    if (action < 16 && !initializing) return;
    const Node alternative = initializing ? 0 : alternatives[colony];
    const auto* tour = all_archive + (base * archive_capacity + alternative * n);
    const auto* positions = all_archive_positions + (base * archive_capacity + alternative * n);
    auto& state = colonies[colony]; auto& control = controls[colony];
    if (threadIdx.x == 0) {
        state.epoch_cost = state.parent_cost = control.archive_cost[alternative];
        bounds(state, primary_width, retention, p_best); state.default_trail = state.maximum;
        state.source_uniform = 0; state.source_is_epoch = false; state.iteration_best = 0;
        control.tracked_epoch = state.epoch_cost;
        control.active_identity = control.epoch_identity = control.archive_identity[alternative];
        if (!initializing) restart_feedback(control.feedback);
    }
    __syncthreads();
    for (Node i = threadIdx.x; i < n; i += blockDim.x) {
        all_parent[base + i] = all_epoch[base + i] = tour[i];
        all_parent_positions[base + i] = positions[i];
    }
    for (Node i = threadIdx.x; i < n * primary_width; i += blockDim.x) {
        const auto index = base * primary_width + i;
        all_trails[index] = state.maximum; all_products[index] = state.maximum * all_heuristic[index];
    }
    for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(ants) * n; i += blockDim.x) {
        const auto index = base * ants + i;
        all_tours[index] = tour[i % n];
        all_positions[index] = all_parent_positions_per_ant[index] = positions[i % n];
        all_scratch[index] = 0; all_visited[index] = 0;
    }
    for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(ants) * n * 5; i += blockDim.x)
        all_pending[base * ants * 5 + i] = 0;
    for (Node i = threadIdx.x; i < ants * ls_width * 2; i += blockDim.x)
        all_gains[(static_cast<std::size_t>(colony) * ants * ls_width * 2) + i] = 0;
    for (Node ant = threadIdx.x; ant < ants; ant += blockDim.x) {
        all_info[colony * ants + ant] = {}; all_identities[colony * ants + ant] = {};
    }
}

}  // namespace gp_faco::cuda_detail
