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
    bool force_collisions = false, Node geometry_count = 0) {
    const Node colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const auto geometry = geometry_count ? colony % geometry_count : colony;
    const auto* initial = all_initial + static_cast<std::size_t>(geometry) * n;
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
        state.archive_cost[0] = state.tracked_global = state.tracked_epoch = costs[geometry];
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
        if (a == b) return true;
        return identity(a) == identity(b) && same_tour(view(a), view(b));
    }
    __device__ Node better(Node a, Node b) const {
        if (a == UINT32_MAX) return b;
        if (b == UINT32_MAX) return a;
        if (a == b) return a;
        if (cost(a) != cost(b)) return cost(a) < cost(b) ? a : b;
        if (identity(a) != identity(b)) return identity(a) < identity(b) ? a : b;
        const int order = canonical_compare(view(a), view(b));
        if (order) return order < 0 ? a : b;
        return min(a, b);  // 完全相等时保留原串行循环中最先出现的候选。
    }
};

static __global__ void update_control(Node n, Node ants, const Node* all_tours, const Node* all_positions,
    const FacoDiagnosticInfo* all_info, const TourFingerprint* all_identities, const Colony* colonies,
    Node* all_archive, Node* all_archive_positions, Node* all_scratch, Node* all_scratch_positions,
    ControllerState* controls, std::uint64_t evaluation_limit, EscapeFootprints footprints = {}) {
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
    __shared__ Node best[128];
    if (threadIdx.x == 0) {
        const bool improved = state.global_cost < control.tracked_global;
        chosen[0] = improved ? archive_capacity + state.iteration_best : 0;
        count = 1;
    }
    __syncthreads();
    // 一个线程最多负责两个候选；排除状态跨档案slot复用，只比较新选入的路线。
    bool excluded[2]{};
    for (Node slot = 1; slot < archive_capacity; ++slot) {
        Node selected = UINT32_MAX;
        // 每个线程筛选不同候选；比较规则与串行选择相同，原有碰撞后邻接比较保留。
        for (Node candidate = threadIdx.x; candidate < archive_capacity + ants; candidate += blockDim.x) {
                const Node local = candidate / blockDim.x;
                // 正式128蚂蚁走缓存路径；其他诊断形状保持通用循环。
                const bool cached = ants <= 128;
                if (cached && excluded[local]) continue;
                if (candidate < archive_capacity && candidate >= control.archive_size) continue;
                if (pool.cost(candidate) > state.global_cost * (1 + archive_quality_band)) continue;
                bool duplicate = false;
                for (Node j = cached ? count - 1 : 0; j < count; ++j)
                    if (pool.duplicate(candidate, chosen[j])) { duplicate = true; break; }
                if (cached) excluded[local] = duplicate;
                if (!duplicate) selected = pool.better(selected, candidate);
        }
        best[threadIdx.x] = selected;
        __syncthreads();
        for (Node stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) best[threadIdx.x] = pool.better(best[threadIdx.x], best[threadIdx.x + stride]);
            __syncthreads();
        }
        if (best[0] == UINT32_MAX) break;
        if (threadIdx.x == 0) chosen[count++] = best[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const bool improved = state.global_cost < control.tracked_global;
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
        if (footprints.parent) {
            const Node entry = chosen[slot];
            footprints.scratch[base * archive_capacity + i] = entry < archive_capacity
                ? footprints.archive[base * archive_capacity + static_cast<std::size_t>(entry) * n + position]
                : footprints.ants[base * ants + static_cast<std::size_t>(entry - archive_capacity) * n + position];
        }
    }
    __syncthreads();
    for (Node i = threadIdx.x; i < archive_capacity * n; i += blockDim.x) {
        archive[i] = i < count * n ? scratch[i] : 0;
        positions[i] = i < count * n ? scratch_positions[i] : 0;
        if (footprints.parent) footprints.archive[base * archive_capacity + i] =
            i < count * n ? footprints.scratch[base * archive_capacity + i] : 0;
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
    Node* alternatives, std::uint32_t* masks, float* features, Node geometry_count = 0,
    std::uint32_t regional_mask = 15, const std::uint32_t* program_masks = nullptr) {
    const Node colony = blockIdx.x;
    const auto base = static_cast<std::size_t>(colony) * n;
    const Node geometry = geometry_count ? colony % geometry_count : colony;
    if (program_masks) regional_mask = program_masks[colony / geometry_count];
    const auto geometry_base = static_cast<std::size_t>(geometry) * n;
    const TourView active{all_parent + base, all_parent_positions + base, n};
    const auto* archive = all_archive + base * archive_capacity;
    const auto* positions = all_archive_positions + base * archive_capacity;
    const auto& control = controls[colony]; const auto& state = colonies[colony];
    const auto* samples = all_samples + geometry * sample_capacity;
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
    // 每区域16线程分别计算节点贡献，lane0仍按原顺序累加，保持逐位数值语义。
    const Node group = threadIdx.x / 16, lane = threadIdx.x % 16;
    const Node mode = group / 4, region = group % 4;
    const bool valid = mode == 0 || alternative < archive_capacity;
    auto* nodes = regions[colony].nodes[mode][region];
    __shared__ double contributions[128][5];
    float regional[4]{};
    if (lane == 0) {
        for (Node i = 0; i < region_capacity; ++i) nodes[i] = 0;
        if (valid) {
            const TourView reference = mode == 0 ? active : TourView{archive + alternative * n, positions + alternative * n, n};
            build_region(reference, all_primary + geometry_base * width, width, keys[colony], batch, mode, region, nodes);
        }
    }
    __syncthreads();
    auto* values = contributions[threadIdx.x];
    for (Node i = 0; i < 5; ++i) values[i] = 0;
    if (valid && lane < regions[colony].count) {
        const TourView reference = mode == 0 ? active : TourView{archive + alternative * n, positions + alternative * n, n};
        const Node node = nodes[lane], previous = reference.predecessor(node), next = reference.successor(node);
        if (regional_mask & 1) {
            const CoordinateDistance distance{all_xy + geometry_base * 2};
            const double scale = all_scale[geometry_base + node] < epsilons[geometry] ? epsilons[geometry] : all_scale[geometry_base + node];
            values[0] = unit_clip(((0.5 * distance(node, previous) + 0.5 * distance(node, next)) / scale - 1) / 3);
        }
        const Node neighbors[2]{previous, next};
        const DevicePheromoneView pheromone{all_primary + geometry_base * width, all_trails + base * width, width, state.default_trail};
        for (Node k = 0; k < 2; ++k) {
            const Node neighbor = neighbors[k];
            if (regional_mask & 2) for (Node j = 0; j < control.archive_size; ++j)
                values[1] += !TourView{archive + j * n, positions + j * n, n}.contains(node, neighbor);
            if (regional_mask & 4) values[2 + k] = state.minimum == state.maximum ? 1 :
                unit_clip((pheromone(node, neighbor) - state.minimum) / (state.maximum - state.minimum));
            if (regional_mask & 8) values[4] += !has_node(nodes, regions[colony].count, neighbor);
        }
    }
    __syncthreads();
    if (lane != 0) return;
    if (valid) {
        double excess = 0, disagreement = 0, strength = 0, dispersion = 0;
        const Node count = regions[colony].count;
        for (Node i = 0; i < count; ++i) {
            const auto* v = contributions[group * 16 + i];
            excess += v[0]; disagreement += v[1];
            strength += v[2]; strength += v[3]; dispersion += v[4];
        }
        regional[0] = static_cast<float>(excess / count);
        regional[1] = static_cast<float>(disagreement / (2 * count * control.archive_size));
        regional[2] = static_cast<float>(strength / (2 * count));
        regional[3] = static_cast<float>(dispersion / (2 * count));
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
    TourFingerprint* all_identities, bool initializing = false, EscapeFootprints footprints = {},
    Node geometry_count = 0, bool compact_workspace = false) {
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
        if (footprints.parent) footprints.parent[base + i] = footprints.epoch[base + i] =
            footprints.archive[base * archive_capacity + static_cast<std::size_t>(alternative) * n + i];
    }
    for (Node i = threadIdx.x; i < n * primary_width; i += blockDim.x) {
        const auto index = base * primary_width + i;
        const auto geometry_index = geometry_count ?
            static_cast<std::size_t>(colony % geometry_count) * n * primary_width + i : index;
        all_trails[index] = state.maximum; all_products[index] = state.maximum * all_heuristic[geometry_index];
    }
    // 紧凑生产路径在构造内完整覆盖有效工作区，无需重启时重复清空数 GiB 暂存数据。
    if (compact_workspace) return;
    for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(ants) * n; i += blockDim.x) {
        const auto index = base * ants + i;
        all_tours[index] = tour[i % n];
        all_positions[index] = all_parent_positions_per_ant[index] = positions[i % n];
        all_scratch[index] = 0; all_visited[index] = 0;
        if (footprints.parent) footprints.ants[index] = 0;
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
