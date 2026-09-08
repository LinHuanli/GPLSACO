#pragma once

#include "gp_faco/faco_cpu.hpp"

#ifdef __CUDACC__
#define GPFACO_CONTROL_HD __host__ __device__
#else
#define GPFACO_CONTROL_HD
#endif

namespace gp_faco {

constexpr Node archive_capacity = 4, region_capacity = 16, sample_capacity = 64;
constexpr double archive_quality_band = 0.02;

GPFACO_CONTROL_HD inline std::uint64_t control_mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct TourFingerprint {
    std::uint64_t first = 0, second = 0;
    GPFACO_CONTROL_HD bool operator==(TourFingerprint other) const {
        return first == other.first && second == other.second;
    }
    GPFACO_CONTROL_HD bool operator!=(TourFingerprint other) const { return !(*this == other); }
    GPFACO_CONTROL_HD bool operator<(TourFingerprint other) const {
        return first < other.first || (first == other.first && second < other.second);
    }
};

struct TourView {
    const Node* tour;
    const Node* positions;
    Node n;
    GPFACO_CONTROL_HD Node successor(Node node) const { return tour[(positions[node] + 1) % n]; }
    GPFACO_CONTROL_HD Node predecessor(Node node) const { return tour[(positions[node] + n - 1) % n]; }
    GPFACO_CONTROL_HD bool contains(Node a, Node b) const { return successor(a) == b || predecessor(a) == b; }
};

GPFACO_CONTROL_HD inline TourFingerprint fingerprint_edge(Node a, Node b) {
    if (b < a) { const Node temp = a; a = b; b = temp; }
    const auto edge = (static_cast<std::uint64_t>(a) << 32) | b;
    return {control_mix(edge ^ 0xb5ad4eceda1ce2a9ULL), control_mix(edge ^ 0xd6e8feb86659fd93ULL)};
}

GPFACO_CONTROL_HD inline TourFingerprint fingerprint(TourView view) {
    TourFingerprint result;
    for (Node i = 0; i < view.n; ++i) {
        const auto edge = fingerprint_edge(view.tour[i], view.tour[(i + 1) % view.n]);
        result.first ^= edge.first; result.second ^= edge.second;
    }
    return result;
}

GPFACO_CONTROL_HD inline bool same_tour(TourView a, TourView b) {
    if (a.n != b.n) return false;
    for (Node node = 0; node < a.n; ++node)
        if (!b.contains(node, a.successor(node)) || !b.contains(node, a.predecessor(node))) return false;
    return true;
}

GPFACO_CONTROL_HD inline bool canonical_less(TourView a, TourView b) {
    const bool forward_a = a.successor(0) < a.predecessor(0);
    const bool forward_b = b.successor(0) < b.predecessor(0);
    Node left = 0, right = 0;
    for (Node i = 1; i < a.n; ++i) {
        left = forward_a ? a.successor(left) : a.predecessor(left);
        right = forward_b ? b.successor(right) : b.predecessor(right);
        if (left != right) return left < right;
    }
    return false;
}

struct ControlRandom {
    std::uint64_t key, counter = 0;
    GPFACO_CONTROL_HD Node bounded(Node bound) {
        const auto threshold = -static_cast<std::uint64_t>(bound) % bound;
        std::uint64_t word;
        do { word = control_mix(key + counter++); } while (word < threshold);
        return word % bound;
    }
};

// 将剩余集合中的秩映射为城市ID；已有集合最多64，不扫描整张tour。
GPFACO_CONTROL_HD inline Node remaining_node(Node rank, const Node* existing, Node count) {
    Node sorted[sample_capacity];
    for (Node i = 0; i < count; ++i) {
        Node j = i;
        while (j && sorted[j - 1] > existing[i]) { sorted[j] = sorted[j - 1]; --j; }
        sorted[j] = existing[i];
    }
    Node node = rank;
    for (Node i = 0; i < count; ++i) if (sorted[i] <= node) ++node;
    return node;
}

GPFACO_CONTROL_HD inline void sample_nodes(Node n, std::uint64_t instance_key, Node* output) {
    const Node count = n < sample_capacity ? n : sample_capacity;
    ControlRandom random{control_mix(instance_key ^ 0x6a09e667f3bcc909ULL)};
    for (Node i = 0; i < count; ++i) output[i] = remaining_node(random.bounded(n - i), output, i);
}

struct StartRegions {
    Node nodes[2][4][region_capacity]{};
    Node count = 0;
};

GPFACO_CONTROL_HD inline bool has_node(const Node* nodes, Node count, Node node) {
    for (Node i = 0; i < count; ++i) if (nodes[i] == node) return true;
    return false;
}

GPFACO_CONTROL_HD inline void build_region(TourView reference, const Node* primary, Node width,
    std::uint64_t key, Node batch, Node mode, Node region, Node* nodes) {
    const Node n = reference.n, count = n < region_capacity ? n : region_capacity;
    ControlRandom random{control_mix(key ^ 0xbb67ae8584caa73bULL ^
        control_mix(static_cast<std::uint64_t>(batch) ^ 0x3c6ef372fe94f82bULL) ^
        control_mix((mode * 4 + region + 1) ^ 0xa54ff53a5f1d36f1ULL))};
    Node used = 0;
    if (region < 2) {
        const Node start = random.bounded(n);
        for (Node i = 0; i < count; ++i) nodes[i] = reference.tour[(start + i) % n];
        return;
    }
    if (region == 2) {
        nodes[used++] = random.bounded(n);
        for (Node head = 0; head < used && used < count; ++head)
            for (Node j = 0; j < width && used < count; ++j) {
                const Node node = primary[nodes[head] * width + j];
                if (!has_node(nodes, used, node)) nodes[used++] = node;
            }
    }
    while (used < count) {
        const Node node = remaining_node(random.bounded(n - used), nodes, used);
        nodes[used++] = node;
    }
}

GPFACO_CONTROL_HD inline Node sampled_difference(TourView candidate, TourView active,
                                                 const Node* samples, Node count) {
    Node missing = 0;
    for (Node i = 0; i < count; ++i) {
        const Node node = samples[i];
        missing += !active.contains(node, candidate.successor(node));
        missing += !active.contains(node, candidate.predecessor(node));
    }
    return missing;
}

GPFACO_CONTROL_HD inline double unit_clip(double value) { return value < 0 ? 0 : value > 1 ? 1 : value; }

struct FeedbackState {
    float return_rate = 0, ls_work = 0;
    std::uint64_t stagnant_batches = 0, epoch_batches = 0, restarts = 0;
};

// 进程内设备状态/诊断快照；不将其原始内存当成公共序列化格式。
struct ControllerState {
    FeedbackState feedback;
    Node archive_size = 0;
    double archive_cost[archive_capacity]{};
    TourFingerprint archive_identity[archive_capacity]{};
    double tracked_global = 0, tracked_epoch = 0;
    TourFingerprint active_identity, epoch_identity;
};

GPFACO_CONTROL_HD inline void update_feedback(FeedbackState& state, bool improved,
    double returned_fraction, double work_fraction, Node ants) {
    if (!ants) return;
    const float returned = static_cast<float>(unit_clip(returned_fraction));
    const float work = static_cast<float>(unit_clip(work_fraction));
    state.return_rate += (returned - state.return_rate) * 0.0625f;
    state.ls_work += (work - state.ls_work) * 0.0625f;
    if (improved) state.stagnant_batches = 0;
    else if (state.stagnant_batches != UINT64_MAX) ++state.stagnant_batches;
    if (state.epoch_batches != UINT64_MAX) ++state.epoch_batches;
}

GPFACO_CONTROL_HD inline void restart_feedback(FeedbackState& state) {
    state.return_rate = state.ls_work = 0; state.epoch_batches = 0;
    if (state.restarts != UINT64_MAX) ++state.restarts;
}

template<class Distance, class Pheromone>
GPFACO_CONTROL_HD void region_features(TourView reference, const Node* nodes, Node count,
    const TourView* archive, Node archive_size, const double* local_scale, double epsilon,
    double minimum, double maximum, Distance distance, Pheromone pheromone, float* output) {
    double excess = 0, disagreement = 0, strength = 0, dispersion = 0;
    for (Node i = 0; i < count; ++i) {
        const Node node = nodes[i], previous = reference.predecessor(node), next = reference.successor(node);
        const double scale = local_scale[node] < epsilon ? epsilon : local_scale[node];
        excess += unit_clip(((0.5 * distance(node, previous) + 0.5 * distance(node, next)) / scale - 1) / 3);
        const Node neighbors[2]{previous, next};
        for (Node neighbor : neighbors) {
            for (Node j = 0; j < archive_size; ++j) disagreement += !archive[j].contains(node, neighbor);
            strength += minimum == maximum ? 1 : unit_clip((pheromone(node, neighbor) - minimum) / (maximum - minimum));
            dispersion += !has_node(nodes, count, neighbor);
        }
    }
    output[0] = static_cast<float>(excess / count);
    output[1] = static_cast<float>(disagreement / (2 * count * archive_size));
    output[2] = static_cast<float>(strength / (2 * count));
    output[3] = static_cast<float>(dispersion / (2 * count));
}

}  // namespace gp_faco

#undef GPFACO_CONTROL_HD
