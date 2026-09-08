// Escape随机域、固定长度视图与当前蚂蚁工作区；与生产FACO共同内核配合。
#pragma once
#include "control_state.cuh"
#include "gp_faco/escape_slots.hpp"

namespace gp_faco::cuda_detail {

struct EscapeRandom {
    curandStatePhilox4_32_10_t state;
    std::uint64_t draws = 0;
    __device__ EscapeRandom(std::uint64_t seed, Node batch, Node ant, Node event, bool ls) {
        // ant<2^30；两个专属ant域与普通蚂蚁、UINT32_MAX colony域互不相交。
        // 每事件保留2^32个word；耗尽时硬错误，绝不跨入下一事件随机流。
        const Node domain = ls ? 0x80000000u : 0x40000000u;
        curand_init(seed, (static_cast<std::uint64_t>(batch) << 32) | domain | ant,
                    static_cast<std::uint64_t>(event) << 32, &state);
    }
    __device__ Node draw() {
        if (draws == (std::uint64_t{1} << 32)) asm("trap;");
        ++draws; return curand(&state);
    }
    __device__ Node below(Node count) {
        const Node threshold = -count % count;
        Node value; do { value = draw(); } while (value < threshold);
        return value % count;
    }
};

struct EscapeChoices : StochasticChoices {
    DevicePheromoneView pheromone;
    double beta;
    std::uint8_t *anchors, *footprint, *ls_replaced;
    const std::uint8_t* parent_footprint;
    Node* ls_rows;
    EscapeStats* stats;
    bool disabled;
    EscapeMoveEvent* events;
    mutable bool selected_novel = false;
    mutable Node selected_slot = 0;

    template<class Distance>
    __device__ Node next(Node ant, Node current, const std::uint8_t* visited,
        Node step, Node n, Distance distance, const Node* tour, const Node* position,
        const EscapeEdges& allowed) const {
        auto ordinary = random_state(seed, batch, ant, step);
        const double uniform = uniform53(ordinary);
        const auto offset = static_cast<std::size_t>(current) * primary_width;
        EscapeRow row;
        for (Node j = 0; j < primary_width; ++j) row.members[j] = primary[offset + j];
        ++stats->construction_opportunities;
        EscapeRandom random(seed, batch, ant, step, false);
        if (!disabled && (random.draw() & 15u) == 0) {
            ++stats->construction_gates;
            row = escape_slot_view(primary + offset, primary_width,
                backup_width ? backup + static_cast<std::size_t>(current) * backup_width : nullptr,
                backup_width, n, current, random);
            stats->construction_replaced_slots += row.replacements;
        }
        double weights[kEscapePrimaryWidth]{};
        bool available[kEscapePrimaryWidth]{};
        double total = 0;
        Node count = 0, chosen = n;
        selected_novel = false;
        selected_slot = n;
        for (Node j = 0; j < primary_width; ++j) {
            const Node node = row.members[j];
            if (node >= n || visited[node]) continue;
            const auto proposal = prepare_escape_relocation(tour, position, n, current, node,
                                                            allowed.with_permission(row.replaced[j]));
            stats->construction_capacity_rejections += proposal.status == EscapeProposalStatus::Capacity;
            if (!proposal) continue;
            available[j] = true;
            if (row.replaced[j]) {
                const double d = distance(current, node);
                weights[j] = pheromone(current, node) * (d > 0 ? 1.0 / pow(d, beta) : 1.0);
            } else weights[j] = products[offset + j];
            total += weights[j]; ++count; chosen = node; selected_novel = row.replaced[j]; selected_slot = j;
        }
        if (count) {
            if (total > 0) {
                double prefix = 0;
                const double threshold = uniform * total;
                for (Node j = 0; j < primary_width; ++j) if (available[j]) {
                    prefix += weights[j];
                    if (threshold < prefix) { chosen = row.members[j]; selected_novel = row.replaced[j]; selected_slot = j; break; }
                }
            } else {
                Node target = static_cast<Node>(uniform * count);
                if (target >= count) target = count - 1;
                for (Node j = 0; j < primary_width; ++j) if (available[j] && target-- == 0) {
                    chosen = row.members[j]; selected_novel = row.replaced[j]; selected_slot = j; break;
                }
            }
        } else {
            for (Node j = 0; j < backup_width; ++j) {
                const Node node = backup[static_cast<std::size_t>(current) * backup_width + j];
                if (available_node(visited, tour, position, n, current, node, allowed)) { chosen = node; break; }
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
        return chosen;
    }

    template<class Distance>
    __device__ void build_ls_row(Node ant, Node node, Node width, Node n,
        const Node* candidates, Distance distance) const {
        const auto offset = static_cast<std::size_t>(node) * width;
        EscapeRow row;
        for (Node j = 0; j < width; ++j) row.members[j] = candidates[offset + j];
        if (anchors[node] && !disabled) {
            EscapeRandom random(seed, batch, ant, node, true);
            row = escape_slot_view(candidates + offset, width,
                backup_width ? backup + static_cast<std::size_t>(node) * backup_width : nullptr,
                backup_width, n, node, random);
            sort_escape_ls_row(row, node, distance);
        }
        for (Node j = 0; j < width; ++j) {
            ls_rows[offset + j] = row.members[j]; ls_replaced[offset + j] = row.replaced[j];
        }
        footprint[node] = row.replacements != 0;
    }

    __device__ EscapeEdges ls_allowed(Node node, Node slot, Node width, EscapeEdges base) const {
        return base.with_permission(ls_replaced[static_cast<std::size_t>(node) * width + slot]);
    }
    __device__ void record_move(Node index, Node n, EscapeMoveEvent event) const {
        if (!events) return;
        if (index >= n * 2) asm("trap;");
        events[index] = event;
    }
};

struct BatchEscapeChoices : BatchStochasticChoices {
    const StartRegions* regions;
    const std::int32_t* actions;
    const double* trails;
    const Colony* states;
    double beta;
    std::uint8_t *anchors, *footprint, *ls_replaced;
    const std::uint8_t* parent_footprint;
    Node* ls_rows;
    FacoDiagnosticInfo* info;
    Node ls_width;
    bool disabled;
    EscapeMoveEvent* events;
    __device__ EscapeChoices for_ant(Node ant, Node n) const {
        auto ordinary = BatchStochasticChoices::for_ant(ant, n);
        const Node colony = ant / ants;
        if (regions) {
            const Node action = actions[colony];
            ordinary.start_nodes = regions[colony].nodes[action / 16][(action / 4) % 4];
            ordinary.start_count = regions[colony].count;
        }
        const auto base = static_cast<std::size_t>(ant) * n;
        return {ordinary, {ordinary.primary, trails + static_cast<std::size_t>(colony) * n * primary_width,
                    primary_width, states[colony].default_trail}, beta,
            anchors + base, footprint + base, ls_replaced + base * ls_width,
            parent_footprint + static_cast<std::size_t>(colony) * n, ls_rows + base * ls_width,
            &info[ant].escape, disabled, events ? events + base * 2 : nullptr};
    }
};

struct BatchEscapeEdges : BatchSparseGraphView {
    EscapeCache* caches;
    __device__ EscapeEdges for_ant(Node ant, Node n) const {
        return {BatchSparseGraphView::for_ant(ant, n), caches + ant, false};
    }
};

}  // namespace gp_faco::cuda_detail
