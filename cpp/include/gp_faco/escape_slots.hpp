#pragma once

#include "gp_faco/escape_edges.hpp"

#ifdef __CUDACC__
#define GPFACO_SLOT_HD __host__ __device__
#else
#define GPFACO_SLOT_HD
#endif

namespace gp_faco {

struct EscapeRow {
    Node members[kEscapeLsWidth]{};
    std::uint8_t replaced[kEscapeLsWidth]{};
    Node valid = 0, replacements = 0;
};

// Random::draw提供均匀uint32，below采用无模偏差抽样；行构建不读取visited或tour。
template<class Random>
GPFACO_SLOT_HD EscapeRow escape_slot_view(const Node* original, Node width,
    const Node* backup, Node backup_width, Node n, Node from, Random& random) {
    EscapeRow result;
    for (Node j = 0; j < width; ++j) {
        result.members[j] = original[j];
        result.valid += original[j] < n;
    }
    Node wanted = result.valid / 4;
    if (result.valid % 4) wanted += (random.draw() & 3u) < result.valid % 4;
    if (!wanted) return result;
    Node eligible[kEscapeBackupWidth], count = 0;
    for (Node j = 0; j < backup_width; ++j) {
        const Node node = backup[j];
        if (node >= n) break;
        if (node == from) continue;
        bool duplicate = false;
        for (Node k = 0; k < result.valid; ++k) duplicate |= original[k] == node;
        for (Node k = 0; k < count; ++k) duplicate |= eligible[k] == node;
        if (!duplicate) eligible[count++] = node;
    }
    wanted = wanted < count ? wanted : count;
    Node slots[kEscapeLsWidth];
    for (Node j = 0; j < result.valid; ++j) slots[j] = j;
    for (Node j = 0; j < wanted; ++j) {
        const Node pick_slot = j + random.below(result.valid - j);
        const Node slot = slots[pick_slot]; slots[pick_slot] = slots[j]; slots[j] = slot;
        const Node pick_node = j + random.below(count - j);
        const Node node = eligible[pick_node]; eligible[pick_node] = eligible[j]; eligible[j] = node;
        result.members[slot] = node; result.replaced[slot] = 1;
    }
    result.replacements = wanted;
    return result;
}

template<class Distance>
GPFACO_SLOT_HD void sort_escape_ls_row(EscapeRow& row, Node from, Distance distance) {
    // 槽位来源标记跟随成员排序，2-opt许可不能留在被移动的旧位置上。
    for (Node j = 1; j < row.valid; ++j) {
        const Node node = row.members[j];
        const auto flag = row.replaced[j];
        const double d = distance(from, node);
        Node at = j;
        while (at) {
            const Node prior = row.members[at - 1];
            const double previous = distance(from, prior);
            if (previous < d || (previous == d && prior < node)) break;
            row.members[at] = prior; row.replaced[at] = row.replaced[at - 1]; --at;
        }
        row.members[at] = node; row.replaced[at] = flag;
    }
}

struct EscapeStats {
    std::uint64_t construction_opportunities = 0, construction_gates = 0;
    std::uint64_t construction_replaced_slots = 0, escape_relocations = 0;
    std::uint64_t ls_anchor_nodes = 0, ls_replaced_slots = 0;
    std::uint64_t old_view_reactivations = 0, anchor_reactivations = 0;
    std::uint64_t new_edges = 0, construction_capacity_rejections = 0, ls_capacity_rejections = 0;
};

// 指针为空时普通/Hard控制内核不处理足迹；数组均按node ID而非tour位置索引。
struct EscapeFootprints {
    std::uint8_t *ants = nullptr, *parent = nullptr, *epoch = nullptr, *global = nullptr;
    std::uint8_t *archive = nullptr, *scratch = nullptr;
};

// 仅诊断记录已接受移动，CPU从父tour独立重放，不相信内核自己声明的新增边。
struct EscapeMoveEvent {
    Node kind, a, b, slot, cache_size;  // kind=0重定位、1前向2-opt、2后向2-opt。
    bool permit_novel;
};

}  // namespace gp_faco

#undef GPFACO_SLOT_HD
