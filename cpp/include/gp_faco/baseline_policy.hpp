#pragma once

#include "gp_faco/control_ops.hpp"

#include <cmath>
#include <stdexcept>

#ifdef __CUDACC__
#define GPFACO_BASELINE_HD __host__ __device__
#else
#define GPFACO_BASELINE_HD
#endif

namespace gp_faco {

enum class BaselineKind { Static, Rule };
enum class StaticRestart { None, Periodic, Bernoulli };

// 无标签、无Python回调的有限参数族；公开JSON规格与该结构分离。
struct BaselinePolicy {
    BaselineKind kind = BaselineKind::Static;
    Node mne_level = 0, max_mne_level = 0, region = 0;
    StaticRestart restart_mode = StaticRestart::None;
    Node restart_period = 0;
    double restart_probability = 0;
    Node stagnation_step = 0, restart_stagnation = 0, restart_cooldown = 0;
};

inline std::uint32_t baseline_keep_mask(const BaselinePolicy& policy) {
    std::uint32_t mask = 0;
    for (Node level = policy.mne_level; level <= policy.max_mne_level; ++level)
        mask |= 1u << (policy.region * 4 + level);
    return mask;
}

inline void validate_baseline(const BaselinePolicy& p, std::uint32_t mask = UINT32_MAX) {
    if ((p.kind != BaselineKind::Static && p.kind != BaselineKind::Rule) ||
        p.mne_level > 3 || p.max_mne_level > 3 || p.max_mne_level < p.mne_level || p.region > 3 ||
        !std::isfinite(p.restart_probability) || p.restart_probability < 0 || p.restart_probability > 1)
        throw std::invalid_argument("基线类别、MNE、区域或概率无效");
    if (p.kind == BaselineKind::Static) {
        if (p.max_mne_level != p.mne_level || p.stagnation_step || p.restart_stagnation || p.restart_cooldown)
            throw std::invalid_argument("静态控制不能附带反馈规则");
        if (!((p.restart_mode == StaticRestart::None && !p.restart_period && !p.restart_probability) ||
              (p.restart_mode == StaticRestart::Periodic && p.restart_period && !p.restart_probability) ||
              (p.restart_mode == StaticRestart::Bernoulli && !p.restart_period)))
            throw std::invalid_argument("静态重启模式与周期/概率不符");
    } else if (p.restart_mode != StaticRestart::None || p.restart_period || p.restart_probability ||
               ((p.max_mne_level == p.mne_level) != (p.stagnation_step == 0)) ||
               ((p.restart_stagnation == 0) != (p.restart_cooldown == 0))) {
        throw std::invalid_argument("规则控制需要规范的停滞升级/重启冷却参数");
    }
    const auto required = baseline_keep_mask(p);
    if ((mask & required) != required)
        throw std::invalid_argument("实验mask必须保留基线可能选择的保持动作");
}

// 已验证配置；random_01由独立colony级随机槽产生，不消耗蚂蚁选点流。
GPFACO_BASELINE_HD inline Node baseline_action(const BaselinePolicy& p, const FeedbackState& feedback,
                                              Node batch, double random_01, std::uint32_t mask) {
    Node level = p.mne_level;
    bool restart = false;
    if (p.kind == BaselineKind::Static) {
        restart = p.restart_mode == StaticRestart::Periodic ?
            batch > 0 && batch % p.restart_period == 0 :
            p.restart_mode == StaticRestart::Bernoulli && random_01 < p.restart_probability;
    } else {
        if (p.stagnation_step) {
            const auto extra = feedback.stagnant_batches / p.stagnation_step;
            const Node room = p.max_mne_level - p.mne_level;
            level += extra < room ? static_cast<Node>(extra) : room;
        }
        // 全局停滞跨epoch保留；实际完成epoch批次数提供显式冷却，避免逐批重启。
        restart = p.restart_stagnation && feedback.stagnant_batches >= p.restart_stagnation &&
            feedback.epoch_batches >= p.restart_cooldown;
    }
    const Node keep = p.region * 4 + level, requested = keep + 16;
    return restart && (mask & (1u << requested)) ? requested : keep;
}

}  // namespace gp_faco

#undef GPFACO_BASELINE_HD
