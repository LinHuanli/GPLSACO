#pragma once

#include "gp_faco/baseline_policy.hpp"

#ifdef __CUDACC__
#define GPFACO_FACTORIAL_HD __host__ __device__
#else
#define GPFACO_FACTORIAL_HD
#endif

namespace gp_faco {

enum class FactorialVariant { M00, M10, M01, M11 };

// 固定的是同一规则及参数，规则读取当前变体自己的已提交状态。
struct FactorialPolicy {
    FactorialVariant variant = FactorialVariant::M11;
    BaselinePolicy baseline;
};

inline void validate_factorial(const FactorialPolicy& p, std::uint32_t mask = UINT32_MAX) {
    if (p.variant != FactorialVariant::M00 && p.variant != FactorialVariant::M10 &&
        p.variant != FactorialVariant::M01 && p.variant != FactorialVariant::M11)
        throw std::invalid_argument("未知E2析因变体");
    validate_baseline(p.baseline);
    if (!(mask & 0xffffu)) throw std::invalid_argument("析因mask必须保留保持动作");
    if (p.variant == FactorialVariant::M00 || p.variant == FactorialVariant::M01)
        validate_baseline(p.baseline, mask);
}

// legal_mask已包含档案替代解可行性。局部学习时，重启请求不能被未学习的j/k干扰。
GPFACO_FACTORIAL_HD inline std::uint32_t factorial_mask(const FactorialPolicy& p,
    const FeedbackState& feedback, Node batch, double uniform, std::uint32_t legal_mask) {
    if (p.variant == FactorialVariant::M11) return legal_mask;
    if (p.variant == FactorialVariant::M00)
        return 1u << baseline_action(p.baseline, feedback, batch, uniform, legal_mask);
    const Node requested = baseline_action(p.baseline, feedback, batch, uniform, UINT32_MAX);
    if (p.variant == FactorialVariant::M01) {
        const Node keep = requested % 16;
        return legal_mask & ((1u << keep) | (1u << (keep + 16)));
    }
    const auto allowed = legal_mask & (requested >= 16 ? 0xffff0000u : 0xffffu);
    // 无合法替代解或外部mask关闭全部重启时，保持搜索中心；不积压重启请求。
    return allowed ? allowed : legal_mask & 0xffffu;
}

}  // namespace gp_faco

#undef GPFACO_FACTORIAL_HD
