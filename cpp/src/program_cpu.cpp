#include "gp_faco/program.hpp"
#include "gp_faco/program_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gp_faco {

void validate_program(const Program& p) {
    if (p.ir_version != 1 || p.numeric_spec_id != 1 ||
        (p.feature_spec_id != 1 && p.feature_spec_id != 2) ||
        p.length == 0 || p.length > 63 || p.constants_count > 63) {
        throw std::invalid_argument("未知版本或程序大小超限");
    }
    for (int i = 0; i < p.constants_count; ++i) {
        if (!std::isfinite(p.constants[i]) || std::abs(p.constants[i]) > 2.0f) {
            throw std::invalid_argument("ERC 必须有限且在 [-2,2]");
        }
    }
    std::vector<int> depths;
    for (int i = 0; i < p.length; ++i) {
        const auto op = p.opcode[i];
        const auto operand = p.operand[i];
        if (op == 0 || op == 1) {
            if ((op == 0 && operand >= 12) || (op == 1 && operand >= p.constants_count)) {
                throw std::invalid_argument("terminal 下标越界");
            }
            depths.push_back(0);
        } else if (op >= 2 && op <= 8) {
            const std::size_t arity = op == 7 ? 1 : 2;
            if (operand != 0 || depths.size() < arity) {
                throw std::invalid_argument("函数操作数非零或栈下溢");
            }
            const auto first = depths.end() - static_cast<std::ptrdiff_t>(arity);
            const int depth = *std::max_element(first, depths.end()) + 1;
            depths.erase(first, depths.end());
            if (depth > 5) throw std::invalid_argument("树深超过 5");
            depths.push_back(depth);
        } else {
            throw std::invalid_argument("未知 opcode");
        }
        if (depths.size() > 6) throw std::invalid_argument("栈峰值超过 6");
    }
    if (depths.size() != 1) throw std::invalid_argument("最终栈不是单值");
}

void validate_features(const std::vector<float>& features,
                       const std::vector<std::uint32_t>& masks) {
    if (masks.empty() || features.size() != 12 * 32 * masks.size()) {
        throw std::invalid_argument("特征或 mask 形状错误");
    }
    for (float value : features) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("feature spec v1 输入必须在 [0,1]");
        }
    }
    for (auto mask : masks) {
        if (!mask) throw std::invalid_argument("没有合法动作");
    }
}

Scores score_cpu(const Program& p, const std::vector<float>& features,
                 const std::vector<std::uint32_t>& masks) {
    validate_program(p);
    validate_features(features, masks);
    const auto colonies = masks.size();
    Scores output{std::vector<float>(colonies * 32), std::vector<std::int32_t>(colonies)};
    for (std::size_t colony = 0; colony < colonies; ++colony) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int action = 0; action < 32; ++action) {
            float stack[6]{};
            int top = 0;
            for (int i = 0; i < p.length; ++i) {
                const int op = p.opcode[i];
                if (op == 0) {
                    stack[top++] = features[(p.operand[i] * colonies + colony) * 32 + action];
                } else if (op == 1) {
                    stack[top++] = p.constants[p.operand[i]];
                } else {
                    // 二元函数先弹右再弹左，避免 SUB/AQ 方向颠倒。
                    const float right = op == 7 ? 0.0f : stack[--top];
                    const float left = stack[--top];
                    stack[top++] = apply_operation(op, left, right);
                }
            }
            output.scores[colony * 32 + action] = stack[0];
            if ((masks[colony] >> action) & 1u) maximum = std::max(maximum, stack[0]);
        }
        // 先求全局最大，再取容差内最小 ID，避免近似比较不传递。
        for (int action = 0; action < 32; ++action) {
            if (((masks[colony] >> action) & 1u) &&
                maximum - output.scores[colony * 32 + action] <= 1e-6f) {
                output.actions[colony] = action;
                break;
            }
        }
    }
    return output;
}

}  // namespace gp_faco
