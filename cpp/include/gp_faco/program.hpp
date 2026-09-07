#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace gp_faco {

// 固定 ABI 的内存对象只在进程内使用；公共序列化使用验证后的 JSON。
struct Program {
    std::uint16_t ir_version = 1;
    std::uint16_t numeric_spec_id = 1;
    std::uint16_t feature_spec_id = 1;
    std::uint16_t length = 0;
    std::uint16_t constants_count = 0;
    std::uint8_t opcode[63]{};
    std::uint8_t operand[63]{};
    float constants[63]{};
};

struct Scores {
    std::vector<float> scores;
    std::vector<std::int32_t> actions;
};

void validate_program(const Program& program);
void validate_features(const std::vector<float>& features,
                       const std::vector<std::uint32_t>& masks);
Scores score_cpu(const Program& program, const std::vector<float>& features,
                 const std::vector<std::uint32_t>& masks);
Scores score_cuda(const Program& program, const std::vector<float>& features,
                  const std::vector<std::uint32_t>& masks);

}  // namespace gp_faco
