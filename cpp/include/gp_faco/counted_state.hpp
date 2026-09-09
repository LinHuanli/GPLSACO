#pragma once

#include "gp_faco/fixed_faco_gpu.hpp"

#include <string>

namespace gp_faco {

struct BatchTask { std::uint64_t instance_key, seed; };

// 原生缓冲保留实际 ABI 字节，读取时检查结构版本和缓冲形状。
struct CountedStateBuffer {
    std::string name;
    std::uint64_t element_bytes = 0, planes = 1;
    std::vector<std::uint8_t> bytes;
};

struct CountedStateIncumbent {
    std::vector<Node> tour;
    double cost = 0;
};

struct CountedState {
    Node dimension = 0, colonies = 0;
    FixedFacoSettings settings;
    std::uint32_t experiment_mask = UINT32_MAX;
    std::uint64_t completed_batches = 0, progress_evaluation_limit = 0;
    std::vector<BatchTask> tasks;
    // 主机已提交incumbent也属于状态，不能用设备global的近似成本重新择优。
    std::vector<CountedStateIncumbent> incumbents;
    std::vector<CountedStateBuffer> buffers;

    void seal();
    void validate() const;
    std::vector<std::uint8_t> serialize() const;
    static CountedState deserialize(const std::vector<std::uint8_t>& bytes);
    CountedState select_colony(Node colony) const;
};

struct ForkIntervention {
    bool enabled = false;
    std::uint64_t seed = 0;
    Node region = 0, mne = 2;
};

bool same_settings(const FixedFacoSettings& a, const FixedFacoSettings& b);

}  // namespace gp_faco
