// 只有诊断调用创建events。复用同一组句柄；已有的批末同步完成后读取，不插入额外同步。
#pragma once

#include "gp_faco/profiling.hpp"
#include "colony_state.cuh"

namespace gp_faco::cuda_detail {

class ProfileEvents {
public:
    ProfileEvents() {
        try {
            for (auto& event : events_) checked(cudaEventCreate(&event));
        } catch (...) {
            for (auto event : events_) if (event) cudaEventDestroy(event);
            throw;
        }
    }
    ~ProfileEvents() { for (auto event : events_) if (event) cudaEventDestroy(event); }
    ProfileEvents(const ProfileEvents&) = delete;
    ProfileEvents& operator=(const ProfileEvents&) = delete;
    void reset() { active_.fill(false); }
    void begin(GpuProfileStage stage) {
        const auto i = static_cast<unsigned>(stage);
        checked(cudaEventRecord(events_[2 * i])); active_[i] = true;
    }
    void end(GpuProfileStage stage) {
        checked(cudaEventRecord(events_[2 * static_cast<unsigned>(stage) + 1]));
    }
    std::array<double, gpu_profile_stage_count> elapsed() const {
        std::array<double, gpu_profile_stage_count> result{};
        for (unsigned i = 0; i < gpu_profile_stage_count; ++i) if (active_[i]) {
            float milliseconds = 0;
            checked(cudaEventElapsedTime(&milliseconds, events_[2 * i], events_[2 * i + 1]));
            result[i] = milliseconds;
        }
        return result;
    }
private:
    std::array<cudaEvent_t, 2 * gpu_profile_stage_count> events_{};
    std::array<bool, gpu_profile_stage_count> active_{};
};

}  // namespace gp_faco::cuda_detail
