#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gp_faco {

// event测同一流中的阶段区间；block周期只描述并行工作分布，不能相加当墙钟。
enum class GpuProfileStage : unsigned {
    Initialization, Features, Scoring, Action, ConstructionAndSearch,
    Reduction, Pheromone, Fingerprints, ArchiveAndFeedback, Count
};
constexpr unsigned gpu_profile_stage_count = static_cast<unsigned>(GpuProfileStage::Count);
constexpr std::array<const char*, gpu_profile_stage_count> gpu_profile_stage_names{{
    "initialization", "features_and_regions", "gp_scoring", "action_and_restart",
    "construction_and_ls", "reduction", "pheromone", "ant_fingerprints", "archive_and_feedback"
}};

struct AntPhaseCycles {
    std::uint64_t initialization = 0, construction = 0, local_search = 0, finalization = 0;
};

struct BatchPhaseProfile {
    std::uint32_t batch = 0;
    bool committed = false;
    double wall_seconds = 0, download_seconds = 0, verification_seconds = 0;
    double collection_seconds = 0;
    std::array<double, gpu_profile_stage_count> gpu_milliseconds{};
    std::vector<AntPhaseCycles> ant_cycles;
};

struct EvaluationProfile {
    double setup_seconds = 0, host_pack_seconds = 0, upload_seconds = 0;
    double initialization_gpu_milliseconds = 0, initialization_download_seconds = 0;
    std::size_t diagnostic_device_bytes = 0;
    std::vector<BatchPhaseProfile> batches;
};

struct PreparationProfile {
    double candidates_and_scales_seconds = 0, nearest_neighbor_seconds = 0;
    double initial_ls_seconds = 0, finalization_seconds = 0;
};

}  // namespace gp_faco
