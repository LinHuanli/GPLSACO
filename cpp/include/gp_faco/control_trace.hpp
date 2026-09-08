#pragma once

#include "gp_faco/control_ops.hpp"
#include "gp_faco/faco_cuda_diagnostic.hpp"

namespace gp_faco {

struct ColonyObservation {
    double global_cost, epoch_cost, parent_cost, minimum, maximum, default_trail, source_uniform;
    Node iteration_best;
    bool source_is_epoch;
};

// 仅C++诊断开启时复制完整面板，不进入Python生产结果。
struct ControlDeviceSnapshot {
    std::vector<ControllerState> controls;
    std::vector<ColonyObservation> colonies;
    std::vector<Node> parent, parent_positions, epoch, global, archive, archive_positions,
        targets, tours, positions, per_ant_parent_positions, scratch, pending;
    std::vector<std::uint8_t> visited;
    std::vector<double> trails, products, gains;
    std::vector<FacoDiagnosticInfo> info;
    std::vector<TourFingerprint> ant_identities;
};

struct ControlBatchTrace {
    Node batch;
    double elapsed_ratio;
    std::vector<float> features, scores;
    std::vector<std::uint32_t> masks;
    std::vector<std::int32_t> actions;
    std::vector<Node> alternatives;
    std::vector<StartRegions> regions;
    ControlDeviceSnapshot before, after_restart, after_batch;
};

}  // namespace gp_faco
