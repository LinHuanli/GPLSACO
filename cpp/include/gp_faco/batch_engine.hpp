#pragma once

#include "gp_faco/deadline.hpp"
#include "gp_faco/fixed_faco_gpu.hpp"
#include "gp_faco/control_trace.hpp"
#include "gp_faco/program.hpp"

#include <memory>

namespace gp_faco {

enum class PreparationMode { CachedCharged, EndToEnd };
struct BatchTask { std::uint64_t instance_key, seed; };
struct RegistrationInfo { double cheap_seconds, preparation_seconds; };

struct BatchEvaluation {
    std::vector<TimedIncumbent> incumbents;
    double budget_seconds = 0, elapsed_seconds = 0, actual_seconds = 0, charged_seconds = 0;
    double last_batch_completed_seconds = 0, overrun_seconds = 0;
    std::uint64_t launched_batches = 0, completed_batches = 0, discarded_batches = 0;
    std::uint64_t completed_construction_steps = 0, completed_ls_evaluations = 0;
    std::size_t allocated_device_bytes = 0;
    bool preparation_completed = false;
    // 仅C++诊断入口填充，Python正式结果不暴露迟到tour或成本。
    std::vector<double> discarded_costs;
    std::vector<ControllerState> completed_control_states;
    std::vector<ControlBatchTrace> control_trace;
};

struct BatchDiagnosticControls {
    Node fixed_batches = 0;
    Node delay_batch = 0;
    unsigned completion_delay_ms = 0;
    bool capture_discarded = false;
    bool capture_control = false;
    bool force_fingerprint_collisions = false;  // 只供C++诊断检验完整邻接去重。
};

class FacoBatchEngine {
public:
    // 固定形状缓冲预分配属于通用worker准备，不处理实例或标签。
    FacoBatchEngine(Node dimension, Node colonies, FixedFacoSettings settings = {});
    ~FacoBatchEngine();
    FacoBatchEngine(const FacoBatchEngine&) = delete;
    FacoBatchEngine& operator=(const FacoBatchEngine&) = delete;
    RegistrationInfo register_problem(std::uint64_t key, std::vector<double> coordinates);
    // 实测准备耗时保持不变；指定用于评价扣费的冻结值，同一实例只能赋一次。
    void set_preparation_charges(std::uint64_t key, RegistrationInfo charges);
    BatchEvaluation evaluate(const std::vector<BatchTask>& tasks, double seconds,
                             Node mne_target, PreparationMode mode);
    BatchEvaluation evaluate_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
        Node mne_target, PreparationMode mode, BatchDiagnosticControls controls);
    BatchEvaluation evaluate_program(const std::vector<BatchTask>& tasks, double seconds,
        const Program& program, PreparationMode mode, std::uint32_t experiment_mask = UINT32_MAX);
    BatchEvaluation evaluate_program_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
        const Program& program, PreparationMode mode, std::uint32_t experiment_mask,
        BatchDiagnosticControls controls);
private:
    BatchEvaluation evaluate_impl(const std::vector<BatchTask>& tasks, double seconds,
        Node mne_target, PreparationMode mode, BatchDiagnosticControls controls,
        const Program* program, std::uint32_t experiment_mask);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gp_faco
