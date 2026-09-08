#pragma once

#include "gp_faco/deadline.hpp"
#include "gp_faco/fixed_faco_gpu.hpp"
#include "gp_faco/control_trace.hpp"
#include "gp_faco/program.hpp"
#include "gp_faco/profiling.hpp"
#include "gp_faco/baseline_policy.hpp"
#include "gp_faco/prepared_problem.hpp"

#include <memory>

namespace gp_faco {

enum class PreparationMode { CachedCharged, EndToEnd };
enum class ConstraintMode { Unrestricted, Hard, Escape };
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
    bool count_limited = false;
    std::uint64_t evaluation_limit_per_colony = 0, completed_tour_evaluations_per_colony = 0;
    std::uint64_t total_tour_evaluations = 0;
    ConstraintMode constraint_mode = ConstraintMode::Unrestricted;
    std::vector<std::uint64_t> graph_edges_per_colony;
    std::uint64_t completed_constraint_rejections = 0;
    EscapeStats escape;
    std::size_t reserved_escape_device_bytes = 0;
    std::size_t control_trace_device_bytes = 0;
    // 仅C++诊断入口填充，Python正式结果不暴露迟到tour或成本。
    std::vector<double> discarded_costs;
    std::vector<ControllerState> completed_control_states;
    std::vector<ControlBatchTrace> control_trace;
    EvaluationProfile profile;  // 仅显式诊断返回，生产Python输出不暴露。
};

struct BatchDiagnosticControls {
    Node fixed_batches = 0;
    Node delay_batch = 0;
    unsigned completion_delay_ms = 0;
    bool capture_discarded = false;
    bool capture_control = false;
    bool force_fingerprint_collisions = false;  // 只供C++诊断检验完整邻接去重。
    bool profile = false;
    double fixed_elapsed_ratio = -1;  // 固定批次对照专用；负一表示正常wall-clock特征。
    bool disable_escape = false;  // 仅C++诊断：完全关闭替换，核对Hard逐批一致性。
};

class FacoBatchEngine {
public:
    // 固定形状缓冲预分配属于通用worker准备，不处理实例或标签。
    FacoBatchEngine(Node dimension, Node colonies, FixedFacoSettings settings = {},
                   ConstraintMode constraint_mode = ConstraintMode::Unrestricted);
    ~FacoBatchEngine();
    FacoBatchEngine(const FacoBatchEngine&) = delete;
    FacoBatchEngine& operator=(const FacoBatchEngine&) = delete;
    RegistrationInfo register_problem(std::uint64_t key, std::vector<double> coordinates);
    RegistrationInfo register_graph_problem(std::uint64_t key, std::vector<double> coordinates,
                                            CandidateGraphSpec spec);
    // 实测准备耗时保持不变；指定用于评价扣费的冻结值，同一实例只能赋一次。
    void set_preparation_charges(std::uint64_t key, RegistrationInfo charges);
    PreparationProfile preparation_profile(std::uint64_t key) const;
    BatchEvaluation evaluate(const std::vector<BatchTask>& tasks, double seconds,
                             Node mne_target, PreparationMode mode);
    BatchEvaluation evaluate_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
        Node mne_target, PreparationMode mode, BatchDiagnosticControls controls);
    BatchEvaluation evaluate_program(const std::vector<BatchTask>& tasks, double seconds,
        const Program& program, PreparationMode mode, std::uint32_t experiment_mask = UINT32_MAX);
    BatchEvaluation evaluate_program_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
        const Program& program, PreparationMode mode, std::uint32_t experiment_mask,
        BatchDiagnosticControls controls);
    // 主次数入口：每个colony的蚂蚁完整tour计数，不设置wall-clock截止。
    BatchEvaluation evaluate_program_evaluations(const std::vector<BatchTask>& tasks,
        std::uint64_t evaluation_limit_per_colony, const Program& program, PreparationMode mode,
        std::uint32_t experiment_mask = UINT32_MAX, BatchDiagnosticControls controls = {});
    BatchEvaluation evaluate_baseline_evaluations(const std::vector<BatchTask>& tasks,
        std::uint64_t evaluation_limit_per_colony, const BaselinePolicy& policy, PreparationMode mode,
        std::uint32_t experiment_mask = UINT32_MAX, BatchDiagnosticControls controls = {});
private:
    BatchEvaluation evaluate_impl(const std::vector<BatchTask>& tasks, double seconds,
        Node mne_target, PreparationMode mode, BatchDiagnosticControls controls,
        const Program* program, std::uint32_t experiment_mask,
        bool count_limited = false, std::uint64_t evaluation_limit_per_colony = 0,
        const BaselinePolicy* baseline = nullptr);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gp_faco
