// 固定多实例面板与截止前incumbent提交；全程无Python内层回调。
#include "gp_faco/batch_engine.hpp"
#include "gp_faco/prepared_problem.hpp"
#include "colony_state.cuh"
#include "faco_device.cuh"
#include "control_state.cuh"
#include "baseline_policy.cuh"
#include "behavior.cuh"
#include "gp_score.cuh"
#include "profile_events.cuh"

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>

namespace gp_faco {
namespace {
using Clock = std::chrono::steady_clock;
using cuda_detail::DeviceArray;
using cuda_detail::Colony;
using cuda_detail::checked;
using cuda_detail::require;

std::uint64_t mixed(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
}

struct FacoBatchEngine::Impl {
    Node n, colonies;
    FixedFacoSettings config;
    ConstraintMode constraint_mode;
    Node graph_neighbor_stride = 0;
    int device = 0;
    std::mutex mutex;
    std::map<std::uint64_t, PreparedProblem> registry;
    std::map<std::uint64_t, RegistrationInfo> preparation_charges;
    std::size_t allocated_bytes = 0;
    DeviceArray<double> xy, costs, heuristic, trails, products, gains;
    DeviceArray<Node> primary, backup, ls, initial, parent, parent_position, epoch, global,
        targets, tours, positions, parent_positions, scratch, pending;
    DeviceArray<std::uint64_t> keys;
    DeviceArray<std::uint8_t> visited;
    DeviceArray<FacoDiagnosticInfo> info;
    DeviceArray<Colony> state;
    DeviceArray<double> local_scale, epsilons;
    DeviceArray<Node> samples, archive, archive_positions, archive_scratch,
        archive_scratch_positions, alternatives;
    DeviceArray<TourFingerprint> identities;
    DeviceArray<ControllerState> controllers;
    DeviceArray<StartRegions> regions;
    DeviceArray<float> features, scores;
    DeviceArray<std::uint32_t> experiment_masks, masks;
    DeviceArray<std::int32_t> actions;
    DeviceArray<Node> graph_offsets, graph_neighbors;
    DeviceArray<EscapeCache> escape_caches;
    DeviceArray<EscapeMoveEvent> escape_moves;
    DeviceArray<Node> escape_ls_rows;
    DeviceArray<std::uint8_t> escape_anchors, escape_ls_replaced, ant_footprints,
        parent_footprints, epoch_footprints, global_footprints, archive_footprints, archive_footprint_scratch;
    std::size_t reserved_escape_bytes = 0;
    struct ProfileBuffers {
        cuda_detail::ProfileEvents events;
        DeviceArray<AntPhaseCycles> cycles;
        explicit ProfileBuffers(std::size_t ants) { cycles.allocate(ants); }
    };
    std::unique_ptr<ProfileBuffers> profiling;

    Impl(Node dimension, Node count, FixedFacoSettings settings, ConstraintMode mode)
        : n(dimension), colonies(count), config(normalized_settings(settings, dimension)),
          constraint_mode(mode) {
        require(mode == ConstraintMode::Unrestricted || mode == ConstraintMode::Hard || mode == ConstraintMode::Escape,
                "未知图约束模式");
        require(mode != ConstraintMode::Escape || (config.primary_width <= kEscapePrimaryWidth &&
            config.backup_width <= kEscapeBackupWidth && config.ls_width <= kEscapeLsWidth && config.ants < (1u << 30)),
            "Escape形状超过冻结16/64/20行宽或随机ant域容量");
        require(colonies >= 1 && colonies <= 128, "并发colony数必须在1..128");
        const std::size_t nodes = static_cast<std::size_t>(n) * colonies;
        const std::size_t ants = static_cast<std::size_t>(config.ants) * colonies;
        require(nodes * std::max({config.primary_width, config.backup_width, config.ls_width}) <=
                std::numeric_limits<Node>::max(), "候选索引超出uint32范围");
        checked(cudaGetDevice(&device));
        const auto allocate = [&](auto& buffer, std::size_t count) {
            buffer.allocate(count); allocated_bytes += buffer.bytes();
        };
        allocate(xy, nodes * 2); allocate(costs, colonies); allocate(keys, colonies);
        allocate(primary, nodes * config.primary_width); allocate(backup, nodes * config.backup_width);
        allocate(ls, nodes * config.ls_width); allocate(heuristic, nodes * config.primary_width);
        allocate(trails, nodes * config.primary_width); allocate(products, nodes * config.primary_width);
        allocate(initial, nodes); allocate(parent, nodes); allocate(parent_position, nodes);
        allocate(epoch, nodes); allocate(global, nodes); allocate(targets, ants);
        allocate(tours, ants * n); allocate(positions, ants * n); allocate(parent_positions, ants * n);
        allocate(scratch, ants * n); allocate(pending, ants * n * 5); allocate(visited, ants * n);
        allocate(gains, ants * config.ls_width * 2); allocate(info, ants); allocate(state, colonies);
        allocate(local_scale, nodes); allocate(epsilons, colonies); allocate(samples, colonies * sample_capacity);
        allocate(archive, nodes * archive_capacity); allocate(archive_positions, nodes * archive_capacity);
        allocate(archive_scratch, nodes * archive_capacity); allocate(archive_scratch_positions, nodes * archive_capacity);
        allocate(alternatives, colonies); allocate(identities, ants); allocate(controllers, colonies);
        allocate(regions, colonies); allocate(features, colonies * 12 * 32); allocate(scores, colonies * 32);
        allocate(experiment_masks, colonies); allocate(masks, colonies); allocate(actions, colonies);
        if (constraint_mode != ConstraintMode::Unrestricted) {
            const auto stride = static_cast<std::size_t>(n) * 2 * (config.primary_width + 1);
            require(stride <= UINT32_MAX, "图邻接容量超过uint32");
            graph_neighbor_stride = static_cast<Node>(stride);
            allocate(graph_offsets, static_cast<std::size_t>(colonies) * (n + 1));
            allocate(graph_neighbors, static_cast<std::size_t>(colonies) * stride);
            // Hard/Escape机制比较预留相同容量；普通主底座不承担这些缓冲。
            const auto before = allocated_bytes;
            allocate(escape_caches, ants); allocate(escape_ls_rows, ants * n * config.ls_width);
            allocate(escape_ls_replaced, ants * n * config.ls_width); allocate(escape_anchors, ants * n);
            allocate(ant_footprints, ants * n); allocate(parent_footprints, nodes);
            allocate(epoch_footprints, nodes); allocate(global_footprints, nodes);
            allocate(archive_footprints, nodes * archive_capacity);
            allocate(archive_footprint_scratch, nodes * archive_capacity);
            reserved_escape_bytes = allocated_bytes - before;
        }
    }

    EscapeFootprints escape_footprints() const {
        if (constraint_mode != ConstraintMode::Escape) return {};
        return {ant_footprints.data(), parent_footprints.data(), epoch_footprints.data(),
                global_footprints.data(), archive_footprints.data(), archive_footprint_scratch.data()};
    }

    void reset_escape() {
        escape_caches.zero(); escape_ls_rows.zero(); escape_ls_replaced.zero(); escape_anchors.zero();
        ant_footprints.zero(); parent_footprints.zero(); epoch_footprints.zero(); global_footprints.zero();
        archive_footprints.zero(); archive_footprint_scratch.zero();
        escape_moves.zero();
    }

    ControlDeviceSnapshot snapshot() const {
        ControlDeviceSnapshot result;
        result.controls = controllers.download();
        for (const auto& value : state.download()) result.colonies.push_back({value.global_cost,
            value.epoch_cost, value.parent_cost, value.minimum, value.maximum, value.default_trail,
            value.source_uniform, value.iteration_best, value.source_is_epoch});
        result.parent = parent.download(); result.parent_positions = parent_position.download();
        result.epoch = epoch.download(); result.global = global.download();
        result.archive = archive.download(); result.archive_positions = archive_positions.download();
        result.targets = targets.download(); result.tours = tours.download(); result.positions = positions.download();
        result.per_ant_parent_positions = parent_positions.download(); result.scratch = scratch.download();
        result.pending = pending.download(); result.visited = visited.download();
        result.trails = trails.download(); result.products = products.download(); result.gains = gains.download();
        result.info = info.download(); result.ant_identities = identities.download();
        if (constraint_mode == ConstraintMode::Escape) {
            result.escape_moves = escape_moves.download();
            result.escape_caches = escape_caches.download(); result.escape_ls_rows = escape_ls_rows.download();
            result.escape_ls_replaced = escape_ls_replaced.download(); result.escape_anchors = escape_anchors.download();
            result.ant_footprints = ant_footprints.download(); result.parent_footprints = parent_footprints.download();
            result.epoch_footprints = epoch_footprints.download(); result.global_footprints = global_footprints.download();
            result.archive_footprints = archive_footprints.download();
        }
        return result;
    }

    void apply_actions(bool initializing = false) {
        const auto& c = config;
        cuda_detail::apply_control_action<<<colonies, 128>>>(n, c.ants, c.primary_width, c.ls_width,
            actions.data(), alternatives.data(), archive.data(), archive_positions.data(), parent.data(),
            parent_position.data(), epoch.data(), state.data(), controllers.data(), c.retention, c.p_best,
            trails.data(), heuristic.data(), products.data(), targets.data(), tours.data(), positions.data(),
            parent_positions.data(), scratch.data(), pending.data(), visited.data(), gains.data(), info.data(),
            identities.data(), initializing, escape_footprints());
    }
};

FacoBatchEngine::FacoBatchEngine(Node n, Node colonies, FixedFacoSettings settings, ConstraintMode mode)
    : impl_(std::make_unique<Impl>(n, colonies, settings, mode)) {}
FacoBatchEngine::~FacoBatchEngine() = default;

RegistrationInfo FacoBatchEngine::register_problem(std::uint64_t key, std::vector<double> coordinates) {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine的注册与评价不能重叠");
    require(p.constraint_mode == ConstraintMode::Unrestricted, "受限实例须通过图注册入口");
    require(coordinates.size() == static_cast<std::size_t>(p.n) * 2, "实例维数与固定面板不符");
    const auto existing = p.registry.find(key);
    if (existing != p.registry.end()) {
        require(existing->second.coordinates == coordinates, "相同实例key对应了不同坐标");
        return {existing->second.cheap_seconds, existing->second.preparation_seconds};
    }
    auto prepared = make_cheap_problem(std::move(coordinates), p.config);
    prepare_problem(prepared);
    const RegistrationInfo info{prepared.cheap_seconds, prepared.preparation_seconds};
    p.registry.emplace(key, std::move(prepared));
    return info;
}

RegistrationInfo FacoBatchEngine::register_graph_problem(std::uint64_t key,
    std::vector<double> coordinates, CandidateGraphSpec spec) {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine的图注册与评价不能重叠");
    require(p.constraint_mode != ConstraintMode::Unrestricted &&
            coordinates.size() == static_cast<std::size_t>(p.n) * 2, "图注册模式/维数不符");
    const auto existing = p.registry.find(key);
    if (existing != p.registry.end()) {
        require(existing->second.coordinates == coordinates && existing->second.graph_spec &&
                *existing->second.graph_spec == spec, "相同key的坐标、初始tour或图枚举行改变");
        return {existing->second.cheap_seconds, existing->second.preparation_seconds};
    }
    auto prepared = make_cheap_problem(std::move(coordinates), p.config);
    prepare_problem(prepared);
    apply_candidate_graph(prepared, std::move(spec));
    const RegistrationInfo info{prepared.cheap_seconds, prepared.preparation_seconds};
    p.registry.emplace(key, std::move(prepared));
    return info;
}

void FacoBatchEngine::set_preparation_charges(std::uint64_t key, RegistrationInfo charges) {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine的费用登记与评价不能重叠");
    require(p.registry.find(key) != p.registry.end(), "费用对应实例尚未注册");
    require(std::isfinite(charges.cheap_seconds) && charges.cheap_seconds >= 0 &&
            std::isfinite(charges.preparation_seconds) && charges.preparation_seconds >= 0,
            "冻结准备费用必须有限且非负");
    const auto [entry, inserted] = p.preparation_charges.emplace(key, charges);
    require(inserted || (entry->second.cheap_seconds == charges.cheap_seconds &&
                        entry->second.preparation_seconds == charges.preparation_seconds),
            "同一Engine中的冻结准备费用不能改变");
}

PreparationProfile FacoBatchEngine::preparation_profile(std::uint64_t key) const {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine的准备剖析读取与评价不能重叠");
    require(p.registry.find(key) != p.registry.end(), "准备剖析对应实例尚未注册");
    return p.registry.at(key).preparation_profile;
}

BatchEvaluation FacoBatchEngine::evaluate(const std::vector<BatchTask>& tasks, double seconds,
                                        Node mne_target, PreparationMode mode) {
    return evaluate_diagnostic(tasks, seconds, mne_target, mode, {});
}

BatchEvaluation FacoBatchEngine::evaluate_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
    Node mne_target, PreparationMode mode, BatchDiagnosticControls controls) {
    require(!controls.capture_control && !controls.force_fingerprint_collisions,
            "控制轨迹和强制碰撞只能由程序评价诊断入口启用");
    return evaluate_impl(tasks, seconds, mne_target, mode, controls, nullptr, UINT32_MAX);
}

BatchEvaluation FacoBatchEngine::evaluate_program(const std::vector<BatchTask>& tasks, double seconds,
    const Program& program, PreparationMode mode, std::uint32_t experiment_mask) {
    return evaluate_impl(tasks, seconds, 2, mode, {}, &program, experiment_mask);
}

BatchEvaluation FacoBatchEngine::evaluate_program_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
    const Program& program, PreparationMode mode, std::uint32_t experiment_mask, BatchDiagnosticControls controls) {
    return evaluate_impl(tasks, seconds, 2, mode, controls, &program, experiment_mask);
}

BatchEvaluation FacoBatchEngine::evaluate_program_evaluations(const std::vector<BatchTask>& tasks,
    std::uint64_t evaluations, const Program& program, PreparationMode mode,
    std::uint32_t experiment_mask, BatchDiagnosticControls controls) {
    return evaluate_impl(tasks, 0, 2, mode, controls, &program, experiment_mask, true, evaluations);
}

BatchEvaluation FacoBatchEngine::evaluate_baseline_evaluations(const std::vector<BatchTask>& tasks,
    std::uint64_t evaluations, const BaselinePolicy& policy, PreparationMode mode,
    std::uint32_t experiment_mask, BatchDiagnosticControls controls) {
    return evaluate_impl(tasks, 0, 2, mode, controls, nullptr, experiment_mask, true, evaluations, &policy);
}

BatchEvaluation FacoBatchEngine::evaluate_factorial_evaluations(const std::vector<BatchTask>& tasks,
    std::uint64_t evaluations, const Program& program, const FactorialPolicy& policy, PreparationMode mode,
    std::uint32_t experiment_mask, BatchDiagnosticControls controls) {
    validate_factorial(policy, experiment_mask);
    validate_program(program);
    require(program.feature_spec_id == 2, "析因入口只接受评价次数特征v2");
    // M00不执行评分树，M11完全沿用Full入口；两端点没有额外的动作限制内核。
    if (policy.variant == FactorialVariant::M00)
        return evaluate_baseline_evaluations(tasks, evaluations, policy.baseline, mode, experiment_mask, controls);
    if (policy.variant == FactorialVariant::M11)
        return evaluate_program_evaluations(tasks, evaluations, program, mode, experiment_mask, controls);
    return evaluate_impl(tasks, 0, 2, mode, controls, &program, experiment_mask, true, evaluations,
                         nullptr, &policy);
}

BatchEvaluation FacoBatchEngine::evaluate_impl(const std::vector<BatchTask>& tasks, double seconds,
    Node mne_target, PreparationMode mode, BatchDiagnosticControls controls, const Program* program,
    std::uint32_t experiment_mask, bool count_limited, std::uint64_t evaluation_limit,
    const BaselinePolicy* baseline, const FactorialPolicy* factorial) {
    const auto started = Clock::now();
    auto& p = *impl_;
    const bool controlled = program || baseline;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine只允许一个活动评价");
    require(tasks.size() == p.colonies && mne_target > 0 && controls.completion_delay_ms <= 2000,
            "任务数量、MNE或诊断延迟无效");
    require(mode == PreparationMode::CachedCharged || mode == PreparationMode::EndToEnd, "未知准备模式");
    require(p.constraint_mode == ConstraintMode::Unrestricted || count_limited,
            "Hard主入口只支持评价次数，不能以旧截止路径发布图外廉价解");
    require(!count_limited || (evaluation_limit % p.config.ants == 0 &&
            evaluation_limit / p.config.ants <= UINT32_MAX && controls.fixed_batches == 0 &&
            controls.fixed_elapsed_ratio == -1),
            "次数限额须为完整蚂蚁批次，不允许另设批次或固定进度覆盖");
    require(std::isfinite(controls.fixed_elapsed_ratio) &&
            (controls.fixed_elapsed_ratio == -1 || (controls.fixed_batches > 0 &&
             controls.fixed_elapsed_ratio >= 0 && controls.fixed_elapsed_ratio <= 1)),
            "固定elapsed特征只能用于固定批次诊断且须在[0,1]");
    require(!controls.profile || count_limited || (controls.fixed_batches > 0 && controls.fixed_elapsed_ratio >= 0),
            "profile必须固定批次数与elapsed特征，不能混入主fitness");
    require(!controls.record_behavior || (count_limited && controlled),
            "轻量行为记录只允许受控的评价次数入口");
    if (program) {
        validate_program(*program);
        require(program->feature_spec_id == (count_limited ? 2 : 1),
                "程序进度特征版本与时间/评价次数入口不符");
    }
    if (baseline) {
        require(count_limited && !program, "基线需要独立的次数入口");
        validate_baseline(*baseline, experiment_mask);
    }
    if (factorial) {
        require(count_limited && program && !baseline, "析因学习需要独立的次数入口");
        validate_factorial(*factorial, experiment_mask);
    }
    if (controlled) {
        require((experiment_mask & 0xffffu) != 0, "实验mask必须保留至少一个保持模式动作");
        require(!controls.force_fingerprint_collisions || controls.capture_control,
                "强制指纹碰撞必须同时收集诊断轨迹");
    }
    int device = -1; checked(cudaGetDevice(&device));
    require(device == p.device, "Engine必须在创建它的CUDA设备上评价");
    DeviceArray<double> baseline_uniforms;
    if ((baseline || factorial) && controls.capture_control) baseline_uniforms.allocate(p.colonies);
    if (controls.capture_control && p.constraint_mode == ConstraintMode::Escape && !p.escape_moves.bytes())
        p.escape_moves.allocate(static_cast<std::size_t>(p.config.ants) * p.colonies * p.n * 2);
    const auto actual_elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - started).count(); };
    DeadlineLedger budget = count_limited ? DeadlineLedger(std::nullopt, actual_elapsed)
                                         : DeadlineLedger(seconds, actual_elapsed);
    BatchEvaluation result;
    result.budget_seconds = seconds;
    result.count_limited = count_limited;
    result.constraint_mode = p.constraint_mode;
    result.evaluation_limit_per_colony = evaluation_limit;
    result.allocated_device_bytes = p.allocated_bytes;
    result.behavior_recorded = controls.record_behavior;
    DeviceArray<BatchBehaviorRow> behavior_rows;
    DeviceArray<Node> construction_new_edges, final_new_edges, diagnostic_construction;
    if (controls.record_behavior) {
        behavior_rows.allocate(p.colonies);
        // cudaMemcpy会复制结构体padding；先完整清零，避免未初始化字节进入host诊断。
        behavior_rows.zero();
        construction_new_edges.allocate(static_cast<std::size_t>(p.config.ants) * p.colonies);
        final_new_edges.allocate(static_cast<std::size_t>(p.config.ants) * p.colonies);
        result.behavior_device_bytes = behavior_rows.bytes() + construction_new_edges.bytes() +
            final_new_edges.bytes();
        // 完整构造tour仅供C++诊断，生产记录不分配该缓冲。
        if (controls.capture_control) diagnostic_construction.allocate(
            static_cast<std::size_t>(p.n) * p.config.ants * p.colonies);
    }
    result.reserved_escape_device_bytes = p.reserved_escape_bytes;
    result.control_trace_device_bytes = p.escape_moves.bytes();
    if (controls.profile) {
        const auto setup = Clock::now();
        // 临时对象完整构造后才发布；分配失败不能留下可被后续调用误用的半成品。
        if (!p.profiling) p.profiling = std::make_unique<Impl::ProfileBuffers>(
            static_cast<std::size_t>(p.config.ants) * p.colonies);
        result.profile.setup_seconds = std::chrono::duration<double>(Clock::now() - setup).count();
        result.profile.diagnostic_device_bytes = p.profiling->cycles.bytes();
    }
    auto* events = controls.profile ? &p.profiling->events : nullptr;
    const auto event_begin = [&](GpuProfileStage stage) { if (events) events->begin(stage); };
    const auto event_end = [&](GpuProfileStage stage) { if (events) events->end(stage); };
    const auto profile_now = [&]() { return controls.profile ? Clock::now() : Clock::time_point{}; };
    const auto host_duration = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    result.incumbents.resize(p.colonies);
    if (controlled) result.completed_control_states.resize(p.colonies);
    std::map<std::uint64_t, std::vector<Node>> groups;
    for (Node colony = 0; colony < p.colonies; ++colony) {
        require(p.registry.find(tasks[colony].instance_key) != p.registry.end(), "任务实例尚未注册");
        groups[tasks[colony].instance_key].push_back(colony);
    }
    const auto finish = [&]() {
        result.actual_seconds = actual_elapsed(); result.charged_seconds = budget.charged();
        result.elapsed_seconds = result.actual_seconds + result.charged_seconds;
        result.overrun_seconds = count_limited ? 0 : std::max(0.0, result.elapsed_seconds - seconds);
        result.completed_tour_evaluations_per_colony = result.completed_batches * p.config.ants;
        result.total_tour_evaluations = result.completed_tour_evaluations_per_colony * p.colonies;
        return std::move(result);
    };
    std::map<std::uint64_t, PreparedProblem> fresh;
    const auto offer = [&](const std::vector<Node>& ids, const std::vector<Node>& tour, double cost) {
        // 先完成所有host复制，再测量共同完成时刻，避免参数求值顺序漏计复制。
        std::vector<std::vector<Node>> candidates(ids.size(), tour);
        const double completed = budget.elapsed();
        for (std::size_t i = 0; i < ids.size(); ++i)
            result.incumbents[ids[i]].offer(std::move(candidates[i]), cost, completed, budget);
    };

    // 全面板先建立廉价解，再开始昂贵准备；重复seed不重复收取同一实例CPU费用。
    for (const auto& [key, ids] : groups) {
        if (!budget.can_start()) return finish();
        const auto& cached = p.registry.at(key);
        if (mode == PreparationMode::CachedCharged) {
            const auto fixed = p.preparation_charges.find(key);
            if (!count_limited) budget.charge(fixed == p.preparation_charges.end() ? cached.cheap_seconds : fixed->second.cheap_seconds);
            offer(ids, cached.cheap_tour, cached.cheap_cost);
        } else {
            auto prepared = make_cheap_problem(cached.coordinates, p.config);
            offer(ids, prepared.cheap_tour, prepared.cheap_cost);
            fresh.emplace(key, std::move(prepared));
        }
        if (budget.expired()) return finish();
    }
    for (const auto& [key, ids] : groups) {
        if (!budget.can_start()) return finish();
        if (mode == PreparationMode::CachedCharged) {
            const auto& cached = p.registry.at(key);
            const auto fixed = p.preparation_charges.find(key);
            if (!count_limited) budget.charge(fixed == p.preparation_charges.end() ? cached.preparation_seconds : fixed->second.preparation_seconds);
            offer(ids, cached.initial_tour, cached.initial_cost);
        } else {
            auto& prepared = fresh.at(key);
            if (!prepare_problem(prepared, [&]() { return budget.expired(); })) return finish();
            if (p.constraint_mode != ConstraintMode::Unrestricted)
                apply_candidate_graph(prepared, *p.registry.at(key).graph_spec);
            offer(ids, prepared.initial_tour, prepared.initial_cost);
        }
        if (budget.expired()) return finish();
    }

    const auto pack_started = profile_now();
    std::vector<double> xy, costs, local_scale, epsilons;
    std::vector<Node> primary, backup, ls, initial, samples;
    std::vector<Node> graph_offsets, graph_neighbors;
    std::vector<std::uint64_t> keys;
    std::map<std::uint64_t, std::vector<Node>> sample_cache;
    for (const auto& task : tasks) {
        const auto& prepared = mode == PreparationMode::CachedCharged ? p.registry.at(task.instance_key)
                                                                      : fresh.at(task.instance_key);
        xy.insert(xy.end(), prepared.coordinates.begin(), prepared.coordinates.end());
        costs.push_back(prepared.initial_cost);
        initial.insert(initial.end(), prepared.initial_tour.begin(), prepared.initial_tour.end());
        for (const auto& row : prepared.primary) primary.insert(primary.end(), row.begin(), row.end());
        for (const auto& row : prepared.backup) backup.insert(backup.end(), row.begin(), row.end());
        for (const auto& row : prepared.ls) ls.insert(ls.end(), row.begin(), row.end());
        if (p.constraint_mode != ConstraintMode::Unrestricted) {
            const auto& graph = *prepared.graph;
            graph_offsets.insert(graph_offsets.end(), graph.offsets().begin(), graph.offsets().end());
            graph_neighbors.insert(graph_neighbors.end(), graph.neighbors().begin(), graph.neighbors().end());
            graph_neighbors.insert(graph_neighbors.end(), p.graph_neighbor_stride - graph.neighbors().size(), p.n);
            result.graph_edges_per_colony.push_back(graph.edges());
        }
        keys.push_back(mixed(task.seed ^ mixed(task.instance_key + 0xd1b54a32d192ed03ULL)));
        if (controlled) {
            local_scale.insert(local_scale.end(), prepared.local_scale.begin(), prepared.local_scale.end());
            epsilons.push_back(prepared.scale_epsilon);
            auto [entry, inserted] = sample_cache.try_emplace(task.instance_key, sample_capacity, 0);
            if (inserted) sample_nodes(p.n, task.instance_key, entry->second.data());
            samples.insert(samples.end(), entry->second.begin(), entry->second.end());
        }
    }
    if (controls.profile) result.profile.host_pack_seconds = host_duration(pack_started);
    if (!budget.can_start()) return finish();
    if (events) events->reset();
    event_begin(GpuProfileStage::Initialization);
    auto upload_started = profile_now();
    p.xy.upload(xy); p.costs.upload(costs); p.primary.upload(primary); p.backup.upload(backup);
    p.ls.upload(ls); p.initial.upload(initial); p.keys.upload(keys);
    if (p.constraint_mode != ConstraintMode::Unrestricted) {
        p.graph_offsets.upload(graph_offsets); p.graph_neighbors.upload(graph_neighbors);
        // 公平性需要相同容量，不要求Hard执行不会读取的例外缓冲清零。
        if (p.constraint_mode == ConstraintMode::Escape) p.reset_escape();
    }
    const auto& c = p.config;
    const Node ants = c.ants * p.colonies, cells = p.n * c.primary_width * p.colonies;
    p.targets.upload(std::vector<Node>(ants, mne_target));
    if (controls.profile) result.profile.upload_seconds += host_duration(upload_started);
    cuda_detail::reset_colony<<<p.colonies, 128>>>(p.n, p.initial.data(), p.costs.data(), p.parent.data(),
        p.epoch.data(), p.global.data(), p.state.data(), c.primary_width, c.retention, c.p_best);
    cuda_detail::initialize_products<<<(cells + 255) / 256, 256>>>(
        cuda_detail::CoordinateDistance{p.xy.data()}, cells, p.n, c.primary_width, p.primary.data(),
        c.beta, p.state.data(), p.heuristic.data(), p.trails.data(), p.products.data());
    if (controlled) {
        upload_started = profile_now();
        p.local_scale.upload(local_scale); p.epsilons.upload(epsilons); p.samples.upload(samples);
        p.experiment_masks.upload(std::vector<std::uint32_t>(p.colonies, experiment_mask));
        if (controls.profile) result.profile.upload_seconds += host_duration(upload_started);
        cuda_detail::initialize_control<<<p.colonies, 128>>>(p.n, p.initial.data(), p.costs.data(),
            p.parent_position.data(), p.archive.data(), p.archive_positions.data(), p.controllers.data(),
            controls.force_fingerprint_collisions);
        p.actions.zero(); p.apply_actions(true);
    }
    event_end(GpuProfileStage::Initialization);
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    if (events) result.profile.initialization_gpu_milliseconds =
        events->elapsed()[static_cast<unsigned>(GpuProfileStage::Initialization)];
    const auto initial_download_started = profile_now();
    const auto initial_control = controlled ? p.controllers.download() : std::vector<ControllerState>{};
    if (controls.profile) result.profile.initialization_download_seconds = host_duration(initial_download_started);
    if (budget.expired()) return finish();
    result.preparation_completed = true;
    if (controlled) result.completed_control_states = initial_control;

    for (Node batch = 0; budget.can_start(); ++batch) {
        if (count_limited && result.completed_batches == evaluation_limit / p.config.ants) break;
        if (controls.fixed_batches && batch >= controls.fixed_batches) break;
        ++result.launched_batches;
        const auto batch_started = profile_now();
        BatchPhaseProfile profile; profile.batch = batch;
        if (events) events->reset();
        ControlBatchTrace trace;
        if (controlled) {
            trace.batch = batch;
            if (controls.capture_control) trace.before = p.snapshot();
            // feature_spec=2的进度只依赖已完成FE；计时插桩和主机调度不改变动作输入。
            trace.elapsed_ratio = count_limited ?
                static_cast<double>(result.completed_batches * c.ants) / evaluation_limit :
                (controls.fixed_elapsed_ratio >= 0 ? controls.fixed_elapsed_ratio : budget.elapsed() / seconds);
            event_begin(GpuProfileStage::Features);
            cuda_detail::build_control_features<<<p.colonies, 128>>>(p.n, c.primary_width, p.colonies,
                p.xy.data(), p.primary.data(), p.local_scale.data(), p.epsilons.data(), p.samples.data(),
                p.keys.data(), batch, trace.elapsed_ratio, p.parent.data(), p.parent_position.data(),
                p.archive.data(), p.archive_positions.data(), p.controllers.data(), p.state.data(),
                p.trails.data(), p.experiment_masks.data(), p.regions.data(), p.alternatives.data(),
                p.masks.data(), p.features.data());
            event_end(GpuProfileStage::Features);
            if (controls.record_behavior) cuda_detail::begin_behavior<<<(p.colonies + 127) / 128, 128>>>(
                p.n, c.ants, p.colonies, batch, p.controllers.data(), p.state.data(), p.alternatives.data(),
                p.masks.data(), p.keys.data(), baseline || factorial,
                baseline ? *baseline : factorial ? factorial->baseline : BaselinePolicy{}, behavior_rows.data());
            if (factorial && controls.capture_control) trace.legal_masks = p.masks.download();
            event_begin(GpuProfileStage::Scoring);
            if (factorial) {
                cuda_detail::restrict_factorial_actions<<<(p.colonies + 127) / 128, 128>>>(*factorial,
                    p.controllers.data(), p.keys.data(), batch, p.masks.data(), p.colonies,
                    baseline_uniforms.data());
            }
            if (program) {
                cuda_detail::score_actions<<<(p.colonies + 3) / 4, 128>>>(*program, p.features.data(),
                    p.masks.data(), p.scores.data(), p.actions.data(), p.colonies);
            } else {
                cuda_detail::select_baseline_actions<<<(p.colonies + 127) / 128, 128>>>(*baseline,
                    p.controllers.data(), p.keys.data(), batch, p.masks.data(), p.actions.data(), p.colonies,
                    baseline_uniforms.data());
            }
            event_end(GpuProfileStage::Scoring);
            if (controls.capture_control) {
                trace.features = p.features.download();
                if (program) trace.scores = p.scores.download();
                if (baseline || factorial) trace.baseline_uniforms = baseline_uniforms.download();
                trace.masks = p.masks.download(); trace.actions = p.actions.download();
                trace.alternatives = p.alternatives.download(); trace.regions = p.regions.download();
            }
            event_begin(GpuProfileStage::Action);
            if (controls.record_behavior) cuda_detail::observe_behavior_action<<<(p.colonies + 127) / 128, 128>>>(
                p.colonies, p.controllers.data(), p.state.data(), p.actions.data(), p.masks.data(),
                behavior_rows.data());
            p.apply_actions();
            event_end(GpuProfileStage::Action);
            if (controls.capture_control) trace.after_restart = p.snapshot();
        }
        const cuda_detail::BatchCoordinateDistance distance{p.xy.data(), p.n, c.ants};
        const cuda_detail::BatchStochasticChoices choices{p.primary.data(), p.backup.data(), p.products.data(),
            p.keys.data(), c.primary_width, c.backup_width, batch, c.ants};
        const auto construct = [&](auto view, auto allowed) {
            if (controls.profile) {
                cuda_detail::construct_and_search<decltype(distance), decltype(view), decltype(allowed), true>
                    <<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                    p.gains.data(), diagnostic_construction.data(), p.info.data(), p.visited.data(), allowed,
                    p.profiling->cycles.data(), construction_new_edges.data());
            } else {
                cuda_detail::construct_and_search<<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                    p.gains.data(), diagnostic_construction.data(), p.info.data(), p.visited.data(), allowed,
                    nullptr, construction_new_edges.data());
            }
        };
        event_begin(GpuProfileStage::ConstructionAndSearch);
        const auto with_constraint = [&](auto allowed) {
            if (controlled) construct(cuda_detail::BatchRegionChoices{choices, p.regions.data(), p.actions.data()}, allowed);
            else construct(choices, allowed);
        };
        if (p.constraint_mode == ConstraintMode::Escape) {
            const cuda_detail::BatchEscapeChoices escape_choices{choices,
                controlled ? p.regions.data() : nullptr, controlled ? p.actions.data() : nullptr,
                p.trails.data(), p.state.data(), c.beta, p.escape_anchors.data(), p.ant_footprints.data(),
                p.escape_ls_replaced.data(), p.parent_footprints.data(), p.escape_ls_rows.data(), p.info.data(),
                c.ls_width, controls.disable_escape, controls.capture_control ? p.escape_moves.data() : nullptr};
            construct(escape_choices, cuda_detail::BatchEscapeEdges{
                {p.graph_offsets.data(), p.graph_neighbors.data(), c.ants, p.graph_neighbor_stride}, p.escape_caches.data()});
        } else if (p.constraint_mode == ConstraintMode::Hard)
            with_constraint(BatchSparseGraphView{p.graph_offsets.data(), p.graph_neighbors.data(),
                                                c.ants, p.graph_neighbor_stride});
        else with_constraint(UnrestrictedEdges{});
        event_end(GpuProfileStage::ConstructionAndSearch);
        event_begin(GpuProfileStage::Reduction);
        cuda_detail::reduce_and_select<<<p.colonies, 128>>>(p.n, c.ants, p.tours.data(), p.info.data(),
            p.parent.data(), p.parent_position.data(), p.epoch.data(), p.global.data(), p.state.data(),
            c.primary_width, c.retention, c.p_best, c.epoch_source_probability, p.keys.data(), batch,
            p.escape_footprints());
        event_end(GpuProfileStage::Reduction);
        event_begin(GpuProfileStage::Pheromone);
        cuda_detail::update_pheromone<<<(cells + 255) / 256, 256>>>(p.n, c.primary_width, p.colonies,
            p.primary.data(), p.parent.data(), p.parent_position.data(), c.retention, p.state.data(),
            p.heuristic.data(), p.trails.data(), p.products.data());
        event_end(GpuProfileStage::Pheromone);
        if (controlled) {
            event_begin(GpuProfileStage::Fingerprints);
            cuda_detail::fingerprint_ants<<<ants, 128>>>(p.n, p.tours.data(), p.identities.data(),
                controls.force_fingerprint_collisions);
            event_end(GpuProfileStage::Fingerprints);
            if (controls.record_behavior) {
                cuda_detail::observe_terminal_edges<<<ants, 128>>>(p.n, p.tours.data(),
                    p.parent_positions.data(), final_new_edges.data());
                cuda_detail::finish_behavior<<<(p.colonies + 127) / 128, 128>>>(c.ants, p.colonies,
                    p.controllers.data(), p.state.data(), p.info.data(), p.identities.data(),
                    construction_new_edges.data(), final_new_edges.data(), behavior_rows.data());
            }
            event_begin(GpuProfileStage::ArchiveAndFeedback);
            cuda_detail::update_control<<<p.colonies, 128>>>(p.n, c.ants, p.tours.data(), p.positions.data(),
                p.info.data(), p.identities.data(), p.state.data(), p.archive.data(), p.archive_positions.data(),
                p.archive_scratch.data(), p.archive_scratch_positions.data(), p.controllers.data(), c.ls_evaluation_limit,
                p.escape_footprints());
            event_end(GpuProfileStage::ArchiveAndFeedback);
        }
        checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
        const auto download_started = profile_now();
        const auto states = p.state.download();
        const auto tours = p.global.download();
        const auto info = p.info.download();
        const auto controller_state = controlled ? p.controllers.download() : std::vector<ControllerState>{};
        auto observations = controls.record_behavior ? behavior_rows.download() : std::vector<BatchBehaviorRow>{};
        for (auto& row : observations) row.feedback_after = controller_state[row.colony].feedback;
        if (controls.capture_control && controls.record_behavior)
            trace.construction_tours = diagnostic_construction.download();
        if (controls.capture_control) trace.after_batch = p.snapshot();
        if (controls.profile) profile.download_seconds = host_duration(download_started);
        const auto verification_started = profile_now();
        std::vector<std::vector<Node>> candidates;
        std::vector<double> candidate_costs;
        for (Node colony = 0; colony < p.colonies; ++colony) {
            const auto base = static_cast<std::size_t>(colony) * p.n;
            std::vector<Node> candidate(tours.begin() + base, tours.begin() + base + p.n);
            const auto& problem = p.registry.at(tasks[colony].instance_key);
            require(p.constraint_mode != ConstraintMode::Hard || problem.graph->contains_tour(candidate),
                    "Hard设备incumbent存在图外边");
            // 同步完成后验证完整排列/成本；验证耗时也在完成时间戳之前。
            CpuTour verified(candidate, [&](Node a, Node b) { return problem.distance(a, b); });
            const double value = verified.cost();
            require(std::isfinite(states[colony].global_cost) &&
                    std::abs(value - states[colony].global_cost) <= 1e-8 + 1e-12 * value,
                    "设备incumbent成本与完整重算不符");
            candidates.push_back(std::move(candidate)); candidate_costs.push_back(value);
        }
        if (controls.profile) {
            profile.verification_seconds = host_duration(verification_started);
            const auto collection_started = Clock::now();
            profile.gpu_milliseconds = events->elapsed();
            profile.ant_cycles = p.profiling->cycles.download();
            profile.collection_seconds = host_duration(collection_started);
            profile.wall_seconds = host_duration(batch_started);
        }
        if (controls.completion_delay_ms && batch == controls.delay_batch)
            std::this_thread::sleep_for(std::chrono::milliseconds(controls.completion_delay_ms));
        const double completed = budget.elapsed();
        result.last_batch_completed_seconds = completed;
        if (!budget.completed_on_time(completed)) {
            ++result.discarded_batches;
            if (controls.profile) result.profile.batches.push_back(std::move(profile));
            if (controls.capture_discarded) result.discarded_costs = candidate_costs;
            break;
        }
        for (Node colony = 0; colony < p.colonies; ++colony)
            result.incumbents[colony].offer(std::move(candidates[colony]), candidate_costs[colony], completed, budget);
        ++result.completed_batches;
        if (controls.profile) { profile.committed = true; result.profile.batches.push_back(std::move(profile)); }
        if (controlled) result.completed_control_states = controller_state;
        result.behavior_rows.insert(result.behavior_rows.end(), observations.begin(), observations.end());
        if (controls.capture_control) result.control_trace.push_back(std::move(trace));
        for (const auto& ant : info) {
            result.completed_construction_steps += ant.construction.steps;
            result.completed_ls_evaluations += ant.local_search.move_evaluations;
            result.completed_constraint_rejections += ant.local_search.constraint_rejections;
            result.escape.construction_opportunities += ant.escape.construction_opportunities;
            result.escape.construction_gates += ant.escape.construction_gates;
            result.escape.construction_replaced_slots += ant.escape.construction_replaced_slots;
            result.escape.escape_relocations += ant.escape.escape_relocations;
            result.escape.ls_anchor_nodes += ant.escape.ls_anchor_nodes;
            result.escape.ls_replaced_slots += ant.escape.ls_replaced_slots;
            result.escape.old_view_reactivations += ant.escape.old_view_reactivations;
            result.escape.anchor_reactivations += ant.escape.anchor_reactivations;
            result.escape.new_edges += ant.escape.new_edges;
            result.escape.construction_capacity_rejections += ant.escape.construction_capacity_rejections;
            result.escape.ls_capacity_rejections += ant.escape.ls_capacity_rejections;
        }
        if (batch == std::numeric_limits<Node>::max()) throw std::runtime_error("随机批次编号用尽");
    }
    return finish();
}

}  // namespace gp_faco
