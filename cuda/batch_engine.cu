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
#include <cstring>
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
    Node n, colonies, geometries, population_size;
    FixedFacoSettings config;
    ConstraintMode constraint_mode;
    Node graph_neighbor_stride = 0;
    int device = 0;
    std::mutex mutex;
    std::map<std::uint64_t, PreparedProblem> registry;
    std::vector<std::uint64_t> resident_instances;
    bool resident_control = false;
    std::map<std::uint64_t, RegistrationInfo> preparation_charges;
    std::size_t allocated_bytes = 0;
    DeviceArray<double> xy, costs, heuristic, trails, products, gains, ls_distances, parent_costs, pair_distances;
    DeviceArray<Node> primary, backup, ls, initial, parent, parent_position, epoch, global,
        targets, tours, positions, parent_positions, scratch, pending;
    DeviceArray<std::uint64_t> keys;
    DeviceArray<std::uint8_t> visited, queued;
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
    DeviceArray<Program> population_programs;
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

    Impl(Node dimension, Node count, FixedFacoSettings settings, ConstraintMode mode, Node population)
        : n(dimension), colonies(count * population), geometries(count), population_size(population),
          config(normalized_settings(settings, dimension)),
          constraint_mode(mode) {
        require(mode == ConstraintMode::Unrestricted || mode == ConstraintMode::Hard || mode == ConstraintMode::Escape,
                "未知图约束模式");
        require(mode != ConstraintMode::Escape || (config.primary_width <= kEscapePrimaryWidth &&
            config.backup_width <= kEscapeBackupWidth && config.ls_width <= kEscapeLsWidth && config.ants < (1u << 30)),
            "Escape形状超过冻结16/64/20行宽或随机ant域容量");
        require(count >= 1 && count <= 128 && population >= 1 && population <= 128,
                "面板和种群大小分别必须在1..128");
        require(population == 1 || (mode == ConstraintMode::Unrestricted && n <= 1500 && config.ants % 4 == 0),
                "种群展开支持 n<=1500、完整 warp 蚂蚁组的普通 FACO");
        const bool compact = population > 1;
        const std::size_t nodes = static_cast<std::size_t>(n) * colonies;
        const std::size_t geometry_nodes = static_cast<std::size_t>(n) * geometries;
        const std::size_t ants = static_cast<std::size_t>(config.ants) * colonies;
        require(nodes * std::max({config.primary_width, config.backup_width, config.ls_width}) <=
                std::numeric_limits<Node>::max(), "候选索引超出uint32范围");
        checked(cudaGetDevice(&device));
        const auto allocate = [&](auto& buffer, std::size_t count) {
            buffer.allocate(count); allocated_bytes += buffer.bytes();
        };
        allocate(xy, geometry_nodes * 2); allocate(costs, geometries); allocate(keys, colonies);
        allocate(parent_costs, colonies);
        allocate(pair_distances, n <= 1500 ? geometry_nodes * n : 0);
        allocate(primary, geometry_nodes * config.primary_width); allocate(backup, geometry_nodes * config.backup_width);
        allocate(ls, geometry_nodes * config.ls_width); allocate(ls_distances, geometry_nodes * config.ls_width);
        allocate(heuristic, geometry_nodes * config.primary_width);
        allocate(trails, nodes * config.primary_width); allocate(products, nodes * config.primary_width);
        allocate(initial, geometry_nodes); allocate(parent, nodes); allocate(parent_position, nodes);
        allocate(epoch, nodes); allocate(global, nodes); allocate(targets, ants);
        allocate(tours, ants * n); allocate(positions, ants * n); allocate(parent_positions, compact ? 0 : ants * n);
        allocate(scratch, compact ? 0 : ants * n); allocate(pending, ants * n * (compact ? 1 : 5));
        allocate(visited, ants * n); allocate(queued, ants * n);
        allocate(gains, ants * config.ls_width * 2); allocate(info, ants); allocate(state, colonies);
        allocate(local_scale, geometry_nodes); allocate(epsilons, geometries); allocate(samples, geometries * sample_capacity);
        allocate(archive, nodes * archive_capacity); allocate(archive_positions, nodes * archive_capacity);
        allocate(archive_scratch, nodes * archive_capacity); allocate(archive_scratch_positions, nodes * archive_capacity);
        allocate(alternatives, colonies); allocate(identities, ants); allocate(controllers, colonies);
        allocate(regions, colonies); allocate(features, colonies * 12 * 32); allocate(scores, colonies * 32);
        allocate(experiment_masks, colonies); allocate(masks, colonies); allocate(actions, colonies);
        allocate(population_programs, compact ? population : 0);
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

    template<class Visitor> void visit_buffers(Visitor visit) {
        // 唯一持久算法缓冲清单；features按特征分面，其余均按colony连续排列。
#define STATE_BUFFER(name) visit(#name, name, 1)
        STATE_BUFFER(xy); STATE_BUFFER(costs); STATE_BUFFER(heuristic); STATE_BUFFER(trails);
        STATE_BUFFER(products); STATE_BUFFER(gains); STATE_BUFFER(primary); STATE_BUFFER(backup);
        STATE_BUFFER(ls); STATE_BUFFER(initial); STATE_BUFFER(parent); STATE_BUFFER(parent_position);
        STATE_BUFFER(epoch); STATE_BUFFER(global); STATE_BUFFER(targets); STATE_BUFFER(tours);
        STATE_BUFFER(positions); STATE_BUFFER(parent_positions); STATE_BUFFER(scratch); STATE_BUFFER(pending);
        STATE_BUFFER(keys); STATE_BUFFER(visited); STATE_BUFFER(queued); STATE_BUFFER(ls_distances); STATE_BUFFER(parent_costs); STATE_BUFFER(pair_distances); STATE_BUFFER(info); STATE_BUFFER(state);
        STATE_BUFFER(local_scale); STATE_BUFFER(epsilons); STATE_BUFFER(samples); STATE_BUFFER(archive);
        STATE_BUFFER(archive_positions); STATE_BUFFER(archive_scratch); STATE_BUFFER(archive_scratch_positions);
        STATE_BUFFER(alternatives); STATE_BUFFER(identities); STATE_BUFFER(controllers); STATE_BUFFER(regions);
        visit("features", features, 12);
        STATE_BUFFER(scores); STATE_BUFFER(experiment_masks); STATE_BUFFER(masks); STATE_BUFFER(actions);
#undef STATE_BUFFER
    }

    void build_features(Node batch, double progress, std::uint32_t regional_mask = 15,
                        const std::uint32_t* program_masks = nullptr) {
        cuda_detail::build_control_features<<<colonies, 128>>>(n, config.primary_width, colonies,
            xy.data(), primary.data(), local_scale.data(), epsilons.data(), samples.data(),
            keys.data(), batch, progress, parent.data(), parent_position.data(), archive.data(),
            archive_positions.data(), controllers.data(), state.data(), trails.data(),
            experiment_masks.data(), regions.data(), alternatives.data(), masks.data(), features.data(),
            population_size > 1 ? geometries : 0, regional_mask, program_masks);
    }

    CountedState capture(const std::vector<BatchTask>& tasks, std::uint64_t next_batch,
                         std::uint64_t progress_limit, std::uint32_t experiment_mask,
                         const std::vector<TimedIncumbent>& incumbents) {
        CountedState out; out.dimension = n; out.colonies = colonies; out.settings = config;
        out.tasks = tasks; out.completed_batches = next_batch; out.progress_evaluation_limit = progress_limit;
        out.experiment_mask = experiment_mask;
        for (const auto& item : incumbents) {
            require(item.present, "捕获时缺少已提交主机incumbent");
            out.incumbents.push_back({item.tour, item.cost});
        }
        // 在源随机key下固定下一决策的区域；该内核只写决策视图，不推进算法状态。
        build_features(static_cast<Node>(next_batch), static_cast<double>(next_batch * config.ants) / progress_limit);
        checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
        std::size_t bytes = 0;
        visit_buffers([&](const char* name, auto& buffer, std::uint64_t planes) {
            CountedStateBuffer saved; saved.name = name; saved.planes = planes;
            saved.element_bytes = sizeof(*buffer.data()); saved.bytes.resize(buffer.bytes());
            checked(cudaMemcpy(saved.bytes.data(), buffer.data(), buffer.bytes(), cudaMemcpyDeviceToHost));
            bytes += buffer.bytes(); out.buffers.push_back(std::move(saved));
        });
        require(bytes == allocated_bytes, "快照未覆盖全部持久算法缓冲");
        out.seal(); return out;
    }

    void restore(const CountedState& saved) {
        saved.validate();
        require(saved.dimension == n && saved.colonies == colonies && same_settings(saved.settings, config),
                "快照形状或求解器配置与Engine不符");
        std::size_t index = 0, bytes = 0;
        // 先完整验证，再上传；结构错误不能留下半份恢复的GPU状态。
        visit_buffers([&](const char* name, auto& buffer, std::uint64_t planes) {
            require(index < saved.buffers.size(), "快照缺少持久缓冲");
            const auto& source = saved.buffers[index++];
            require(source.name == name && source.planes == planes &&
                    source.element_bytes == sizeof(*buffer.data()) && source.bytes.size() == buffer.bytes(),
                    "快照缓冲名称、布局或ABI与Engine不符");
            bytes += buffer.bytes();
        });
        require(index == saved.buffers.size() && bytes == allocated_bytes, "快照缓冲集合不完整或多余");
        std::vector<double> coordinates;
        for (const auto& task : saved.tasks) {
            require(registry.find(task.instance_key) != registry.end(), "快照实例尚未注册");
            const auto& xy = registry.at(task.instance_key).coordinates;
            coordinates.insert(coordinates.end(), xy.begin(), xy.end());
        }
        require(saved.buffers.front().name == "xy" &&
            saved.buffers.front().bytes.size() == coordinates.size() * sizeof(double) &&
            std::memcmp(saved.buffers.front().bytes.data(), coordinates.data(), coordinates.size() * sizeof(double)) == 0,
            "快照实际坐标与已注册实例不同");
        index = 0;
        visit_buffers([&](const char*, auto& buffer, std::uint64_t) {
            const auto& source = saved.buffers[index++];
            checked(cudaMemcpy(buffer.data(), source.bytes.data(), buffer.bytes(), cudaMemcpyHostToDevice));
        });
    }

    void apply_actions(bool initializing = false) {
        const auto& c = config;
        cuda_detail::apply_control_action<<<colonies, 128>>>(n, c.ants, c.primary_width, c.ls_width,
            actions.data(), alternatives.data(), archive.data(), archive_positions.data(), parent.data(),
            parent_position.data(), epoch.data(), state.data(), controllers.data(), c.retention, c.p_best,
            trails.data(), heuristic.data(), products.data(), targets.data(), tours.data(), positions.data(),
            parent_positions.data(), scratch.data(), pending.data(), visited.data(), gains.data(), info.data(),
            identities.data(), initializing, escape_footprints(), population_size > 1 ? geometries : 0,
            population_size > 1);
    }
};

FacoBatchEngine::FacoBatchEngine(Node n, Node colonies, FixedFacoSettings settings, ConstraintMode mode, Node population)
    : impl_(std::make_unique<Impl>(n, colonies, settings, mode, population)) {}
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

BatchEvaluation FacoBatchEngine::evaluate_faco_evaluations(const std::vector<BatchTask>& tasks,
    std::uint64_t evaluations, Node mne_target, PreparationMode mode, BatchDiagnosticControls controls) {
    return evaluate_impl(tasks, 0, mne_target, mode, controls, nullptr, UINT32_MAX, true, evaluations);
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

std::vector<BatchEvaluation> FacoBatchEngine::evaluate_population_evaluations(
    const std::vector<BatchTask>& panel, std::uint64_t evaluations, const std::vector<Program>& programs,
    std::uint32_t mask, const FactorialPolicy* factorial, BatchDiagnosticControls controls,
    const std::vector<ControllerProgram>* controllers) {
    auto& p = *impl_;
    require(!programs.empty() && programs.size() <= p.population_size && panel.size() == p.geometries,
            "种群程序数或共用面板形状不符");
    if (factorial) require(factorial->variant == FactorialVariant::M10 ||
        factorial->variant == FactorialVariant::M01, "种群析因入口只接受 M10/M01");
    if (controllers) {
        require(controllers->size() == programs.size() && !factorial, "控制器种群形状或析因参数不符");
        for (const auto& c : *controllers) validate_controller(c);
    }
    if (p.population_size == 1 && controllers)
        return {evaluate_controller_evaluations(panel, evaluations, controllers->front(), mask, controls)};
    if (p.population_size == 1) {
        auto result = factorial ? evaluate_factorial_evaluations(panel, evaluations, programs.front(), *factorial,
            PreparationMode::CachedCharged, mask, controls) : evaluate_program_evaluations(panel, evaluations,
            programs.front(), PreparationMode::CachedCharged, mask, controls);
        return {std::move(result)};
    }
    std::vector<BatchTask> tasks;
    const Node count = static_cast<Node>(programs.size());
    tasks.reserve(p.geometries * count);
    for (Node i = 0; i < count; ++i) tasks.insert(tasks.end(), panel.begin(), panel.end());
    auto batch = evaluate_impl(tasks, 0, 2, PreparationMode::CachedCharged, controls, &programs.front(), mask,
        true, evaluations, nullptr, nullptr, 0, nullptr, {}, factorial, &programs, controllers);
    std::vector<BatchEvaluation> output;
    output.reserve(count);
    for (Node individual = 0; individual < count; ++individual) {
        BatchEvaluation item;
        item.population_size = count;
        item.population_actual_seconds = batch.actual_seconds;
        // 与 population_actual_seconds 一样表示整个批次，不能跨 individual 求和。
        item.production_kernel_milliseconds = batch.production_kernel_milliseconds;
        // 逐个体成本等额分摊实测批次时间，保留整批时间，避免总账把并行成本重复计 128 次。
        item.actual_seconds = item.elapsed_seconds = batch.actual_seconds / count;
        item.last_batch_completed_seconds = batch.last_batch_completed_seconds / count;
        item.allocated_device_bytes = batch.allocated_device_bytes;
        item.preparation_completed = batch.preparation_completed;
        item.count_limited = true;
        item.evaluation_limit_per_colony = evaluations;
        item.completed_tour_evaluations_per_colony = batch.completed_tour_evaluations_per_colony;
        item.total_tour_evaluations = item.completed_tour_evaluations_per_colony * p.geometries;
        item.launched_batches = batch.launched_batches;
        item.completed_batches = batch.completed_batches;
        for (Node replica = 0; replica < p.geometries; ++replica) {
            const auto index = individual * p.geometries + replica;
            item.incumbents.push_back(std::move(batch.incumbents[index]));
            item.incumbents.back().completed_seconds /= count;
            item.completed_control_states.push_back(batch.completed_control_states[index]);
            item.completed_construction_steps += batch.colony_work[index][0];
            item.completed_ls_evaluations += batch.colony_work[index][1];
            item.completed_constraint_rejections += batch.colony_work[index][2];
        }
        output.push_back(std::move(item));
    }
    return output;
}

BatchEvaluation FacoBatchEngine::evaluate_controller_evaluations(const std::vector<BatchTask>& tasks,
    std::uint64_t evaluations, const ControllerProgram& controller, std::uint32_t mask,
    BatchDiagnosticControls controls) {
    validate_controller(controller);
    if (!controller.kind) return evaluate_program_evaluations(tasks, evaluations, controller.trees[0],
        PreparationMode::CachedCharged, mask, controls);
    const std::vector<ControllerProgram> controllers{controller};
    return evaluate_impl(tasks, 0, 2, PreparationMode::CachedCharged, controls, &controller.trees[0],
        mask, true, evaluations, nullptr, nullptr, 0, nullptr, {}, nullptr, nullptr, &controllers);
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
                         nullptr, nullptr, 0, nullptr, {}, &policy);
}

BatchEvaluation FacoBatchEngine::capture_program_state(const std::vector<BatchTask>& tasks,
    std::uint64_t source_evaluations, std::uint64_t capture_after, const Program& program,
    CountedState& output, std::uint32_t mask, BatchDiagnosticControls controls) {
    return evaluate_impl(tasks, 0, 2, PreparationMode::CachedCharged, controls, &program, mask,
        true, source_evaluations, nullptr, &output, capture_after);
}

BatchEvaluation FacoBatchEngine::capture_baseline_state(const std::vector<BatchTask>& tasks,
    std::uint64_t source_evaluations, std::uint64_t capture_after, const BaselinePolicy& policy,
    CountedState& output, std::uint32_t mask, BatchDiagnosticControls controls) {
    return evaluate_impl(tasks, 0, 2, PreparationMode::CachedCharged, controls, nullptr, mask,
        true, source_evaluations, &policy, &output, capture_after);
}

BatchEvaluation FacoBatchEngine::continue_program_state(const CountedState& state,
    std::uint64_t additional, const Program& program, BatchDiagnosticControls controls) {
    return evaluate_impl(state.tasks, 0, 2, PreparationMode::CachedCharged, controls, &program,
        state.experiment_mask, true, additional, nullptr, nullptr, 0, &state);
}

BatchEvaluation FacoBatchEngine::continue_baseline_state(const CountedState& state,
    std::uint64_t additional, const BaselinePolicy& policy, ForkIntervention intervention,
    BatchDiagnosticControls controls) {
    return evaluate_impl(state.tasks, 0, 2, PreparationMode::CachedCharged, controls, nullptr,
        state.experiment_mask, true, additional, &policy, nullptr, 0, &state, intervention);
}

BatchEvaluation FacoBatchEngine::evaluate_impl(const std::vector<BatchTask>& tasks, double seconds,
    Node mne_target, PreparationMode mode, BatchDiagnosticControls controls, const Program* program,
    std::uint32_t experiment_mask, bool count_limited, std::uint64_t evaluation_limit,
    const BaselinePolicy* baseline, CountedState* snapshot_output, std::uint64_t capture_after,
    const CountedState* resumed_state, ForkIntervention intervention, const FactorialPolicy* factorial,
    const std::vector<Program>* population, const std::vector<ControllerProgram>* controllers) {
    const auto started = Clock::now();
    auto& p = *impl_;
    const bool controlled = program || baseline;
    const bool final_only = count_limited && !controls.capture_control && !controls.profile &&
        !controls.record_behavior && !snapshot_output && !resumed_state && !controls.completion_delay_ms;
    const auto feature_mask = [](const Program& value) {
        std::uint32_t mask = 0;
        for (Node i = 0; i < value.length; ++i)
            if (value.opcode[i] == 0 && value.operand[i] >= 8 && value.operand[i] < 12)
                mask |= 1u << (value.operand[i] - 8);
        return mask;
    };
    auto regional_mask = program ? feature_mask(*program) : 0u;
    const auto controller_mask = [&](const ControllerProgram& c) {
        std::uint32_t mask = 0;
        for (unsigned r = 0; r < (c.kind ? 3u : 1u); ++r) mask |= feature_mask(c.trees[r]);
        return mask;
    };
    DeviceArray<ControllerProgram> controller_programs;
    if (controllers) {
        for (const auto& c : *controllers) validate_controller(c);
        controller_programs.allocate(controllers->size()); controller_programs.upload(*controllers);
        regional_mask = controller_mask(controllers->front());
    }
    DeviceArray<std::uint32_t> regional_masks;
    if (population && final_only) {
        std::vector<std::uint32_t> masks;
        for (std::size_t i = 0; i < population->size(); ++i)
            masks.push_back(controllers ? controller_mask((*controllers)[i]) : feature_mask((*population)[i]));
        regional_masks.allocate(masks.size()); regional_masks.upload(masks);
    }
    require(p.population_size == 1 || (population && final_only),
            "种群 Engine 只通过完整次数种群入口执行");
    const Node geometry_count = p.population_size > 1 ? p.geometries : 0;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine只允许一个活动评价");
    // 预分配容量不随验证尾块/恢复剩余成员改变；只有活动个体进入 grid 和 FE 账目。
    p.colonies = p.geometries * (population ? static_cast<Node>(population->size()) : 1);
    require(tasks.size() == p.colonies && mne_target > 0 && controls.completion_delay_ms <= 2000,
            "任务数量、MNE或诊断延迟无效");
    require(mode == PreparationMode::CachedCharged || mode == PreparationMode::EndToEnd, "未知准备模式");
    require(p.constraint_mode == ConstraintMode::Unrestricted || count_limited,
            "Hard主入口只支持评价次数，不能以旧截止路径发布图外廉价解");
    require(!(snapshot_output && resumed_state), "捕获和恢复不能同时启用");
    if (snapshot_output || resumed_state) {
        require(p.constraint_mode == ConstraintMode::Unrestricted && !factorial &&
                count_limited && controlled && mode == PreparationMode::CachedCharged &&
                !controls.profile && !controls.force_fingerprint_collisions,
                "状态分叉只接受普通控制次数入口，不混入计时剖析或强制碰撞");
    }
    if (snapshot_output) require(evaluation_limit > 0 && capture_after <= evaluation_limit &&
            capture_after % p.config.ants == 0, "快照停止点必须在完整源FE预算内且对齐蚂蚁批次");
    if (resumed_state) {
        resumed_state->validate();
        require(same_settings(resumed_state->settings, p.config) &&
            evaluation_limit <= resumed_state->progress_evaluation_limit -
                resumed_state->completed_batches * p.config.ants,
            "继续FE不能越过原progress预算或改变配置");
    }
    require(!intervention.enabled || (resumed_state && baseline && evaluation_limit > 0 &&
            intervention.region < 4 && (intervention.mne == 2 || intervention.mne == 16)),
            "配对干预须从快照开始并使用MNE2或16及合法固定区域");
    require(intervention.enabled || (intervention.seed == 0 && intervention.region == 0 && intervention.mne == 2),
            "未启用干预时不能附带分叉参数");
    if (intervention.enabled) require(experiment_mask & (1u << (intervention.region * 4 +
        (intervention.mne == 2 ? 0 : 3))), "干预动作不属于原实验mask");
    const Node first_batch = resumed_state ? static_cast<Node>(resumed_state->completed_batches) : 0;
    const std::uint64_t progress_limit = resumed_state ? resumed_state->progress_evaluation_limit : evaluation_limit;
    const std::uint64_t run_limit = snapshot_output ? capture_after : evaluation_limit;
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
    if (population) {
        require(!population->empty() && population->size() <= p.population_size, "程序数超过种群 Engine 容量");
        for (const auto& individual : *population) {
            validate_program(individual);
            require(individual.feature_spec_id == 2, "种群入口只接受评价次数特征 v2");
        }
        p.population_programs.upload_prefix(*population);
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
    result.evaluation_limit_per_colony = run_limit;
    result.checkpoint_iterations = controls.checkpoint_iterations;
    DeviceArray<float> decision_features, decision_scores;
    DeviceArray<std::uint32_t> decision_masks;
    DeviceArray<std::int32_t> decision_actions;
    std::size_t decision_cursor = 0;
    if (!controls.decision_iterations.empty()) {
        const auto& points = controls.decision_iterations;
        require(count_limited && controlled && !population && !snapshot_output && !resumed_state &&
            points.front() > 0 && points.back() <= run_limit / p.config.ants &&
            std::is_sorted(points.begin(), points.end()) &&
            std::adjacent_find(points.begin(), points.end()) == points.end(),
            "决策采样要求完整次数预算内递增的单程序迭代编号");
        const auto count = points.size() * p.colonies;
        decision_features.allocate(count * 12 * 32); decision_scores.allocate(count * 32);
        decision_masks.allocate(count); decision_actions.allocate(count);
    }
    DeviceArray<Node> checkpoint_tours;
    if (!controls.checkpoint_iterations.empty()) {
        require(count_limited && !snapshot_output && !resumed_state &&
            std::is_sorted(controls.checkpoint_iterations.begin(), controls.checkpoint_iterations.end()) &&
            std::adjacent_find(controls.checkpoint_iterations.begin(), controls.checkpoint_iterations.end()) ==
                controls.checkpoint_iterations.end() && controls.checkpoint_iterations.back() <= run_limit / p.config.ants,
            "曲线记录点必须为预算内递增、不重复的迭代数");
        checkpoint_tours.allocate(controls.checkpoint_iterations.size() * p.colonies * p.n);
    }
    std::size_t checkpoint_cursor = 0;
    const auto record_checkpoint = [&]() {
        if (checkpoint_cursor < controls.checkpoint_iterations.size() &&
            result.completed_batches == controls.checkpoint_iterations[checkpoint_cursor]) {
            const auto nodes = static_cast<std::size_t>(p.colonies) * p.n;
            checked(cudaMemcpyAsync(checkpoint_tours.data() + checkpoint_cursor * nodes,
                                    p.global.data(), nodes * sizeof(Node), cudaMemcpyDeviceToDevice));
            ++checkpoint_cursor;
        }
    };
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
    std::vector<std::unique_ptr<cuda_detail::ProfileEvents>> production_events;
    if (controls.profile_events_only) {
        require(final_only, "生产 event 测量只支持无轨迹的次数入口");
        // 计时仅在专用测量调用开启；正常训练不创建 event、不增加批末同步。
        const auto count = 1 + run_limit / p.config.ants;
        production_events.reserve(count);
        for (std::uint64_t i = 0; i < count; ++i)
            production_events.push_back(std::make_unique<cuda_detail::ProfileEvents>());
        events = production_events.front().get();
    }
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

    const auto& c = p.config;
    const Node ants = c.ants * p.colonies, cells = p.n * c.primary_width * p.colonies;
    if (!resumed_state) {
        // 捕获入口先清空全部持久字节，避免把未定义的工作缓冲内容写入快照。
        if (snapshot_output) p.visit_buffers([](const char*, auto& buffer, std::uint64_t) { buffer.zero(); });
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
        std::vector<std::uint64_t> instance_ids, keys;
        for (const auto& task : tasks) {
            if (instance_ids.size() < p.geometries) instance_ids.push_back(task.instance_key);
            keys.push_back(mixed(task.seed ^ mixed(task.instance_key + 0xd1b54a32d192ed03ULL)));
            if (p.constraint_mode != ConstraintMode::Unrestricted)
                result.graph_edges_per_colony.push_back(p.registry.at(task.instance_key).graph->edges());
        }
        const bool refresh_geometry = mode != PreparationMode::CachedCharged || snapshot_output ||
            controls.profile || p.resident_instances != instance_ids || (controlled && !p.resident_control);
        if (events) events->reset();
        event_begin(GpuProfileStage::Initialization);
        auto upload_started = profile_now();
        if (refresh_geometry) {
        std::vector<double> xy, costs, local_scale, epsilons;
        std::vector<Node> primary, backup, ls, initial, samples;
        std::vector<Node> graph_offsets, graph_neighbors;
        std::map<std::uint64_t, std::vector<Node>> sample_cache;
        for (Node geometry = 0; geometry < p.geometries; ++geometry) {
            const auto& task = tasks[geometry];
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
            }
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
        upload_started = profile_now();
        p.xy.upload(xy); p.costs.upload(costs); p.primary.upload(primary); p.backup.upload(backup);
        p.ls.upload(ls); p.initial.upload(initial);
        if (p.constraint_mode != ConstraintMode::Unrestricted) {
            p.graph_offsets.upload(graph_offsets); p.graph_neighbors.upload(graph_neighbors);
        }
        if (controlled) {
            p.local_scale.upload(local_scale); p.epsilons.upload(epsilons); p.samples.upload(samples);
        }
        cuda_detail::cache_candidate_distances<<<(p.n * c.ls_width * p.geometries + 255) / 256, 256>>>(
            p.xy.data(), p.n * c.ls_width * p.geometries, p.n, c.ls_width, p.ls.data(), p.ls_distances.data());
        if (p.pair_distances.bytes()) cuda_detail::cache_pair_distances<<<
            (static_cast<std::size_t>(p.n) * p.n * p.geometries + 255) / 256, 256>>>(
            p.xy.data(), p.n, p.geometries, p.pair_distances.data());
        p.resident_instances = std::move(instance_ids);
        p.resident_control = controlled;
        }
        if (p.constraint_mode == ConstraintMode::Escape) p.reset_escape();
        p.keys.upload_prefix(keys);
        p.targets.upload_prefix(std::vector<Node>(ants, mne_target));
        if (controls.profile) result.profile.upload_seconds += host_duration(upload_started);
        cuda_detail::reset_colony<<<p.colonies, 128>>>(p.n, p.initial.data(), p.costs.data(), p.parent.data(),
            p.epoch.data(), p.global.data(), p.state.data(), c.primary_width, c.retention, c.p_best, geometry_count);
        cuda_detail::initialize_products<<<(cells + 255) / 256, 256>>>(
            cuda_detail::CoordinateDistance{p.xy.data()}, cells, p.n, c.primary_width, p.primary.data(),
            c.beta, p.state.data(), p.heuristic.data(), p.trails.data(), p.products.data(), geometry_count);
        if (controlled) {
            upload_started = profile_now();
            p.experiment_masks.upload_prefix(std::vector<std::uint32_t>(p.colonies, experiment_mask));
            if (controls.profile) result.profile.upload_seconds += host_duration(upload_started);
            cuda_detail::initialize_control<<<p.colonies, 128>>>(p.n, p.initial.data(), p.costs.data(),
                p.parent_position.data(), p.archive.data(), p.archive_positions.data(), p.controllers.data(),
                controls.force_fingerprint_collisions, geometry_count);
            p.actions.zero(); p.apply_actions(true);
        }
    } else {
        p.restore(*resumed_state);
        p.resident_instances.clear();
        if (intervention.enabled) {
            auto fork_keys = p.keys.download();
            // 同一快照/seed配对使用相同keys；分支顺序和MNE不参与随机域派生。
            for (auto& key : fork_keys) key = mixed(key ^ mixed(intervention.seed + 0xa0761d6478bd642fULL));
            p.keys.upload(fork_keys);
        }
        const auto states = p.state.download();
        const auto global = p.global.download();
        for (Node colony = 0; colony < p.colonies; ++colony) {
            const auto first = global.begin() + static_cast<std::size_t>(colony) * p.n;
            std::vector<Node> tour(first, first + p.n);
            const auto& problem = p.registry.at(tasks[colony].instance_key);
            CpuTour verified(tour, [&](Node a, Node b) { return problem.distance(a, b); });
            const double cost = verified.cost();
            require(std::isfinite(states[colony].global_cost) &&
                std::abs(cost - states[colony].global_cost) <= 1e-8 + 1e-12 * cost,
                "快照全局tour成本与完整重算不符");
            const auto& item = resumed_state->incumbents[colony];
            CpuTour host_verified(item.tour, [&](Node a, Node b) { return problem.distance(a, b); });
            require(host_verified.cost() == item.cost, "快照主机incumbent成本改变");
            offer({colony}, item.tour, item.cost);
        }
    }
    event_end(GpuProfileStage::Initialization);
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    if (events) result.profile.initialization_gpu_milliseconds =
        events->elapsed()[static_cast<unsigned>(GpuProfileStage::Initialization)];
    const auto initial_download_started = profile_now();
    record_checkpoint();
    const auto initial_control = controlled && !final_only ? p.controllers.download() : std::vector<ControllerState>{};
    if (controls.profile) result.profile.initialization_download_seconds = host_duration(initial_download_started);
    if (budget.expired()) return finish();
    result.preparation_completed = true;
    if (controlled) result.completed_control_states = initial_control;

    for (Node batch = first_batch; budget.can_start(); ++batch) {
        if (count_limited && result.completed_batches == run_limit / p.config.ants) break;
        if (controls.fixed_batches && batch >= controls.fixed_batches) break;
        ++result.launched_batches;
        const auto batch_started = profile_now();
        BatchPhaseProfile profile; profile.batch = batch;
        if (controls.profile_events_only) events = production_events[1 + batch - first_batch].get();
        if (events) events->reset();
        ControlBatchTrace trace;
        if (controlled) {
            trace.batch = batch;
            trace.intervened = intervention.enabled && batch == first_batch;
            if (controls.capture_control) trace.before = p.snapshot();
            // feature_spec=2的进度只依赖已完成FE；计时插桩和主机调度不改变动作输入。
            trace.elapsed_ratio = count_limited ?
                static_cast<double>((first_batch + result.completed_batches) * c.ants) / progress_limit :
                (controls.fixed_elapsed_ratio >= 0 ? controls.fixed_elapsed_ratio : budget.elapsed() / seconds);
            event_begin(GpuProfileStage::Features);
            // 干预第一批保留快照的区域，不能因更换分叉seed重新抽取区域。
            const bool capture_features = !final_only || (decision_cursor < controls.decision_iterations.size() &&
                batch + 1 == controls.decision_iterations[decision_cursor]);
            if (!trace.intervened) p.build_features(batch, trace.elapsed_ratio,
                capture_features ? 15u : regional_mask, capture_features ? nullptr : regional_masks.data());
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
            if (controllers) {
                cuda_detail::score_controllers<<<(p.colonies + 3) / 4, 128>>>(controller_programs.data(),
                    p.features.data(), p.masks.data(), p.scores.data(), p.actions.data(), p.colonies, p.geometries);
            } else if (program) {
                cuda_detail::score_actions<<<(p.colonies + 3) / 4, 128>>>(*program, p.features.data(),
                    p.masks.data(), p.scores.data(), p.actions.data(), p.colonies,
                    p.population_programs.data(), p.geometries);
            } else {
                cuda_detail::select_baseline_actions<<<(p.colonies + 127) / 128, 128>>>(*baseline,
                    p.controllers.data(), p.keys.data(), batch, p.masks.data(), p.actions.data(), p.colonies,
                    baseline_uniforms.data());
            }
            if (trace.intervened) p.actions.upload(std::vector<std::int32_t>(p.colonies,
                static_cast<std::int32_t>(intervention.region * 4 + (intervention.mne == 2 ? 0 : 3))));
            event_end(GpuProfileStage::Scoring);
            if (decision_cursor < controls.decision_iterations.size() &&
                batch + 1 == controls.decision_iterations[decision_cursor]) {
                // 仅设备到设备保存；循环完成后一次返回，不进行逐步 CPU 同步。
                const auto offset = decision_cursor * p.colonies;
                checked(cudaMemcpyAsync(decision_features.data() + offset * 12 * 32,
                    p.features.data(), p.colonies * 12 * 32 * sizeof(float), cudaMemcpyDeviceToDevice));
                checked(cudaMemcpyAsync(decision_scores.data() + offset * 32,
                    p.scores.data(), p.colonies * 32 * sizeof(float), cudaMemcpyDeviceToDevice));
                checked(cudaMemcpyAsync(decision_masks.data() + offset, p.masks.data(),
                    p.colonies * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice));
                checked(cudaMemcpyAsync(decision_actions.data() + offset, p.actions.data(),
                    p.colonies * sizeof(std::int32_t), cudaMemcpyDeviceToDevice));
                ++decision_cursor;
            }
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
        const cuda_detail::BatchCoordinateDistance distance{p.xy.data(), p.n, c.ants, p.pair_distances.data(), geometry_count};
        const cuda_detail::BatchStochasticChoices choices{p.primary.data(), p.backup.data(), p.products.data(),
            p.keys.data(), c.primary_width, c.backup_width, batch, c.ants, geometry_count};
        const auto* cached_parent_costs = p.n <= 1500 ? p.parent_costs.data() : nullptr;
        const auto construct = [&](auto view, auto allowed) {
            if (controls.profile) {
                cuda_detail::construct_and_search<decltype(distance), decltype(view), decltype(allowed), true>
                    <<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                    p.gains.data(), diagnostic_construction.data(), p.info.data(), p.visited.data(), allowed,
                    p.profiling->cycles.data(), construction_new_edges.data(), p.queued.data(),
                    p.constraint_mode == ConstraintMode::Escape ? nullptr : p.ls_distances.data(), cached_parent_costs);
            } else if constexpr (std::is_same_v<std::decay_t<decltype(allowed)>, UnrestrictedEdges>) {
                if (final_only && p.n <= 1500 && ants % 4 == 0) {
                // 一 warp 一条路线、每 block 四条路线；只改变并行协作布局。
                if (p.population_size > 1) {
                cuda_detail::construct_and_search<decltype(distance), decltype(view), decltype(allowed), false, true, true>
                    <<<ants / 4, 128, static_cast<std::size_t>(p.n) * 4 * 3 * sizeof(std::uint16_t)>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_position.data(), nullptr, p.pending.data(),
                    p.gains.data(), nullptr, p.info.data(), p.visited.data(), allowed,
                    nullptr, nullptr, p.queued.data(), p.ls_distances.data(), cached_parent_costs);
                } else {
                cuda_detail::construct_and_search<decltype(distance), decltype(view), decltype(allowed), false, true>
                    <<<ants / 4, 128, static_cast<std::size_t>(p.n) * 4 * 3 * sizeof(std::uint16_t)>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                    p.gains.data(), nullptr, p.info.data(), p.visited.data(), allowed,
                    nullptr, nullptr, p.queued.data(), p.ls_distances.data(), cached_parent_costs);
                }
                } else {
                cuda_detail::construct_and_search<<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                    p.gains.data(), diagnostic_construction.data(), p.info.data(), p.visited.data(), allowed,
                    nullptr, construction_new_edges.data(), p.queued.data(),
                    p.constraint_mode == ConstraintMode::Escape ? nullptr : p.ls_distances.data(), cached_parent_costs);
                }            } else {
                cuda_detail::construct_and_search<<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
                    p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                    p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                    p.gains.data(), diagnostic_construction.data(), p.info.data(), p.visited.data(), allowed,
                    nullptr, construction_new_edges.data(), p.queued.data(),
                    p.constraint_mode == ConstraintMode::Escape ? nullptr : p.ls_distances.data(), cached_parent_costs);
            }
        };
        event_begin(GpuProfileStage::ConstructionAndSearch);
        // 所有蚂蚁共享父代，只计算一次边长并保持原顺序求和；大规模仍走原有路径。
        if (cached_parent_costs) cuda_detail::cache_parent_costs<<<p.colonies, 128, p.n * sizeof(double)>>>(
            p.xy.data(), p.n, p.parent.data(), p.parent_costs.data(), p.pair_distances.data(), geometry_count);
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
            p.heuristic.data(), p.trails.data(), p.products.data(), geometry_count);
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
        checked(cudaGetLastError());
        if (final_only) {
            ++result.completed_batches;
            record_checkpoint();
            continue;
        }
        checked(cudaDeviceSynchronize());
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
        record_checkpoint();
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
    if (final_only) {
        checked(cudaDeviceSynchronize());
        for (const auto& event : production_events) {
            const auto elapsed = event->elapsed();
            for (unsigned i = 0; i < elapsed.size(); ++i)
                result.production_kernel_milliseconds[i] += elapsed[i];
        }
        const auto states = p.state.download_prefix(p.colonies);
        const auto tours = p.global.download_prefix(static_cast<std::size_t>(p.colonies) * p.n);
        const double completed = budget.elapsed();
        result.last_batch_completed_seconds = completed;
        if (controlled) result.completed_control_states = p.controllers.download_prefix(p.colonies);
        for (Node colony = 0; colony < p.colonies; ++colony) {
            const auto first = tours.begin() + static_cast<std::size_t>(colony) * p.n;
            result.incumbents[colony].offer(std::vector<Node>(first, first + p.n),
                                           states[colony].global_cost, completed, budget);
            const auto& work = states[colony].work;
            if (population) result.colony_work.push_back({work.construction, work.local_search, work.constraints});
            result.completed_construction_steps += work.construction;
            result.completed_ls_evaluations += work.local_search;
            result.completed_constraint_rejections += work.constraints;
            result.escape.construction_opportunities += work.escape.construction_opportunities;
            result.escape.construction_gates += work.escape.construction_gates;
            result.escape.construction_replaced_slots += work.escape.construction_replaced_slots;
            result.escape.escape_relocations += work.escape.escape_relocations;
            result.escape.ls_anchor_nodes += work.escape.ls_anchor_nodes;
            result.escape.ls_replaced_slots += work.escape.ls_replaced_slots;
            result.escape.old_view_reactivations += work.escape.old_view_reactivations;
            result.escape.anchor_reactivations += work.escape.anchor_reactivations;
            result.escape.new_edges += work.escape.new_edges;
            result.escape.construction_capacity_rejections += work.escape.construction_capacity_rejections;
            result.escape.ls_capacity_rejections += work.escape.ls_capacity_rejections;
        }
    }
    if (checkpoint_tours.bytes()) result.checkpoint_tours = checkpoint_tours.download();
    if (decision_features.bytes()) {
        const auto features = decision_features.download(), scores = decision_scores.download();
        const auto masks = decision_masks.download();
        const auto actions = decision_actions.download();
        for (std::size_t i = 0; i < decision_cursor; ++i) {
            const auto offset = i * p.colonies;
            DecisionTrace trace; trace.iteration = controls.decision_iterations[i];
            trace.conditional_three = controllers && controllers->front().kind;
            trace.fixed_policy = baseline != nullptr;
            trace.features.assign(features.begin() + offset * 12 * 32,
                                  features.begin() + (offset + p.colonies) * 12 * 32);
            trace.scores.assign(scores.begin() + offset * 32,
                                scores.begin() + (offset + p.colonies) * 32);
            trace.masks.assign(masks.begin() + offset, masks.begin() + offset + p.colonies);
            trace.actions.assign(actions.begin() + offset, actions.begin() + offset + p.colonies);
            result.decisions.push_back(std::move(trace));
        }
    }
    if (snapshot_output) *snapshot_output = p.capture(tasks, first_batch + result.completed_batches,
                                                     progress_limit, experiment_mask, result.incumbents);
    return finish();
}

}  // namespace gp_faco
