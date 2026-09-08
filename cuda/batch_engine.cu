// 固定多实例面板与截止前incumbent提交；全程无Python内层回调。
#include "gp_faco/batch_engine.hpp"
#include "gp_faco/prepared_problem.hpp"
#include "colony_state.cuh"
#include "faco_device.cuh"
#include "control_state.cuh"
#include "gp_score.cuh"

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

    Impl(Node dimension, Node count, FixedFacoSettings settings)
        : n(dimension), colonies(count), config(normalized_settings(settings, dimension)) {
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
        return result;
    }

    void apply_actions(bool initializing = false) {
        const auto& c = config;
        cuda_detail::apply_control_action<<<colonies, 128>>>(n, c.ants, c.primary_width, c.ls_width,
            actions.data(), alternatives.data(), archive.data(), archive_positions.data(), parent.data(),
            parent_position.data(), epoch.data(), state.data(), controllers.data(), c.retention, c.p_best,
            trails.data(), heuristic.data(), products.data(), targets.data(), tours.data(), positions.data(),
            parent_positions.data(), scratch.data(), pending.data(), visited.data(), gains.data(), info.data(),
            identities.data(), initializing);
    }
};

FacoBatchEngine::FacoBatchEngine(Node n, Node colonies, FixedFacoSettings settings)
    : impl_(std::make_unique<Impl>(n, colonies, settings)) {}
FacoBatchEngine::~FacoBatchEngine() = default;

RegistrationInfo FacoBatchEngine::register_problem(std::uint64_t key, std::vector<double> coordinates) {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine的注册与评价不能重叠");
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

BatchEvaluation FacoBatchEngine::evaluate_impl(const std::vector<BatchTask>& tasks, double seconds,
    Node mne_target, PreparationMode mode, BatchDiagnosticControls controls, const Program* program,
    std::uint32_t experiment_mask) {
    const auto started = Clock::now();
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine只允许一个活动评价");
    require(tasks.size() == p.colonies && mne_target > 0 && controls.completion_delay_ms <= 2000,
            "任务数量、MNE或诊断延迟无效");
    require(mode == PreparationMode::CachedCharged || mode == PreparationMode::EndToEnd, "未知准备模式");
    if (program) {
        validate_program(*program);
        require((experiment_mask & 0xffffu) != 0, "实验mask必须保留至少一个保持模式动作");
        require(!controls.force_fingerprint_collisions || controls.capture_control,
                "强制指纹碰撞必须同时收集诊断轨迹");
    }
    int device = -1; checked(cudaGetDevice(&device));
    require(device == p.device, "Engine必须在创建它的CUDA设备上评价");
    const auto actual_elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - started).count(); };
    DeadlineLedger budget(seconds, actual_elapsed);
    BatchEvaluation result;
    result.budget_seconds = seconds;
    result.allocated_device_bytes = p.allocated_bytes;
    result.incumbents.resize(p.colonies);
    if (program) result.completed_control_states.resize(p.colonies);
    std::map<std::uint64_t, std::vector<Node>> groups;
    for (Node colony = 0; colony < p.colonies; ++colony) {
        require(p.registry.find(tasks[colony].instance_key) != p.registry.end(), "任务实例尚未注册");
        groups[tasks[colony].instance_key].push_back(colony);
    }
    const auto finish = [&]() {
        result.actual_seconds = actual_elapsed(); result.charged_seconds = budget.charged();
        result.elapsed_seconds = result.actual_seconds + result.charged_seconds;
        result.overrun_seconds = std::max(0.0, result.elapsed_seconds - seconds);
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
            budget.charge(fixed == p.preparation_charges.end() ? cached.cheap_seconds : fixed->second.cheap_seconds);
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
            budget.charge(fixed == p.preparation_charges.end() ? cached.preparation_seconds : fixed->second.preparation_seconds);
            offer(ids, cached.initial_tour, cached.initial_cost);
        } else {
            auto& prepared = fresh.at(key);
            if (!prepare_problem(prepared, [&]() { return budget.expired(); })) return finish();
            offer(ids, prepared.initial_tour, prepared.initial_cost);
        }
        if (budget.expired()) return finish();
    }

    std::vector<double> xy, costs, local_scale, epsilons;
    std::vector<Node> primary, backup, ls, initial, samples;
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
        keys.push_back(mixed(task.seed ^ mixed(task.instance_key + 0xd1b54a32d192ed03ULL)));
        if (program) {
            local_scale.insert(local_scale.end(), prepared.local_scale.begin(), prepared.local_scale.end());
            epsilons.push_back(prepared.scale_epsilon);
            auto [entry, inserted] = sample_cache.try_emplace(task.instance_key, sample_capacity, 0);
            if (inserted) sample_nodes(p.n, task.instance_key, entry->second.data());
            samples.insert(samples.end(), entry->second.begin(), entry->second.end());
        }
    }
    if (!budget.can_start()) return finish();
    p.xy.upload(xy); p.costs.upload(costs); p.primary.upload(primary); p.backup.upload(backup);
    p.ls.upload(ls); p.initial.upload(initial); p.keys.upload(keys);
    const auto& c = p.config;
    const Node ants = c.ants * p.colonies, cells = p.n * c.primary_width * p.colonies;
    p.targets.upload(std::vector<Node>(ants, mne_target));
    cuda_detail::reset_colony<<<p.colonies, 128>>>(p.n, p.initial.data(), p.costs.data(), p.parent.data(),
        p.epoch.data(), p.global.data(), p.state.data(), c.primary_width, c.retention, c.p_best);
    cuda_detail::initialize_products<<<(cells + 255) / 256, 256>>>(
        cuda_detail::CoordinateDistance{p.xy.data()}, cells, p.n, c.primary_width, p.primary.data(),
        c.beta, p.state.data(), p.heuristic.data(), p.trails.data(), p.products.data());
    if (program) {
        p.local_scale.upload(local_scale); p.epsilons.upload(epsilons); p.samples.upload(samples);
        p.experiment_masks.upload(std::vector<std::uint32_t>(p.colonies, experiment_mask));
        cuda_detail::initialize_control<<<p.colonies, 128>>>(p.n, p.initial.data(), p.costs.data(),
            p.parent_position.data(), p.archive.data(), p.archive_positions.data(), p.controllers.data(),
            controls.force_fingerprint_collisions);
        p.actions.zero(); p.apply_actions(true);
    }
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    const auto initial_control = program ? p.controllers.download() : std::vector<ControllerState>{};
    if (budget.expired()) return finish();
    result.preparation_completed = true;
    if (program) result.completed_control_states = initial_control;

    for (Node batch = 0; budget.can_start(); ++batch) {
        if (controls.fixed_batches && batch >= controls.fixed_batches) break;
        ++result.launched_batches;
        ControlBatchTrace trace;
        if (program) {
            trace.batch = batch;
            if (controls.capture_control) trace.before = p.snapshot();
            trace.elapsed_ratio = budget.elapsed() / seconds;
            cuda_detail::build_control_features<<<p.colonies, 128>>>(p.n, c.primary_width, p.colonies,
                p.xy.data(), p.primary.data(), p.local_scale.data(), p.epsilons.data(), p.samples.data(),
                p.keys.data(), batch, trace.elapsed_ratio, p.parent.data(), p.parent_position.data(),
                p.archive.data(), p.archive_positions.data(), p.controllers.data(), p.state.data(),
                p.trails.data(), p.experiment_masks.data(), p.regions.data(), p.alternatives.data(),
                p.masks.data(), p.features.data());
            cuda_detail::score_actions<<<(p.colonies + 3) / 4, 128>>>(*program, p.features.data(),
                p.masks.data(), p.scores.data(), p.actions.data(), p.colonies);
            if (controls.capture_control) {
                trace.features = p.features.download(); trace.scores = p.scores.download();
                trace.masks = p.masks.download(); trace.actions = p.actions.download();
                trace.alternatives = p.alternatives.download(); trace.regions = p.regions.download();
            }
            p.apply_actions();
            if (controls.capture_control) trace.after_restart = p.snapshot();
        }
        const cuda_detail::BatchCoordinateDistance distance{p.xy.data(), p.n, c.ants};
        const cuda_detail::BatchStochasticChoices choices{p.primary.data(), p.backup.data(), p.products.data(),
            p.keys.data(), c.primary_width, c.backup_width, batch, c.ants};
        const auto construct = [&](auto view) {
            cuda_detail::construct_and_search<<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
                p.parent.data(), view, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
                p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
                p.gains.data(), nullptr, p.info.data(), p.visited.data());
        };
        if (program) construct(cuda_detail::BatchRegionChoices{choices, p.regions.data(), p.actions.data()});
        else construct(choices);
        cuda_detail::reduce_and_select<<<p.colonies, 128>>>(p.n, c.ants, p.tours.data(), p.info.data(),
            p.parent.data(), p.parent_position.data(), p.epoch.data(), p.global.data(), p.state.data(),
            c.primary_width, c.retention, c.p_best, c.epoch_source_probability, p.keys.data(), batch);
        cuda_detail::update_pheromone<<<(cells + 255) / 256, 256>>>(p.n, c.primary_width, p.colonies,
            p.primary.data(), p.parent.data(), p.parent_position.data(), c.retention, p.state.data(),
            p.heuristic.data(), p.trails.data(), p.products.data());
        if (program) {
            cuda_detail::fingerprint_ants<<<ants, 128>>>(p.n, p.tours.data(), p.identities.data(),
                controls.force_fingerprint_collisions);
            cuda_detail::update_control<<<p.colonies, 128>>>(p.n, c.ants, p.tours.data(), p.positions.data(),
                p.info.data(), p.identities.data(), p.state.data(), p.archive.data(), p.archive_positions.data(),
                p.archive_scratch.data(), p.archive_scratch_positions.data(), p.controllers.data(), c.ls_evaluation_limit);
        }
        checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
        const auto states = p.state.download();
        const auto tours = p.global.download();
        const auto info = p.info.download();
        const auto controller_state = program ? p.controllers.download() : std::vector<ControllerState>{};
        if (controls.capture_control) trace.after_batch = p.snapshot();
        std::vector<std::vector<Node>> candidates;
        std::vector<double> candidate_costs;
        for (Node colony = 0; colony < p.colonies; ++colony) {
            const auto base = static_cast<std::size_t>(colony) * p.n;
            std::vector<Node> candidate(tours.begin() + base, tours.begin() + base + p.n);
            const auto& problem = p.registry.at(tasks[colony].instance_key);
            // 同步完成后验证完整排列/成本；验证耗时也在完成时间戳之前。
            CpuTour verified(candidate, [&](Node a, Node b) { return problem.distance(a, b); });
            const double value = verified.cost();
            require(std::isfinite(states[colony].global_cost) &&
                    std::abs(value - states[colony].global_cost) <= 1e-8 + 1e-12 * value,
                    "设备incumbent成本与完整重算不符");
            candidates.push_back(std::move(candidate)); candidate_costs.push_back(value);
        }
        if (controls.completion_delay_ms && batch == controls.delay_batch)
            std::this_thread::sleep_for(std::chrono::milliseconds(controls.completion_delay_ms));
        const double completed = budget.elapsed();
        result.last_batch_completed_seconds = completed;
        if (!budget.completed_on_time(completed)) {
            ++result.discarded_batches;
            if (controls.capture_discarded) result.discarded_costs = candidate_costs;
            break;
        }
        for (Node colony = 0; colony < p.colonies; ++colony)
            result.incumbents[colony].offer(std::move(candidates[colony]), candidate_costs[colony], completed, budget);
        ++result.completed_batches;
        if (program) result.completed_control_states = controller_state;
        if (controls.capture_control) result.control_trace.push_back(std::move(trace));
        for (const auto& ant : info) {
            result.completed_construction_steps += ant.construction.steps;
            result.completed_ls_evaluations += ant.local_search.move_evaluations;
        }
        if (batch == std::numeric_limits<Node>::max()) throw std::runtime_error("随机批次编号用尽");
    }
    return finish();
}

}  // namespace gp_faco
