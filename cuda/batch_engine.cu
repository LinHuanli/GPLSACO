// 固定多实例面板与截止前incumbent提交；全程无Python内层回调。
#include "gp_faco/batch_engine.hpp"
#include "gp_faco/prepared_problem.hpp"
#include "colony_state.cuh"
#include "faco_device.cuh"

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
    std::size_t allocated_bytes = 0;
    DeviceArray<double> xy, costs, heuristic, trails, products, gains;
    DeviceArray<Node> primary, backup, ls, initial, parent, parent_position, epoch, global,
        targets, tours, positions, parent_positions, scratch, pending;
    DeviceArray<std::uint64_t> keys;
    DeviceArray<std::uint8_t> visited;
    DeviceArray<FacoDiagnosticInfo> info;
    DeviceArray<Colony> state;

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

BatchEvaluation FacoBatchEngine::evaluate(const std::vector<BatchTask>& tasks, double seconds,
                                        Node mne_target, PreparationMode mode) {
    return evaluate_diagnostic(tasks, seconds, mne_target, mode, {});
}

BatchEvaluation FacoBatchEngine::evaluate_diagnostic(const std::vector<BatchTask>& tasks, double seconds,
    Node mne_target, PreparationMode mode, BatchDiagnosticControls controls) {
    const auto started = Clock::now();
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    require(lock.owns_lock(), "同一Engine只允许一个活动评价");
    require(tasks.size() == p.colonies && mne_target > 0 && controls.completion_delay_ms <= 2000,
            "任务数量、MNE或诊断延迟无效");
    require(mode == PreparationMode::CachedCharged || mode == PreparationMode::EndToEnd, "未知准备模式");
    int device = -1; checked(cudaGetDevice(&device));
    require(device == p.device, "Engine必须在创建它的CUDA设备上评价");
    const auto actual_elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - started).count(); };
    DeadlineLedger budget(seconds, actual_elapsed);
    BatchEvaluation result;
    result.budget_seconds = seconds;
    result.allocated_device_bytes = p.allocated_bytes;
    result.incumbents.resize(p.colonies);
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
            budget.charge(cached.cheap_seconds);
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
            budget.charge(cached.preparation_seconds);
            offer(ids, cached.initial_tour, cached.initial_cost);
        } else {
            auto& prepared = fresh.at(key);
            if (!prepare_problem(prepared, [&]() { return budget.expired(); })) return finish();
            offer(ids, prepared.initial_tour, prepared.initial_cost);
        }
        if (budget.expired()) return finish();
    }

    std::vector<double> xy, costs;
    std::vector<Node> primary, backup, ls, initial;
    std::vector<std::uint64_t> keys;
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
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    if (budget.expired()) return finish();
    result.preparation_completed = true;

    for (Node batch = 0; budget.can_start(); ++batch) {
        if (controls.fixed_batches && batch >= controls.fixed_batches) break;
        ++result.launched_batches;
        const cuda_detail::BatchCoordinateDistance distance{p.xy.data(), p.n, c.ants};
        const cuda_detail::BatchStochasticChoices choices{p.primary.data(), p.backup.data(), p.products.data(),
            p.keys.data(), c.primary_width, c.backup_width, batch, c.ants};
        cuda_detail::construct_and_search<<<ants, 128>>>(distance, p.ls.data(), p.n, c.ls_width,
            p.parent.data(), choices, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
            p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
            p.gains.data(), nullptr, p.info.data(), p.visited.data());
        cuda_detail::reduce_and_select<<<p.colonies, 128>>>(p.n, c.ants, p.tours.data(), p.info.data(),
            p.parent.data(), p.parent_position.data(), p.epoch.data(), p.global.data(), p.state.data(),
            c.primary_width, c.retention, c.p_best, c.epoch_source_probability, p.keys.data(), batch);
        cuda_detail::update_pheromone<<<(cells + 255) / 256, 256>>>(p.n, c.primary_width, p.colonies,
            p.primary.data(), p.parent.data(), p.parent_position.data(), c.retention, p.state.data(),
            p.heuristic.data(), p.trails.data(), p.products.data());
        checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
        const auto states = p.state.download();
        const auto tours = p.global.download();
        const auto info = p.info.download();
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
        for (const auto& ant : info) {
            result.completed_construction_steps += ant.construction.steps;
            result.completed_ls_evaluations += ant.local_search.move_evaluations;
        }
        if (batch == std::numeric_limits<Node>::max()) throw std::runtime_error("随机批次编号用尽");
    }
    return finish();
}

}  // namespace gp_faco
