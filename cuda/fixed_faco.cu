// 复用已核验Route/LS的固定迭代FACO开发实现。
// 原始FACO语义 Copyright (c) 2024 RSkinderowicz，MIT许可见provenance。
#include "gp_faco/fixed_faco_gpu.hpp"
#include "faco_choices.cuh"
#include "faco_device.cuh"
#include "colony_state.cuh"
#include "gp_faco/prepared_problem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <mutex>
#include <stdexcept>

namespace gp_faco {
namespace {
using Clock = std::chrono::steady_clock;

void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

using cuda_detail::DeviceArray;
using cuda_detail::Colony;
using cuda_detail::reset_colony;
using cuda_detail::initialize_products;
using cuda_detail::reduce_and_select;
using cuda_detail::update_pheromone;

std::uint64_t mixed(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

struct FixedFacoGpu::Impl {
    std::mutex evaluation_mutex;
    FixedFacoSettings config;
    Node n;
    std::uint64_t instance_key;
    std::vector<double> coordinates;
    CandidateRows primary, backup, ls;
    std::vector<Node> initial;
    double initial_cost, preparation_seconds;
    std::size_t allocated_bytes = 0;
    DeviceArray<double> xy, heuristic, trails, products, gains, selection_uniforms, initial_costs;
    DeviceArray<std::uint64_t> keys;
    DeviceArray<Node> d_primary, d_backup, d_ls, d_initial, parent, parent_position,
        epoch, global, targets, tours, positions, parent_positions, scratch, pending,
        constructed, selections;
    DeviceArray<std::uint8_t> visited;
    DeviceArray<FacoDiagnosticInfo> info;
    DeviceArray<Colony> state;

    Impl(std::vector<double> input, std::uint64_t key, FixedFacoSettings settings)
        : config(settings), n(input.size() / 2), instance_key(key), coordinates(std::move(input)) {
        const auto started = Clock::now();
        auto prepared = make_cheap_problem(std::move(coordinates), config);
        prepare_problem(prepared);
        coordinates = std::move(prepared.coordinates); config = prepared.settings;
        primary = std::move(prepared.primary); backup = std::move(prepared.backup);
        ls = std::move(prepared.ls); initial = std::move(prepared.initial_tour);
        initial_cost = prepared.initial_cost;
        auto flat_primary = flattened(primary), flat_backup = flattened(backup), flat_ls = flattened(ls);

        const std::size_t cells = static_cast<std::size_t>(n) * config.ants;
        const auto allocate = [&](auto& buffer, std::size_t count) {
            buffer.allocate(count); allocated_bytes += buffer.bytes();
        };
        allocate(initial_costs, 1); allocate(keys, 1); initial_costs.upload({initial_cost});
        allocate(xy, coordinates.size()); allocate(d_primary, flat_primary.size());
        allocate(d_backup, flat_backup.size()); allocate(d_ls, flat_ls.size());
        allocate(d_initial, n); allocate(parent, n); allocate(parent_position, n);
        allocate(epoch, n); allocate(global, n); allocate(targets, config.ants);
        allocate(tours, cells); allocate(positions, cells); allocate(parent_positions, cells);
        allocate(scratch, cells); allocate(pending, cells * 5); allocate(visited, cells);
        allocate(info, config.ants); allocate(state, 1); allocate(gains, config.ants * config.ls_width * 2);
        allocate(heuristic, flat_primary.size()); allocate(trails, flat_primary.size());
        allocate(products, flat_primary.size());
        // 轨迹缓冲持久分配，正常run不写不传输；当前容量统计包括它们。
        allocate(constructed, cells); allocate(selections, cells); allocate(selection_uniforms, cells);
        xy.upload(coordinates); d_primary.upload(flat_primary); d_backup.upload(flat_backup);
        d_ls.upload(flat_ls); d_initial.upload(initial);
        checked(cudaDeviceSynchronize());
        preparation_seconds = seconds(started);
    }
};

FixedFacoGpu::FixedFacoGpu(std::vector<double> coordinates, std::uint64_t key, FixedFacoSettings settings)
    : impl_(std::make_unique<Impl>(std::move(coordinates), key, settings)) {}
FixedFacoGpu::~FixedFacoGpu() = default;
const CandidateRows& FixedFacoGpu::primary_candidates() const { return impl_->primary; }
const CandidateRows& FixedFacoGpu::backup_candidates() const { return impl_->backup; }
const CandidateRows& FixedFacoGpu::ls_candidates() const { return impl_->ls; }
const std::vector<Node>& FixedFacoGpu::initial_tour() const { return impl_->initial; }

FixedFacoResult FixedFacoGpu::run_iterations(std::uint64_t seed, Node batches, Node mne_target, bool trace) {
    require(mne_target > 0, "MNE必须为正");
    auto& p = *impl_;
    const auto started = Clock::now();
    const auto& c = p.config;
    std::unique_lock<std::mutex> evaluation_lock(p.evaluation_mutex, std::try_to_lock);
    if (!evaluation_lock.owns_lock()) throw std::runtime_error("同一Engine只允许一个活动评价");
    const auto key = mixed(seed ^ mixed(p.instance_key + 0xd1b54a32d192ed03ULL));
    const cuda_detail::CoordinateDistance distance{p.xy.data()};
    const Node cells = p.n * c.primary_width;
    p.keys.upload({key});
    p.targets.upload(std::vector<Node>(c.ants, mne_target));
    reset_colony<<<1, 128>>>(p.n, p.d_initial.data(), p.initial_costs.data(), p.parent.data(),
        p.epoch.data(), p.global.data(), p.state.data(), c.primary_width, c.retention, c.p_best);
    initialize_products<<<(cells + 255) / 256, 256>>>(distance, cells, p.n, c.primary_width,
        p.d_primary.data(), c.beta, p.state.data(), p.heuristic.data(), p.trails.data(), p.products.data());
    FixedFacoResult result{};
    for (Node batch = 0; batch < batches; ++batch) {
        FixedFacoBatchTrace record{};
        if (trace) {
            record.parent_before = p.parent.download();
            record.trails_before = p.trails.download(); record.products_before = p.products.download();
            record.default_before = p.state.download()[0].default_trail;
            p.selections.zero(); p.selection_uniforms.zero();
        }
        const cuda_detail::StochasticChoices choices{p.d_primary.data(), p.d_backup.data(), p.products.data(),
            c.primary_width, c.backup_width, batch, key,
            trace ? p.selections.data() : nullptr, trace ? p.selection_uniforms.data() : nullptr};
        cuda_detail::construct_and_search<<<c.ants, 128>>>(distance, p.d_ls.data(), p.n, c.ls_width,
            p.parent.data(), choices, p.targets.data(), c.ls_evaluation_limit, p.tours.data(),
            p.positions.data(), p.parent_positions.data(), p.scratch.data(), p.pending.data(),
            p.gains.data(), trace ? p.constructed.data() : nullptr, p.info.data(), p.visited.data());
        reduce_and_select<<<1, 128>>>(p.n, c.ants, p.tours.data(), p.info.data(), p.parent.data(),
            p.parent_position.data(), p.epoch.data(), p.global.data(), p.state.data(), c.primary_width,
            c.retention, c.p_best, c.epoch_source_probability, p.keys.data(), batch);
        update_pheromone<<<(cells + 255) / 256, 256>>>(p.n, c.primary_width, 1, p.d_primary.data(),
            p.parent.data(), p.parent_position.data(), c.retention, p.state.data(), p.heuristic.data(),
            p.trails.data(), p.products.data());
        checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
        const auto info = p.info.download();
        for (const auto& ant : info) {
            result.construction_steps += ant.construction.steps;
            result.ls_evaluations += ant.local_search.move_evaluations;
        }
        if (trace) {
            const auto current = p.state.download()[0];
            record.parent_after = p.parent.download(); record.epoch_best = p.epoch.download();
            record.global_best = p.global.download(); record.trails_after = p.trails.download();
            record.default_after = current.default_trail; record.epoch_cost = current.epoch_cost;
            record.global_cost = current.global_cost; record.minimum = current.minimum;
            record.maximum = current.maximum; record.source_uniform = current.source_uniform;
            record.iteration_best = current.iteration_best; record.source_is_epoch = current.source_is_epoch;
            record.selected_nodes = p.selections.download(); record.selection_uniforms = p.selection_uniforms.download();
            const auto tours = p.tours.download(), positions = p.positions.download(),
                constructed = p.constructed.download(), pending = p.pending.download();
            for (Node ant = 0; ant < c.ants; ++ant) {
                const auto base = static_cast<std::size_t>(ant) * p.n;
                require(info[ant].checklist_size <= 5 * p.n, "设备checklist长度无效");
                record.ants.push_back({{constructed.begin() + base, constructed.begin() + base + p.n},
                    {tours.begin() + base, tours.begin() + base + p.n},
                    {positions.begin() + base, positions.begin() + base + p.n},
                    {pending.begin() + base * 5, pending.begin() + base * 5 + info[ant].checklist_size}, info[ant]});
            }
            result.trace.push_back(std::move(record));
        }
    }
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    result.tour = p.global.download(); result.cost = p.state.download()[0].global_cost;
    result.initial_cost = p.initial_cost; result.preparation_seconds = p.preparation_seconds;
    result.solve_seconds = seconds(started); result.batches = batches;
    result.allocated_device_bytes = p.allocated_bytes;
    return result;
}

}  // namespace gp_faco
