// 复用已核验Route/LS的固定迭代FACO开发实现。
// 原始FACO语义 Copyright (c) 2024 RSkinderowicz，MIT许可见provenance。
#include "gp_faco/fixed_faco_gpu.hpp"
#include "faco_choices.cuh"
#include "faco_device.cuh"

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

template<class T> class DeviceArray {
public:
    ~DeviceArray() { if (pointer_) cudaFree(pointer_); }
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    void allocate(std::size_t size) {
        require(!pointer_ && count_ == 0, "持久数组不得重复分配");
        count_ = size;
        if (size) checked(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes()));
    }
    T* data() const { return pointer_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
    void upload(const std::vector<T>& input) {
        require(input.size() == count_, "持久数组上传形状不符");
        if (count_) checked(cudaMemcpy(pointer_, input.data(), bytes(), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> result(count_);
        if (count_) checked(cudaMemcpy(result.data(), pointer_, bytes(), cudaMemcpyDeviceToHost));
        return result;
    }
    void zero() { if (count_) checked(cudaMemset(pointer_, 0, bytes())); }
private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

struct Colony {
    double global_cost, epoch_cost, parent_cost, minimum, maximum, default_trail, source_uniform;
    Node iteration_best;
    bool source_is_epoch;
};

__device__ void bounds(Colony& state, Node width, double retention, double p_best) {
    const double p = pow(p_best, 1.0 / width);
    state.maximum = 1.0 / (state.epoch_cost * (1.0 - retention));
    state.minimum = fmin(state.maximum, state.maximum * (1.0 - p) / ((width - 1.0) * p));
}

__global__ void reset_colony(Node n, const Node* initial, double cost, Node* parent,
    Node* epoch, Node* global, Colony* state, Node width, double retention, double p_best) {
    for (Node i = threadIdx.x; i < n; i += blockDim.x) parent[i] = epoch[i] = global[i] = initial[i];
    if (threadIdx.x == 0) {
        *state = {};
        state->global_cost = state->epoch_cost = state->parent_cost = cost;
        bounds(*state, width, retention, p_best);
        state->default_trail = state->maximum;
    }
}

__global__ void initialize_products(cuda_detail::CoordinateDistance distance, Node cells,
    Node width, const Node* primary, double beta, const Colony* state,
    double* heuristic, double* trails, double* products) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) return;
    const double d = distance(i / width, primary[i]);
    heuristic[i] = d > 0 ? 1.0 / pow(d, beta) : 1.0;
    trails[i] = state->maximum;
    products[i] = trails[i] * heuristic[i];
}

__global__ void reduce_and_select(Node n, Node ants, const Node* tours,
    const FacoDiagnosticInfo* info, Node* parent, Node* parent_position, Node* epoch, Node* global,
    Colony* state, Node width, double retention, double p_best, double epoch_probability,
    std::uint64_t seed, Node batch) {
    __shared__ bool improve_epoch, improve_global;
    if (threadIdx.x == 0) {
        Node best = 0;
        for (Node ant = 1; ant < ants; ++ant) if (info[ant].final_cost < info[best].final_cost) best = ant;
        state->iteration_best = best;
        const double cost = info[best].final_cost;
        improve_epoch = cost < state->epoch_cost;
        improve_global = cost < state->global_cost;
        if (improve_epoch) state->epoch_cost = cost;
        if (improve_global) state->global_cost = cost;
        bounds(*state, width, retention, p_best);
        auto random = cuda_detail::random_state(seed, batch, 0xffffffffu, 0);
        state->source_uniform = cuda_detail::uniform53(random);
        state->source_is_epoch = state->source_uniform < epoch_probability;
        state->parent_cost = state->source_is_epoch ? state->epoch_cost : cost;
    }
    __syncthreads();
    const Node* best = tours + static_cast<std::size_t>(state->iteration_best) * n;
    for (Node i = threadIdx.x; i < n; i += blockDim.x) {
        if (improve_epoch) epoch[i] = best[i];
        if (improve_global) global[i] = best[i];
        const Node node = state->source_is_epoch ? epoch[i] : best[i];
        parent[i] = node;
        parent_position[node] = i;
    }
}

__global__ void update_pheromone(Node n, Node width, const Node* primary,
    const Node* parent, const Node* position, double retention, Colony* state,
    const double* heuristic, double* trails, double* products) {
    const Node i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    const Node from = i / width, to = primary[i], pos = position[from];
    double value = fmax(state->minimum, trails[i] * retention);
    // 每个有向条目独占写入；对称强化匹配tour的前驱或后继，避免原子竞争。
    if (parent[(pos + 1) % n] == to || parent[(pos + n - 1) % n] == to) {
        value = fmin(state->maximum, value + 1.0 / state->parent_cost);
    }
    trails[i] = value;
    products[i] = value * heuristic[i];
    if (i == 0) state->default_trail = fmax(state->minimum, state->default_trail * retention);
}

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
    DeviceArray<double> xy, heuristic, trails, products, gains, selection_uniforms;
    DeviceArray<Node> d_primary, d_backup, d_ls, d_initial, parent, parent_position,
        epoch, global, targets, tours, positions, parent_positions, scratch, pending,
        constructed, selections;
    DeviceArray<std::uint8_t> visited;
    DeviceArray<FacoDiagnosticInfo> info;
    DeviceArray<Colony> state;

    Impl(std::vector<double> input, std::uint64_t key, FixedFacoSettings settings)
        : config(settings), n(input.size() / 2), instance_key(key), coordinates(std::move(input)) {
        const auto started = Clock::now();
        require(coordinates.size() % 2 == 0 && n >= 3 && n <= 10000, "坐标形状或节点数无效");
        require(config.ants > 0 && config.ants <= 4096 && config.primary_width >= 2 &&
                config.ls_width > 0 && std::isfinite(config.beta) && config.beta > 0 &&
                config.retention >= 0 && config.retention < 1 && config.p_best > 0 &&
                config.p_best < 1 && config.epoch_source_probability >= 0 &&
                config.epoch_source_probability <= 1, "固定FACO配置无效");
        for (double coordinate : coordinates) require(std::isfinite(coordinate), "坐标必须有限");
        auto distance = [&](Node a, Node b) {
            const double x = coordinates[a * 2] - coordinates[b * 2];
            const double y = coordinates[a * 2 + 1] - coordinates[b * 2 + 1];
            return std::sqrt(x * x + y * y);
        };
        config.primary_width = std::min(config.primary_width, n - 1);
        config.backup_width = std::min(config.backup_width, n - 1 - config.primary_width);
        config.ls_width = std::min(config.ls_width, n - 1);
        primary.resize(n); backup.resize(n); ls.resize(n);
        std::vector<Node> flat_primary, flat_backup, flat_ls;
        const Node needed = std::max(config.primary_width + config.backup_width, config.ls_width);
        // O(nk)持久存储；只保留一行O(n)临时距离，没有n²矩阵。
        for (Node a = 0; a < n; ++a) {
            std::vector<std::pair<double, Node>> row;
            row.reserve(n - 1);
            for (Node b = 0; b < n; ++b) if (a != b) {
                const double d = distance(a, b);
                require(std::isfinite(d), "坐标距离溢出");
                row.emplace_back(d, b);
            }
            std::partial_sort(row.begin(), row.begin() + needed, row.end());
            for (Node j = 0; j < config.primary_width; ++j) primary[a].push_back(row[j].second);
            for (Node j = config.primary_width; j < config.primary_width + config.backup_width; ++j)
                backup[a].push_back(row[j].second);
            for (Node j = 0; j < config.ls_width; ++j) ls[a].push_back(row[j].second);
            flat_primary.insert(flat_primary.end(), primary[a].begin(), primary[a].end());
            flat_backup.insert(flat_backup.end(), backup[a].begin(), backup[a].end());
            flat_ls.insert(flat_ls.end(), ls[a].begin(), ls[a].end());
        }
        // 开发版固定初始解：从节点0最近邻构造，再用同一checklist 2-opt改进。
        std::vector<bool> seen(n, false);
        Node current = 0;
        for (Node i = 0; i < n; ++i) {
            initial.push_back(current); seen[current] = true;
            Node next = n; double smallest = std::numeric_limits<double>::infinity();
            for (Node b = 0; b < n; ++b) if (!seen[b] && distance(current, b) < smallest) {
                smallest = distance(current, b); next = b;
            }
            current = next;
        }
        CpuTour tour(initial, distance);
        std::vector<Node> checklist(initial);
        tour.checklist_two_opt(DistanceOrderedCandidates(ls, distance), checklist, config.initial_ls_evaluation_limit);
        initial = tour.order(); initial_cost = tour.recomputed_cost();
        require(std::isfinite(initial_cost) && initial_cost > 0, "初始路线成本无效");

        const std::size_t cells = static_cast<std::size_t>(n) * config.ants;
        const auto allocate = [&](auto& buffer, std::size_t count) {
            buffer.allocate(count); allocated_bytes += buffer.bytes();
        };
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
    p.targets.upload(std::vector<Node>(c.ants, mne_target));
    reset_colony<<<1, 128>>>(p.n, p.d_initial.data(), p.initial_cost, p.parent.data(),
        p.epoch.data(), p.global.data(), p.state.data(), c.primary_width, c.retention, c.p_best);
    initialize_products<<<(cells + 255) / 256, 256>>>(distance, cells, c.primary_width,
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
            c.retention, c.p_best, c.epoch_source_probability, key, batch);
        update_pheromone<<<(cells + 255) / 256, 256>>>(p.n, c.primary_width, p.d_primary.data(),
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
