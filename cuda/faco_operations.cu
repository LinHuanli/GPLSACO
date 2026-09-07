// FACO Route/LS语义依据RSkinderowicz（Copyright (c) 2024，MIT）。
// 完整许可见 provenance/licenses/Adaptive-Tuning-MIT.txt。
// 本阶段只实现显式选点的操作诊断，不作为完整搜索/性能基准。
#include "gp_faco/faco_cuda_diagnostic.hpp"
#include "faco_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gp_faco {
namespace {

void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> class Buffer {
public:
    explicit Buffer(std::size_t size) : size_(size) {
        checked(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }
    ~Buffer() { cudaFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() const { return data_; }
    void upload(const std::vector<T>& values) {
        if (values.size() != size_) throw std::invalid_argument("诊断上传形状不符");
        checked(cudaMemcpy(data_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> values(size_);
        checked(cudaMemcpy(values.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
private:
    std::size_t size_;
    T* data_ = nullptr;
};


void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void permutation(const std::vector<Node>& values, Node n) {
    require(values.size() == n, "诊断tour/选点序列的维数不符");
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    for (Node i = 0; i < n; ++i) require(sorted[i] == i, "诊断tour/选点序列不是完整排列");
}

}  // namespace

std::vector<FacoDiagnosticResult> cuda_faco_diagnostic(
    const std::vector<double>& distances, const CandidateRows& ls_candidates,
    const std::vector<FacoDiagnosticTask>& tasks, std::uint64_t evaluation_limit) {
    const auto n = static_cast<Node>(ls_candidates.size());
    // 限定显式矩阵诊断的规模，避免它意外成为10K Engine的表示。
    require(n >= 3 && n <= 1024 && !tasks.empty() && tasks.size() <= 256,
            "诊断仅支持3..1024节点、1..256蚂蚁");
    require(distances.size() == static_cast<std::size_t>(n) * n, "诊断距离矩阵形状不符");
    const Node width = ls_candidates[0].size();
    require(width > 0 && width < n, "诊断候选宽度无效");
    for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b) {
        const double d = distances[a * n + b];
        require(std::isfinite(d) && d >= 0 && d == distances[b * n + a], "距离必须有限、非负且对称");
    }
    const auto distance_fn = [&](Node a, Node b) { return distances[a * n + b]; };
    DistanceOrderedCandidates sorted(ls_candidates, distance_fn);
    std::vector<Node> candidates, tours, visits, targets;
    for (Node a = 0; a < n; ++a) {
        require(sorted[a].size() == width, "诊断候选行必须等宽");
        candidates.insert(candidates.end(), sorted[a].begin(), sorted[a].end());
    }
    for (const auto& task : tasks) {
        permutation(task.tour, n);
        permutation(task.visit_order, n);
        require(task.mne_target > 0, "诊断MNE必须为正");
        tours.insert(tours.end(), task.tour.begin(), task.tour.end());
        visits.insert(visits.end(), task.visit_order.begin(), task.visit_order.end());
        targets.push_back(task.mne_target);
    }
    const std::size_t cells = n * tasks.size();
    Buffer<double> d_distances(distances.size()), d_gains(tasks.size() * width * 2);
    Buffer<Node> d_candidates(candidates.size()), d_parents(cells), d_visits(cells),
        d_targets(tasks.size()), d_tours(cells), d_positions(cells), d_parent_positions(cells),
        d_scratch(cells), d_pending(cells * 5), d_constructed(cells);
    Buffer<FacoDiagnosticInfo> d_info(tasks.size());
    Buffer<std::uint8_t> d_visited(cells);
    d_distances.upload(distances); d_candidates.upload(candidates);
    d_parents.upload(tours); d_visits.upload(visits); d_targets.upload(targets);
    cuda_detail::construct_and_search<<<tasks.size(), 128>>>(
        cuda_detail::MatrixDistance{d_distances.data(), n}, d_candidates.data(), n, width, d_parents.data(), cuda_detail::ExplicitChoices{d_visits.data()},
        d_targets.data(), evaluation_limit, d_tours.data(), d_positions.data(),
        d_parent_positions.data(), d_scratch.data(), d_pending.data(), d_gains.data(),
        d_constructed.data(), d_info.data(), d_visited.data());
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    const auto final_tours = d_tours.download(), final_positions = d_positions.download(),
        constructed = d_constructed.download(), pending = d_pending.download();
    const auto info = d_info.download();
    std::vector<FacoDiagnosticResult> result;
    for (std::size_t ant = 0; ant < tasks.size(); ++ant) {
        const auto base = ant * n;
        require(info[ant].checklist_size <= 5 * n, "设备checklist长度无效");
        result.push_back({
            {constructed.begin() + base, constructed.begin() + base + n},
            {final_tours.begin() + base, final_tours.begin() + base + n},
            {final_positions.begin() + base, final_positions.begin() + base + n},
            {pending.begin() + base * 5, pending.begin() + base * 5 + info[ant].checklist_size},
            info[ant]});
    }
    return result;
}

}  // namespace gp_faco
