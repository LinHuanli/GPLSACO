// Hard真实选点诊断：复用生产StochasticChoices，不使用另一个测试版选择器。
#include "gp_faco/hard_diagnostic.hpp"
#include "gp_faco/prepared_problem.hpp"
#include "colony_state.cuh"
#include "faco_device.cuh"

#include <cmath>

namespace gp_faco {
namespace {

__global__ void select_hard(Node n, Node count, const double* distances, const Node* primary,
    const Node* backup, const double* products, Node width, Node backup_width,
    SparseGraphView graph, const Node* tours, const Node* positions, const std::uint8_t* visited,
    const Node* current, const std::uint64_t* seeds, HardSelectionResult* result) {
    const Node task = blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= count) return;
    const auto base = static_cast<std::size_t>(task) * n;
    const cuda_detail::StochasticChoices choices{primary, backup, products, width, backup_width,
        11, seeds[task], nullptr, nullptr};
    result[task].selected = choices.next(0, current[task], visited + base, 1, n,
        cuda_detail::MatrixDistance{distances, n}, tours + base, positions + base, graph);
    auto random = cuda_detail::random_state(seeds[task], 11, 0, 1);
    result[task].uniform = cuda_detail::uniform53(random);
}

}  // namespace

std::vector<HardSelectionResult> cuda_hard_selection_diagnostic(
    const std::vector<double>& matrix, const CandidateRows& primary, const CandidateRows& backup,
    const std::vector<double>& products, const SparseUndirectedGraph& graph,
    const std::vector<HardSelectionTask>& tasks) {
    using cuda_detail::DeviceArray;
    using cuda_detail::require;
    using cuda_detail::checked;
    const Node n = graph.size();
    require(n <= 1024 && primary.size() == n && backup.size() == n &&
            !tasks.empty() && tasks.size() <= 4096 && matrix.size() == static_cast<std::size_t>(n) * n,
            "Hard选点诊断输入形状无效");
    const Node width = primary[0].size(), backup_width = backup[0].size();
    require(width > 0 && width < n && backup_width < n && products.size() == n * width,
            "Hard选点诊断候选宽度无效");
    for (Node a = 0; a < n; ++a) for (Node b = 0; b < n; ++b)
        require(std::isfinite(matrix[a * n + b]) && matrix[a * n + b] >= 0 &&
                matrix[a * n + b] == matrix[b * n + a], "距离必须有限非负对称");
    for (double product : products) require(std::isfinite(product) && product >= 0, "产品必须非负有限");
    const auto distance = [&](Node a, Node b) { return matrix[a * n + b]; };
    // 校验成员但不使用排序结果，保持输入的主候选/备用先验顺序。
    DistanceOrderedCandidates(primary, distance); DistanceOrderedCandidates(backup, distance);
    for (Node a = 0; a < n; ++a)
        require(primary[a].size() == width && backup[a].size() == backup_width, "候选行必须等宽");
    std::vector<Node> tours, positions, current;
    std::vector<std::uint8_t> visited;
    std::vector<std::uint64_t> seeds;
    for (const auto& task : tasks) {
        CpuTour tour(task.tour, distance);
        require(graph.contains_tour(task.tour) && task.visited.size() == n && task.current < n &&
                task.visited[task.current] == 1, "Hard选点状态无效");
        for (auto v : task.visited) require(v <= 1, "visited必须为0/1");
        tours.insert(tours.end(), task.tour.begin(), task.tour.end());
        positions.insert(positions.end(), tour.positions().begin(), tour.positions().end());
        visited.insert(visited.end(), task.visited.begin(), task.visited.end());
        current.push_back(task.current); seeds.push_back(task.seed);
    }
    DeviceArray<double> d_matrix, d_products;
    DeviceArray<Node> d_primary, d_backup, d_tours, d_positions, d_current, d_offsets, d_neighbors;
    DeviceArray<std::uint8_t> d_visited;
    DeviceArray<std::uint64_t> d_seeds;
    DeviceArray<HardSelectionResult> d_result;
    const auto upload = [](auto& buffer, const auto& values) {
        buffer.allocate(values.size()); buffer.upload(values);
    };
    upload(d_matrix, matrix); upload(d_products, products);
    upload(d_primary, flattened(primary)); upload(d_backup, flattened(backup));
    upload(d_tours, tours); upload(d_positions, positions); upload(d_current, current);
    upload(d_offsets, graph.offsets()); upload(d_neighbors, graph.neighbors());
    upload(d_visited, visited); upload(d_seeds, seeds); d_result.allocate(tasks.size());
    select_hard<<<(tasks.size() + 127) / 128, 128>>>(n, tasks.size(), d_matrix.data(), d_primary.data(),
        d_backup.data(), d_products.data(), width, backup_width,
        {d_offsets.data(), d_neighbors.data(), n}, d_tours.data(), d_positions.data(),
        d_visited.data(), d_current.data(), d_seeds.data(), d_result.data());
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    return d_result.download();
}

}  // namespace gp_faco
