// 容量极限直接在设备上构造；不依赖随机搜索恰好填满64条缓存。
#include "../../cuda/escape_device.cuh"
#include <algorithm>
#include <array>
#include <set>

namespace {
using namespace gp_faco;
struct Observation {
    EscapeProposalStatus status;
    Node proposal_count, final_size;
    bool committed, read_only, expired_rejected;
    std::uint64_t epoch;
    std::uint32_t random[6][4];
};

__global__ void primitives(Observation* output) {
    const Node id = threadIdx.x;
    if (id >= 50) return;
    const Node lengths[5]{0, 61, 62, 63, 64};
    const Node length = lengths[id / 10], variant = (id / 2) % 5;
    const bool permit = id % 2;
    Node offsets[101]{};
    const SparseGraphView graph{offsets, nullptr, 100};
    EscapeCache cache{}; cache.reset();
    for (Node j = 0; j < length; ++j) cache.edges[cache.size++] = {0, j + 1};
    const MoveEdge removed[3]{{4, 5}, {5, 6}, {6, 7}};
    MoveEdge added[3]{{70, 71}, {72, 73}, {74, 75}};
    if (variant == 1) added[1] = {71, 70};
    if (variant == 2) added[1] = added[2] = {71, 70};
    if (variant == 3) for (Node j = 0; j < 3; ++j) added[j] = removed[j];
    if (variant == 4) added[0] = {5, 4};
    auto proposal = prepare_escape_edges(removed, added, graph, cache, permit);
    auto& result = output[id]; result = {};
    result.read_only = cache.size == length && cache.epoch == 1;
    for (Node j = 0; j < 64; ++j) result.read_only &= cache.edges[j].a == 0 && cache.edges[j].b == (j < length ? j + 1 : 0);
    result.status = proposal.status; result.proposal_count = proposal.count;
    result.committed = commit_escape_edges(cache, proposal);
    result.final_size = cache.size;
    cache.reset();
    result.expired_rejected = !commit_escape_edges(cache, proposal) && cache.size == 0;
    result.epoch = cache.epoch;
    // 六种随机域/事件不同；同一元组由相邻奇偶线程重复，检查可重放性。
    for (Node domain = 0; domain < 6; ++domain) {
        if (!domain) {
            auto random = cuda_detail::random_state(991, 17, id / 2, 5);
            for (Node j = 0; j < 4; ++j) result.random[domain][j] = curand(&random);
        } else {
            cuda_detail::EscapeRandom random(991, domain == 5 ? 18 : 17,
                id / 2 + (domain == 4 ? 100 : 0), domain == 3 ? 6 : 5, domain == 2);
            for (Node j = 0; j < 4; ++j) result.random[domain][j] = random.draw();
        }
    }
}
}

void verify_escape_cuda_primitives() {
    gp_faco::cuda_detail::DeviceArray<Observation> result; result.allocate(50);
    primitives<<<1, 128>>>(result.data());
    gp_faco::cuda_detail::checked(cudaGetLastError());
    gp_faco::cuda_detail::checked(cudaDeviceSynchronize());
    const auto values = result.download();
    const unsigned lengths[5]{0, 61, 62, 63, 64}, genuinely_new[5]{3, 2, 1, 0, 2};
    std::set<std::array<std::uint32_t, 4>> streams;
    for (unsigned i = 0; i < 50; ++i) {
        const auto& value = values[i];
        const auto count = genuinely_new[(i / 2) % 5], length = lengths[i / 10];
        const bool expected = count == 0 || ((i % 2) && length + count <= 64);
        const auto status = expected ? gp_faco::EscapeProposalStatus::Allowed : i % 2
            ? gp_faco::EscapeProposalStatus::Capacity : gp_faco::EscapeProposalStatus::OutsideGraph;
        if (value.committed != expected || value.status != status || !value.read_only || !value.expired_rejected ||
            value.epoch != 2 || value.final_size != length + (expected ? count : 0) ||
            ((expected || i % 2) && value.proposal_count != count))
            throw std::runtime_error("CUDA例外容量/去重/原子拒绝/失效与独立预期不一致");
        for (unsigned domain = 0; domain < 6; ++domain) {
            std::array<std::uint32_t, 4> prefix;
            std::copy_n(value.random[domain], 4, prefix.begin());
            if (i % 2) {
                if (!std::equal(prefix.begin(), prefix.end(), values[i - 1].random[domain]))
                    throw std::runtime_error("Escape随机流同元组不可重放");
            } else if (!streams.insert(prefix).second) throw std::runtime_error("声明分离的随机流发生样本碰撞");
        }
    }
}
