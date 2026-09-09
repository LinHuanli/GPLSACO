// 相对构造开始reference的精确无向新边数，仅写独立观测缓冲。
#pragma once

#include "gp_faco/faco_cpu.hpp"

namespace gp_faco::cuda_detail {

__device__ inline Node observed_new_edges(const Node* tour, const Node* parent_positions, Node n) {
    __shared__ Node counts[128];
    Node count = 0;
    for (Node i = threadIdx.x; i < n; i += blockDim.x) {
        const Node a = parent_positions[tour[i]], b = parent_positions[tour[(i + 1) % n]];
        const Node separation = a > b ? a - b : b - a;
        count += separation != 1 && separation != n - 1;
    }
    counts[threadIdx.x] = count;
    __syncthreads();
    for (Node stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    return counts[0];
}

}  // namespace gp_faco::cuda_detail
