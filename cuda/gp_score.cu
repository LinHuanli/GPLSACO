#include "gp_faco/program.hpp"
#include "gp_faco/program_ops.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace gp_faco {
namespace {

void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
struct DeviceBuffer {
    T* pointer = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        checked(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
    }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void score_kernel(Program program, const float* features, const std::uint32_t* masks,
                             float* scores, std::int32_t* actions, int colonies) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int colony = thread / 32;
    const int lane = thread % 32;
    // 整个 warp 同时退出；有效 warp 的所有 lane 都参与后续 collective。
    if (colony >= colonies) return;
    float stack[6]{};
    int top = 0;
    for (int i = 0; i < program.length; ++i) {
        const int op = program.opcode[i];
        if (op == 0) {
            stack[top++] = features[(program.operand[i] * colonies + colony) * 32 + lane];
        } else if (op == 1) {
            stack[top++] = program.constants[program.operand[i]];
        } else {
            const float right = op == 7 ? 0.0f : stack[--top];
            const float left = stack[--top];
            stack[top++] = apply_operation(op, left, right);
        }
    }
    const float value = stack[0];
    scores[colony * 32 + lane] = value;
    const bool legal = (masks[colony] >> lane) & 1u;
    float maximum = legal ? value : -CUDART_INF_F;
    for (int offset = 16; offset; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    int selected = legal && maximum - value <= 1e-6f ? lane : 32;
    for (int offset = 16; offset; offset /= 2) {
        const int other = __shfl_down_sync(0xffffffffu, selected, offset);
        selected = selected < other ? selected : other;
    }
    if (lane == 0) actions[colony] = selected;
}

}  // namespace

Scores score_cuda(const Program& program, const std::vector<float>& features,
                  const std::vector<std::uint32_t>& masks) {
    validate_program(program);
    validate_features(features, masks);
    if (masks.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 384)) {
        throw std::invalid_argument("诊断 batch 太大");
    }
    Scores output{std::vector<float>(masks.size() * 32),
                  std::vector<std::int32_t>(masks.size())};
    DeviceBuffer<float> d_features(features.size()), d_scores(output.scores.size());
    DeviceBuffer<std::uint32_t> d_masks(masks.size());
    DeviceBuffer<std::int32_t> d_actions(masks.size());
    checked(cudaMemcpy(d_features.pointer, features.data(), features.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
    checked(cudaMemcpy(d_masks.pointer, masks.data(), masks.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice));
    score_kernel<<<(masks.size() + 3) / 4, 128>>>(
        program, d_features.pointer, d_masks.pointer, d_scores.pointer,
        d_actions.pointer, static_cast<int>(masks.size()));
    checked(cudaGetLastError());
    checked(cudaMemcpy(output.scores.data(), d_scores.pointer,
                       output.scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(output.actions.data(), d_actions.pointer,
                       output.actions.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    return output;
}

}  // namespace gp_faco
