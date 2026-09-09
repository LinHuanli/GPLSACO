#include "gp_faco/program.hpp"
#include "gp_score.cuh"

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


}  // namespace

Scores score_controller_cuda(const ControllerProgram& controller, const std::vector<float>& features,
                             const std::vector<std::uint32_t>& masks) {
    validate_controller(controller); validate_features(features, masks);
    if (!controller.kind) return score_cuda(controller.trees[0], features, masks);
    Scores output{std::vector<float>(masks.size() * 32), std::vector<std::int32_t>(masks.size())};
    DeviceBuffer<float> f(features.size()), s(output.scores.size());
    DeviceBuffer<std::uint32_t> m(masks.size());
    DeviceBuffer<std::int32_t> a(masks.size());
    DeviceBuffer<ControllerProgram> c(1);
    checked(cudaMemcpy(f.pointer, features.data(), features.size()*sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(m.pointer, masks.data(), masks.size()*sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(c.pointer, &controller, sizeof(controller), cudaMemcpyHostToDevice));
    cuda_detail::score_controllers<<<(masks.size()+3)/4,128>>>(c.pointer, f.pointer, m.pointer,
        s.pointer, a.pointer, masks.size(), masks.size());
    checked(cudaGetLastError());
    checked(cudaMemcpy(output.scores.data(), s.pointer, output.scores.size()*sizeof(float), cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(output.actions.data(), a.pointer, output.actions.size()*sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    return output;
}

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
    cuda_detail::score_actions<<<(masks.size() + 3) / 4, 128>>>(
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
