#include "gp_faco/program.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>

#ifdef GPFACO_CUDA
#include <cuda_runtime.h>
#include "gp_faco/fixed_faco_gpu.hpp"
#endif

namespace py = pybind11;

namespace {

gp_faco::Program read_program(const py::dict& dictionary) {
    if (dictionary.size() != 6) throw std::invalid_argument("IR 字段数量错误");
    // JSON bool 与整数在 Python 中有继承关系，不能让 true 冒充版本 1。
    for (const char* name : {"ir_version", "numeric_spec_id", "feature_spec_id"}) {
        if (!PyLong_CheckExact(dictionary[name].ptr())) {
            throw std::invalid_argument("IR 版本必须为整数");
        }
    }
    for (const char* name : {"opcode", "operand", "constant_bits"}) {
        const auto sequence = py::cast<py::sequence>(dictionary[name]);
        for (const auto& value : sequence) {
            if (!PyLong_CheckExact(value.ptr())) {
                throw std::invalid_argument("IR 操作码和位模式必须为整数");
            }
        }
    }
    gp_faco::Program program;
    program.ir_version = py::cast<std::uint16_t>(dictionary["ir_version"]);
    program.numeric_spec_id = py::cast<std::uint16_t>(dictionary["numeric_spec_id"]);
    program.feature_spec_id = py::cast<std::uint16_t>(dictionary["feature_spec_id"]);
    const auto opcodes = py::cast<std::vector<std::uint8_t>>(dictionary["opcode"]);
    const auto operands = py::cast<std::vector<std::uint8_t>>(dictionary["operand"]);
    const auto constants = py::cast<std::vector<std::uint32_t>>(dictionary["constant_bits"]);
    if (opcodes.empty() || opcodes.size() > 63 || operands.size() != opcodes.size() ||
        constants.size() > 63) throw std::invalid_argument("IR 数组大小错误");
    program.length = static_cast<std::uint16_t>(opcodes.size());
    program.constants_count = static_cast<std::uint16_t>(constants.size());
    std::copy(opcodes.begin(), opcodes.end(), program.opcode);
    std::copy(operands.begin(), operands.end(), program.operand);
    for (std::size_t i = 0; i < constants.size(); ++i) {
        std::memcpy(&program.constants[i], &constants[i], sizeof(float));
    }
    gp_faco::validate_program(program);
    return program;
}

using FeatureArray = py::array_t<float, py::array::c_style>;
using MaskArray = py::array_t<std::uint32_t, py::array::c_style>;

py::dict score(const py::dict& dictionary, const FeatureArray& features,
               const MaskArray& masks, bool use_cuda) {
    const auto program = read_program(dictionary);
    if (features.ndim() != 3 || features.shape(0) != 12 || features.shape(2) != 32 ||
        masks.ndim() != 1 || masks.shape(0) != features.shape(1)) {
        throw std::invalid_argument("需要连续 FP32 [12,colony,32] 和 uint32 [colony]");
    }
    // 诊断接口明确复制到 C++ 自有内存；持有 GIL 验参后再进入纯原生逻辑。
    const std::vector<float> values(features.data(), features.data() + features.size());
    const std::vector<std::uint32_t> legality(masks.data(), masks.data() + masks.size());
    gp_faco::Scores result;
    {
        py::gil_scoped_release release;
        if (use_cuda) {
#ifdef GPFACO_CUDA
            result = gp_faco::score_cuda(program, values, legality);
#else
            throw std::runtime_error("此构建未启用 CUDA");
#endif
        } else {
            result = gp_faco::score_cpu(program, values, legality);
        }
    }
    const auto count = static_cast<py::ssize_t>(legality.size());
    py::array_t<float> scores({count, static_cast<py::ssize_t>(32)});
    py::array_t<std::int32_t> actions(count);
    std::copy(result.scores.begin(), result.scores.end(), scores.mutable_data());
    std::copy(result.actions.begin(), result.actions.end(), actions.mutable_data());
    py::dict output;
    output["scores"] = scores;
    output["actions"] = actions;
    return output;
}

}  // namespace

PYBIND11_MODULE(gp_faco_ext, module) {
    module.doc() = "GP评分与固定迭代FACO开发接口；deadline/GP Engine尚未完成";
    module.def("score_cpu", [](const py::dict& p, const FeatureArray& f, const MaskArray& m) {
        return score(p, f, m, false);
    }, py::arg("program"), py::arg("features").noconvert(), py::arg("masks").noconvert());
#ifdef GPFACO_CUDA
    py::class_<gp_faco::FixedFacoSettings>(module, "FixedFacoSettings")
        .def(py::init<>())
        .def_readwrite("ants", &gp_faco::FixedFacoSettings::ants)
        .def_readwrite("primary_width", &gp_faco::FixedFacoSettings::primary_width)
        .def_readwrite("backup_width", &gp_faco::FixedFacoSettings::backup_width)
        .def_readwrite("ls_width", &gp_faco::FixedFacoSettings::ls_width)
        .def_readwrite("beta", &gp_faco::FixedFacoSettings::beta)
        .def_readwrite("retention", &gp_faco::FixedFacoSettings::retention)
        .def_readwrite("p_best", &gp_faco::FixedFacoSettings::p_best)
        .def_readwrite("epoch_source_probability", &gp_faco::FixedFacoSettings::epoch_source_probability)
        .def_readwrite("ls_evaluation_limit", &gp_faco::FixedFacoSettings::ls_evaluation_limit)
        .def_readwrite("initial_ls_evaluation_limit", &gp_faco::FixedFacoSettings::initial_ls_evaluation_limit);
    using Coordinates = py::array_t<double, py::array::c_style>;
    py::class_<gp_faco::FixedFacoGpu>(module, "FixedFacoGpu")
        .def(py::init([](const Coordinates& coordinates, const py::object& key,
                          const gp_faco::FixedFacoSettings& settings) {
            if (coordinates.ndim() != 2 || coordinates.shape(1) != 2 ||
                !PyLong_CheckExact(key.ptr())) throw std::invalid_argument("需要FP64[n,2]与整数实例key");
            const auto instance_key = py::cast<std::uint64_t>(key);
            const auto copied_settings = settings;
            std::vector<double> values(coordinates.data(), coordinates.data() + coordinates.size());
            py::gil_scoped_release release;
            return std::make_unique<gp_faco::FixedFacoGpu>(std::move(values), instance_key, copied_settings);
        }), py::arg("coordinates").noconvert(), py::arg("instance_key"), py::arg("settings"))
        .def("run_iterations", [](gp_faco::FixedFacoGpu& engine, const py::object& seed,
                                  const py::object& batches, const py::object& mne) {
            for (const auto& value : {seed, batches, mne}) {
                if (!PyLong_CheckExact(value.ptr())) throw std::invalid_argument("seed/batches/MNE必须为整数");
            }
            const auto seed_value = py::cast<std::uint64_t>(seed);
            const auto batch_count = py::cast<gp_faco::Node>(batches);
            const auto target = py::cast<gp_faco::Node>(mne);
            gp_faco::FixedFacoResult result;
            {
                py::gil_scoped_release release;
                result = engine.run_iterations(seed_value, batch_count, target);
            }
            py::dict output;
            output["tour"] = result.tour; output["cost"] = result.cost;
            output["initial_cost"] = result.initial_cost;
            output["preparation_seconds"] = result.preparation_seconds;
            output["solve_seconds"] = result.solve_seconds;
            output["batches"] = result.batches;
            output["construction_steps"] = result.construction_steps;
            output["ls_evaluations"] = result.ls_evaluations;
            output["allocated_device_bytes"] = result.allocated_device_bytes;
            return output;
        }, py::arg("seed"), py::arg("batches"), py::arg("mne_target"));
    module.def("score_cuda", [](const py::dict& p, const FeatureArray& f, const MaskArray& m) {
        return score(p, f, m, true);
    }, py::arg("program"), py::arg("features").noconvert(), py::arg("masks").noconvert());
    module.def("cuda_device_info", []() {
        cudaDeviceProp properties{};
        int device = 0;
        auto status = cudaGetDevice(&device);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        py::dict output;
        output["name"] = properties.name;
        output["compute_capability_major"] = properties.major;
        output["compute_capability_minor"] = properties.minor;
        output["total_memory_bytes"] = properties.totalGlobalMem;
        return output;
    });
#endif
}
