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
#include "gp_faco/batch_engine.hpp"
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

#ifdef GPFACO_CUDA
gp_faco::PreparationMode preparation_mode(const std::string& mode) {
    if (mode == "cached_charged") return gp_faco::PreparationMode::CachedCharged;
    if (mode == "end_to_end") return gp_faco::PreparationMode::EndToEnd;
    throw std::invalid_argument("未知准备模式");
}

py::dict batch_output(const gp_faco::BatchEvaluation& result, const std::string& mode) {
    py::dict output;
    py::list items;
    for (const auto& item : result.incumbents) {
        py::dict entry;
        entry["has_incumbent"] = item.present;
        entry["tour"] = item.tour;
        entry["cost"] = item.present ? py::cast(item.cost) : py::none();
        entry["completed_seconds"] = item.completed_seconds;
        items.append(entry);
    }
    output["items"] = items;
    output["budget_seconds"] = result.budget_seconds;
    output["elapsed_seconds"] = result.elapsed_seconds;
    output["actual_seconds"] = result.actual_seconds;
    output["charged_seconds"] = result.charged_seconds;
    output["last_batch_completed_seconds"] = result.last_batch_completed_seconds;
    output["overrun_seconds"] = result.overrun_seconds;
    output["launched_batches"] = result.launched_batches;
    output["completed_batches"] = result.completed_batches;
    output["discarded_batches"] = result.discarded_batches;
    output["completed_construction_steps"] = result.completed_construction_steps;
    output["completed_ls_evaluations"] = result.completed_ls_evaluations;
    output["allocated_device_bytes"] = result.allocated_device_bytes;
    output["preparation_completed"] = result.preparation_completed;
    output["preparation_mode"] = mode;
    // 只导出最后一个按时完成批次的计数，迟到状态和完整诊断轨迹不进入Python。
    if (!result.completed_control_states.empty()) {
        py::list states;
        for (const auto& control : result.completed_control_states) {
            py::dict state;
            state["archive_size"] = control.archive_size;
            state["stagnant_batches"] = control.feedback.stagnant_batches;
            state["return_rate"] = control.feedback.return_rate;
            state["ls_work"] = control.feedback.ls_work;
            state["epoch_batches"] = control.feedback.epoch_batches;
            state["restarts"] = control.feedback.restarts;
            states.append(state);
        }
        output["control_states"] = states;
    }
    return output;
}
#endif

}  // namespace

PYBIND11_MODULE(gp_faco_ext, module) {
    module.doc() = "GP评分、固定迭代与并发截止FACO接口；在线GP控制在原生Engine内执行";
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
    using Keys = py::array_t<std::uint64_t, py::array::c_style>;
    py::class_<gp_faco::FacoBatchEngine>(module, "FacoBatchEngine")
        .def(py::init([](const py::object& n, const py::object& count,
                         const gp_faco::FixedFacoSettings& settings) {
            if (!PyLong_CheckExact(n.ptr()) || !PyLong_CheckExact(count.ptr()))
                throw std::invalid_argument("dimension/colonies必须为整数");
            const auto dimension = py::cast<gp_faco::Node>(n), colonies = py::cast<gp_faco::Node>(count);
            const auto copied = settings;
            py::gil_scoped_release release;
            return std::make_unique<gp_faco::FacoBatchEngine>(dimension, colonies, copied);
        }), py::arg("dimension"), py::arg("colonies"), py::arg("settings"))
        .def("register_problem", [](gp_faco::FacoBatchEngine& engine, const py::object& key,
                                    const Coordinates& coordinates) {
            if (!PyLong_CheckExact(key.ptr()) || coordinates.ndim() != 2 || coordinates.shape(1) != 2)
                throw std::invalid_argument("需要整数实例key和FP64[n,2]坐标");
            const auto value = py::cast<std::uint64_t>(key);
            std::vector<double> copy(coordinates.data(), coordinates.data() + coordinates.size());
            gp_faco::RegistrationInfo info;
            {
                py::gil_scoped_release release;
                info = engine.register_problem(value, std::move(copy));
            }
            py::dict result;
            result["cheap_seconds"] = info.cheap_seconds;
            result["preparation_seconds"] = info.preparation_seconds;
            return result;
        }, py::arg("instance_key"), py::arg("coordinates").noconvert())
        .def("evaluate", [](gp_faco::FacoBatchEngine& engine, const Keys& keys, const Keys& seeds,
                            const py::object& seconds, const py::object& mne, const std::string& mode) {
            if (keys.ndim() != 1 || seeds.ndim() != 1 || keys.size() != seeds.size() ||
                !(PyFloat_CheckExact(seconds.ptr()) || PyLong_CheckExact(seconds.ptr())) ||
                !PyLong_CheckExact(mne.ptr())) throw std::invalid_argument("任务数组或预算/MNE类型无效");
            const auto preparation = preparation_mode(mode);
            const double budget = py::cast<double>(seconds);
            const auto target = py::cast<gp_faco::Node>(mne);
            std::vector<gp_faco::BatchTask> tasks;
            for (py::ssize_t i = 0; i < keys.size(); ++i) tasks.push_back({keys.data()[i], seeds.data()[i]});
            gp_faco::BatchEvaluation result;
            {
                py::gil_scoped_release release;
                result = engine.evaluate(tasks, budget, target, preparation);
            }
            return batch_output(result, mode);
        }, py::arg("instance_keys").noconvert(), py::arg("seeds").noconvert(),
           py::arg("budget_seconds"), py::arg("mne_target"), py::arg("preparation_mode") = "cached_charged")
        .def("evaluate_program", [](gp_faco::FacoBatchEngine& engine, const Keys& keys, const Keys& seeds,
                                    const py::object& seconds, const py::dict& dictionary,
                                    const std::string& mode, const py::object& mask) {
            if (keys.ndim() != 1 || seeds.ndim() != 1 || keys.size() != seeds.size() ||
                !(PyFloat_CheckExact(seconds.ptr()) || PyLong_CheckExact(seconds.ptr())) ||
                !PyLong_CheckExact(mask.ptr())) throw std::invalid_argument("任务数组或预算/mask类型无效");
            const auto preparation = preparation_mode(mode);
            const auto program = read_program(dictionary);
            const auto experiment_mask = py::cast<std::uint32_t>(mask);
            const double budget = py::cast<double>(seconds);
            std::vector<gp_faco::BatchTask> tasks;
            for (py::ssize_t i = 0; i < keys.size(); ++i) tasks.push_back({keys.data()[i], seeds.data()[i]});
            gp_faco::BatchEvaluation result;
            {
                py::gil_scoped_release release;
                result = engine.evaluate_program(tasks, budget, program, preparation, experiment_mask);
            }
            return batch_output(result, mode);
        }, py::arg("instance_keys").noconvert(), py::arg("seeds").noconvert(),
           py::arg("budget_seconds"), py::arg("program"),
           py::arg("preparation_mode") = "cached_charged", py::arg("experiment_mask") = UINT32_MAX);
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
