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
std::uint64_t read_unsigned(const py::handle& value) {
    if (!PyLong_CheckExact(value.ptr())) throw std::invalid_argument("参数必须为非负整数，不能是布尔值");
    const auto result = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw std::invalid_argument("整数参数超出uint64范围");
    }
    return result;
}

gp_faco::Node read_node(const py::handle& value) {
    const auto result = read_unsigned(value);
    if (result > UINT32_MAX) throw std::invalid_argument("整数参数超出uint32范围");
    return static_cast<gp_faco::Node>(result);
}

gp_faco::BaselinePolicy read_baseline(const py::dict& dictionary) {
    if (dictionary.size() != 11 || read_unsigned(dictionary["policy_spec_id"]) != 1)
        throw std::invalid_argument("基线配置字段或规格版本无效");
    gp_faco::BaselinePolicy p;
    const auto kind = py::cast<std::string>(dictionary["kind"]);
    if (kind == "static") p.kind = gp_faco::BaselineKind::Static;
    else if (kind == "rule") p.kind = gp_faco::BaselineKind::Rule;
    else throw std::invalid_argument("未知基线类别");
    const auto mode = py::cast<std::string>(dictionary["restart_mode"]);
    if (mode == "none") p.restart_mode = gp_faco::StaticRestart::None;
    else if (mode == "periodic") p.restart_mode = gp_faco::StaticRestart::Periodic;
    else if (mode == "bernoulli") p.restart_mode = gp_faco::StaticRestart::Bernoulli;
    else throw std::invalid_argument("未知静态重启模式");
    p.mne_level = read_node(dictionary["mne_level"]);
    p.max_mne_level = read_node(dictionary["max_mne_level"]);
    p.region = read_node(dictionary["region"]);
    p.restart_period = read_node(dictionary["restart_period"]);
    p.stagnation_step = read_node(dictionary["stagnation_step"]);
    p.restart_stagnation = read_node(dictionary["restart_stagnation"]);
    p.restart_cooldown = read_node(dictionary["restart_cooldown"]);
    const auto probability = dictionary["restart_probability"];
    if (!PyFloat_CheckExact(probability.ptr()) && !PyLong_CheckExact(probability.ptr()))
        throw std::invalid_argument("重启概率需要实数，不能是布尔值");
    p.restart_probability = py::cast<double>(probability);
    gp_faco::validate_baseline(p);
    return p;
}

py::dict baseline_output(const gp_faco::BaselinePolicy& p) {
    py::dict result;
    result["policy_spec_id"] = 1;
    result["kind"] = p.kind == gp_faco::BaselineKind::Static ? "static" : "rule";
    result["mne_level"] = p.mne_level; result["max_mne_level"] = p.max_mne_level;
    result["region"] = p.region;
    result["restart_mode"] = p.restart_mode == gp_faco::StaticRestart::None ? "none" :
        p.restart_mode == gp_faco::StaticRestart::Periodic ? "periodic" : "bernoulli";
    result["restart_period"] = p.restart_period; result["restart_probability"] = p.restart_probability;
    result["stagnation_step"] = p.stagnation_step; result["restart_stagnation"] = p.restart_stagnation;
    result["restart_cooldown"] = p.restart_cooldown;
    return result;
}

gp_faco::FactorialPolicy read_factorial(const py::dict& dictionary) {
    if (dictionary.size() != 3 || !dictionary.contains("factorial_spec_id") ||
        !dictionary.contains("variant") || !dictionary.contains("baseline_policy") ||
        read_unsigned(dictionary["factorial_spec_id"]) != 1 ||
        !py::isinstance<py::dict>(dictionary["baseline_policy"]))
        throw std::invalid_argument("析因配置字段或规格版本无效");
    gp_faco::FactorialPolicy result;
    const auto variant = py::cast<std::string>(dictionary["variant"]);
    if (variant == "M00") result.variant = gp_faco::FactorialVariant::M00;
    else if (variant == "M10") result.variant = gp_faco::FactorialVariant::M10;
    else if (variant == "M01") result.variant = gp_faco::FactorialVariant::M01;
    else if (variant == "M11") result.variant = gp_faco::FactorialVariant::M11;
    else throw std::invalid_argument("未知析因变体");
    result.baseline = read_baseline(py::cast<py::dict>(dictionary["baseline_policy"]));
    gp_faco::validate_factorial(result);
    return result;
}

py::dict factorial_output(const gp_faco::FactorialPolicy& policy) {
    py::dict result;
    result["factorial_spec_id"] = 1;
    result["variant"] = policy.variant == gp_faco::FactorialVariant::M00 ? "M00" :
        policy.variant == gp_faco::FactorialVariant::M10 ? "M10" :
        policy.variant == gp_faco::FactorialVariant::M01 ? "M01" : "M11";
    result["baseline_policy"] = baseline_output(policy.baseline);
    return result;
}

gp_faco::PreparationMode preparation_mode(const std::string& mode) {
    if (mode == "cached_charged") return gp_faco::PreparationMode::CachedCharged;
    if (mode == "end_to_end") return gp_faco::PreparationMode::EndToEnd;
    throw std::invalid_argument("未知准备模式");
}

py::dict feedback_output(const gp_faco::FeedbackState& feedback) {
    py::dict output;
    output["return_rate"] = feedback.return_rate;
    output["ls_work"] = feedback.ls_work;
    output["stagnant_batches"] = feedback.stagnant_batches;
    output["epoch_batches"] = feedback.epoch_batches;
    output["restarts"] = feedback.restarts;
    return output;
}

py::dict behavior_output(const gp_faco::BatchEvaluation& result) {
    py::dict output;
    output["behavior_spec_id"] = 1;
    output["device_bytes"] = result.behavior_device_bytes;
    output["host_row_bytes"] = result.behavior_rows.size() * sizeof(gp_faco::BatchBehaviorRow);
    py::list rows;
    for (const auto& row : result.behavior_rows) {
        py::dict item;
        item["batch"] = row.batch; item["colony"] = row.colony;
        item["ants"] = row.ants; item["dimension"] = row.dimension;
        item["alternative"] = row.alternative;
        item["legal_mask"] = row.legal_mask; item["action_mask"] = row.action_mask;
        item["action"] = row.action;
        item["baseline_requested_action"] = row.baseline_requested_action < 0 ? py::none() :
            py::cast(row.baseline_requested_action);
        item["global_before"] = row.global_before; item["reference_before"] = row.reference_before;
        item["reference_used"] = row.reference_used; item["global_after"] = row.global_after;
        item["iteration_best_cost"] = row.iteration_best_cost;
        item["feedback_before"] = feedback_output(row.feedback_before);
        item["feedback_after"] = feedback_output(row.feedback_after);
        item["construction_mne"] = row.construction_mne;
        item["construction_steps"] = row.construction_steps;
        item["construction_relocations"] = row.construction_relocations;
        item["construction_exhausted_ants"] = row.construction_exhausted_ants;
        item["construction_new_edges"] = row.construction_new_edges;
        item["final_new_edges"] = row.final_new_edges;
        item["exact_returns"] = row.exact_returns;
        item["fingerprint_returns"] = row.fingerprint_returns;
        item["ls_move_evaluations"] = row.ls_move_evaluations;
        item["ls_accepted_moves"] = row.ls_accepted_moves;
        item["ls_limit_reached_ants"] = row.ls_limit_reached_ants;
        rows.append(item);
    }
    output["rows"] = rows;
    return output;
}

gp_faco::BatchDiagnosticControls behavior_controls(const py::object& record) {
    if (!PyBool_Check(record.ptr())) throw std::invalid_argument("record_behavior必须是bool");
    gp_faco::BatchDiagnosticControls result;
    result.record_behavior = py::cast<bool>(record);
    return result;
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
    output["budget_seconds"] = result.count_limited ? py::none() : py::cast(result.budget_seconds);
    if (result.count_limited) {
        output["budget_kind"] = "search_tour_evaluations";
        output["evaluation_limit_per_colony"] = result.evaluation_limit_per_colony;
        output["completed_tour_evaluations_per_colony"] = result.completed_tour_evaluations_per_colony;
        output["total_tour_evaluations"] = result.total_tour_evaluations;
    }
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
    if (result.behavior_recorded) output["behavior"] = behavior_output(result);
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

py::dict profiling_output(const gp_faco::EvaluationProfile& profile) {
    py::dict result;
    result["profile_version"] = 1;
    result["setup_seconds"] = profile.setup_seconds;
    result["host_pack_seconds"] = profile.host_pack_seconds;
    result["upload_call_seconds"] = profile.upload_seconds;
    result["initialization_gpu_milliseconds"] = profile.initialization_gpu_milliseconds;
    result["initialization_includes_uploads"] = true;
    result["initialization_download_seconds"] = profile.initialization_download_seconds;
    result["diagnostic_device_bytes"] = profile.diagnostic_device_bytes;
    py::list batches;
    for (const auto& entry : profile.batches) {
        py::dict batch, gpu;
        batch["batch"] = entry.batch; batch["committed"] = entry.committed;
        batch["wall_seconds"] = entry.wall_seconds;
        batch["download_seconds"] = entry.download_seconds;
        batch["verification_seconds"] = entry.verification_seconds;
        batch["collection_seconds"] = entry.collection_seconds;
        for (unsigned i = 0; i < gp_faco::gpu_profile_stage_count; ++i)
            gpu[gp_faco::gpu_profile_stage_names[i]] = entry.gpu_milliseconds[i];
        batch["gpu_milliseconds"] = gpu;
        py::list cycles;
        for (const auto& ant : entry.ant_cycles) cycles.append(py::make_tuple(
            ant.initialization, ant.construction, ant.local_search, ant.finalization));
        batch["ant_phase_cycles"] = cycles;
        batches.append(batch);
    }
    result["batches"] = batches;
    return result;
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
        .def("preparation_profile", [](const gp_faco::FacoBatchEngine& engine, const py::object& key) {
            if (!PyLong_CheckExact(key.ptr())) throw std::invalid_argument("准备剖析需要整数实例key");
            const auto value = py::cast<std::uint64_t>(key);
            gp_faco::PreparationProfile profile;
            { py::gil_scoped_release release; profile = engine.preparation_profile(value); }
            py::dict result;
            result["candidates_and_scales_seconds"] = profile.candidates_and_scales_seconds;
            result["nearest_neighbor_seconds"] = profile.nearest_neighbor_seconds;
            result["initial_ls_seconds"] = profile.initial_ls_seconds;
            result["finalization_seconds"] = profile.finalization_seconds;
            return result;
        }, py::arg("instance_key"))
        .def("set_preparation_charges", [](gp_faco::FacoBatchEngine& engine, const py::object& key,
                                           const py::object& cheap, const py::object& preparation) {
            if (!PyLong_CheckExact(key.ptr()) ||
                !(PyFloat_CheckExact(cheap.ptr()) || PyLong_CheckExact(cheap.ptr())) ||
                !(PyFloat_CheckExact(preparation.ptr()) || PyLong_CheckExact(preparation.ptr())))
                throw std::invalid_argument("冻结费用需要整数key与数值费用");
            const auto value = py::cast<std::uint64_t>(key);
            const gp_faco::RegistrationInfo charges{py::cast<double>(cheap), py::cast<double>(preparation)};
            py::gil_scoped_release release;
            engine.set_preparation_charges(value, charges);
        }, py::arg("instance_key"), py::arg("cheap_seconds"), py::arg("preparation_seconds"))
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
           py::arg("preparation_mode") = "cached_charged", py::arg("experiment_mask") = UINT32_MAX)
        .def("evaluate_program_evaluations", [](gp_faco::FacoBatchEngine& engine,
                const Keys& keys, const Keys& seeds, const py::object& evaluations,
                const py::dict& dictionary, const std::string& mode, const py::object& mask, const py::object& record) {
            if (keys.ndim() != 1 || seeds.ndim() != 1 || keys.size() != seeds.size() ||
                !PyLong_CheckExact(evaluations.ptr()) || !PyLong_CheckExact(mask.ptr()))
                throw std::invalid_argument("次数入口需要整数限额与完整任务数组");
            if (mode != "cached" && mode != "end_to_end")
                throw std::invalid_argument("次数入口只接受cached或end_to_end准备");
            const auto program = read_program(dictionary);
            const auto limit = PyLong_AsUnsignedLongLong(evaluations.ptr());
            if (PyErr_Occurred()) {
                PyErr_Clear();
                throw std::invalid_argument("FE限额必须在uint64非负整数范围内");
            }
            const auto experiment_mask = py::cast<std::uint32_t>(mask);
            const auto preparation = preparation_mode(mode == "cached" ? "cached_charged" : mode);
            std::vector<gp_faco::BatchTask> tasks;
            for (py::ssize_t i = 0; i < keys.size(); ++i) tasks.push_back({keys.data()[i], seeds.data()[i]});
            const auto observation = behavior_controls(record);
            gp_faco::BatchEvaluation result;
            {
                py::gil_scoped_release release;
                result = engine.evaluate_program_evaluations(tasks, limit, program, preparation, experiment_mask, observation);
            }
            return batch_output(result, mode);
        }, py::arg("instance_keys").noconvert(), py::arg("seeds").noconvert(),
           py::arg("evaluation_limit_per_colony"), py::arg("program"),
           py::arg("preparation_mode") = "cached", py::arg("experiment_mask") = UINT32_MAX,
           py::kw_only(), py::arg("record_behavior") = false)
        .def("evaluate_baseline_evaluations", [](gp_faco::FacoBatchEngine& engine,
                const Keys& keys, const Keys& seeds, const py::object& evaluations,
                const py::dict& dictionary, const std::string& mode, const py::object& mask, const py::object& record) {
            if (keys.ndim() != 1 || seeds.ndim() != 1 || keys.size() != seeds.size())
                throw std::invalid_argument("基线次数入口需要完整任务数组");
            if (mode != "cached" && mode != "end_to_end")
                throw std::invalid_argument("次数入口只接受cached或end_to_end准备");
            const auto policy = read_baseline(dictionary);
            const auto limit = read_unsigned(evaluations);
            const auto experiment_mask = read_node(mask);
            const auto preparation = preparation_mode(mode == "cached" ? "cached_charged" : mode);
            std::vector<gp_faco::BatchTask> tasks;
            for (py::ssize_t i = 0; i < keys.size(); ++i) tasks.push_back({keys.data()[i], seeds.data()[i]});
            const auto observation = behavior_controls(record);
            gp_faco::BatchEvaluation result;
            {
                py::gil_scoped_release release;
                result = engine.evaluate_baseline_evaluations(tasks, limit, policy, preparation, experiment_mask, observation);
            }
            auto output = batch_output(result, mode);
            output["baseline_policy"] = baseline_output(policy);
            return output;
        }, py::arg("instance_keys").noconvert(), py::arg("seeds").noconvert(),
           py::arg("evaluation_limit_per_colony"), py::arg("policy"),
           py::arg("preparation_mode") = "cached", py::arg("experiment_mask") = UINT32_MAX,
           py::kw_only(), py::arg("record_behavior") = false)
        .def("evaluate_factorial_evaluations", [](gp_faco::FacoBatchEngine& engine,
                const Keys& keys, const Keys& seeds, const py::object& evaluations,
                const py::dict& dictionary, const py::dict& policy_dictionary,
                const std::string& mode, const py::object& mask, const py::object& record) {
            if (keys.ndim() != 1 || seeds.ndim() != 1 || keys.size() != seeds.size())
                throw std::invalid_argument("析因次数入口需要完整任务数组");
            if (mode != "cached" && mode != "end_to_end")
                throw std::invalid_argument("次数入口只接受cached或end_to_end准备");
            const auto program = read_program(dictionary);
            const auto policy = read_factorial(policy_dictionary);
            const auto limit = read_unsigned(evaluations);
            const auto experiment_mask = read_node(mask);
            const auto preparation = preparation_mode(mode == "cached" ? "cached_charged" : mode);
            std::vector<gp_faco::BatchTask> tasks;
            for (py::ssize_t i = 0; i < keys.size(); ++i) tasks.push_back({keys.data()[i], seeds.data()[i]});
            const auto observation = behavior_controls(record);
            gp_faco::BatchEvaluation result;
            {
                py::gil_scoped_release release;
                result = engine.evaluate_factorial_evaluations(tasks, limit, program, policy,
                                                               preparation, experiment_mask, observation);
            }
            auto output = batch_output(result, mode);
            output["factorial_policy"] = factorial_output(policy);
            return output;
        }, py::arg("instance_keys").noconvert(), py::arg("seeds").noconvert(),
           py::arg("evaluation_limit_per_colony"), py::arg("program"), py::arg("policy"),
           py::arg("preparation_mode") = "cached", py::arg("experiment_mask") = UINT32_MAX,
           py::kw_only(), py::arg("record_behavior") = false)
        .def("run_program_diagnostic", [](gp_faco::FacoBatchEngine& engine, const Keys& keys,
                const Keys& seeds, const py::dict& dictionary, const py::object& batches,
                const py::object& ratio, const py::object& enabled, const py::object& mask,
                const std::string& mode, const py::object& seconds) {
            if (keys.ndim() != 1 || seeds.ndim() != 1 || keys.size() != seeds.size() ||
                !PyLong_CheckExact(batches.ptr()) || !PyBool_Check(enabled.ptr()) ||
                !(PyFloat_CheckExact(ratio.ptr()) || PyLong_CheckExact(ratio.ptr())) ||
                !(PyFloat_CheckExact(seconds.ptr()) || PyLong_CheckExact(seconds.ptr())) ||
                !PyLong_CheckExact(mask.ptr())) throw std::invalid_argument("诊断数组或计时参数类型无效");
            const auto program = read_program(dictionary);
            const auto preparation = preparation_mode(mode);
            gp_faco::BatchDiagnosticControls controls;
            controls.fixed_batches = py::cast<gp_faco::Node>(batches);
            if (!controls.fixed_batches || controls.fixed_batches > 10000)
                throw std::invalid_argument("诊断批次数必须在1..10000");
            controls.fixed_elapsed_ratio = py::cast<double>(ratio);
            controls.profile = py::cast<bool>(enabled);
            const auto experiment_mask = py::cast<std::uint32_t>(mask);
            const auto budget = py::cast<double>(seconds);
            std::vector<gp_faco::BatchTask> tasks;
            for (py::ssize_t i = 0; i < keys.size(); ++i) tasks.push_back({keys.data()[i], seeds.data()[i]});
            gp_faco::BatchEvaluation result;
            {
                py::gil_scoped_release release;
                result = engine.evaluate_program_diagnostic(tasks, budget, program, preparation,
                                                             experiment_mask, controls);
            }
            auto output = batch_output(result, mode);
            output["scope"] = "fixed_batch_diagnostic_not_fitness";
            output["profile"] = profiling_output(result.profile);
            return output;
        }, py::arg("instance_keys").noconvert(), py::arg("seeds").noconvert(), py::arg("program"),
           py::arg("batches"), py::arg("elapsed_ratio") = 0.5, py::arg("profile") = true,
           py::arg("experiment_mask") = UINT32_MAX, py::arg("preparation_mode") = "cached_charged",
           py::arg("budget_seconds") = 120.0);
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
