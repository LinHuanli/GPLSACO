#!/usr/bin/env python3
"""不启动CUDA、不读取标签：逐条件审计成本矩阵并生成分层资源表与待校准预算。"""

import argparse
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402

GPU_PHASES = (
    "initialization",
    "features_and_regions",
    "gp_scoring",
    "action_and_restart",
    "construction_and_ls",
    "reduction",
    "pheromone",
    "ant_fingerprints",
    "archive_and_feedback",
)
CYCLE_PHASES = ("initialization", "construction", "local_search", "finalization")
PREPARATION_PHASES = (
    "candidates_and_scales_seconds",
    "nearest_neighbor_seconds",
    "initial_ls_seconds",
    "finalization_seconds",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def describe(values):
    data = np.asarray(values, dtype=np.float64)
    require(data.size > 0 and bool(np.isfinite(data).all()), "分布为空或含非有限值")
    return {
        "count": int(data.size),
        "sum": float(data.sum()),
        "mean": float(data.mean()),
        "min": float(data.min()),
        "p50": float(np.quantile(data, 0.5)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
    }


def signature(result):
    # 排除所有计时字段，独立比较输出路线、控制反馈及累计搜索工作。
    return content_hash(
        {
            "items": [(v["tour"], v["cost"]) for v in result["items"]],
            "states": result["control_states"],
            "construction": result["completed_construction_steps"],
            "ls": result["completed_ls_evaluations"],
        }
    )


def check_sources(directory, manifest):
    archive = directory / "runtime/source.tar"
    if archive.exists():
        import hashlib

        with tarfile.open(archive, "r") as source:
            require(
                set(source.getnames()) == set(manifest["sources"]), "源码快照成员与manifest不符"
            )
            for name, expected in manifest["sources"].items():
                member = source.getmember(name)
                require(member.isfile(), "源码快照不能含链接或目录")
                stream = source.extractfile(member)
                require(
                    stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected,
                    f"源码快照摘要不符: {name}",
                )
        return {"archive": str(archive.relative_to(PROJECT)), "sha256": file_hash(archive)}
    for name, expected in manifest["sources"].items():
        require(file_hash(PROJECT / name) == expected, f"实际源码已改变且无冻结快照: {name}")
    return {"archive": None, "verified_against": "current_source_files"}


def summarize(directory):
    state = load_checkpoint(directory / "checkpoint.json")
    require(state["status"] == "complete" and not state.get("error"), "成本矩阵尚未完整结束")
    manifest = load_checkpoint(directory / "manifest.json")
    require(
        manifest == state["manifest"] and content_hash(manifest) == state["identity"],
        "矩阵manifest身份不符",
    )
    source_evidence = check_sources(directory, manifest)
    require(
        file_hash(directory / "runtime/gp_faco_ext.so") == manifest["binary_sha256"],
        "冻结运行二进制摘要不符",
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    require(file_hash(database) == manifest["database_sha256"], "数据索引已改变")
    require(
        file_hash(PROJECT / "provenance/splits.v1.json") == manifest["split_sha256"],
        "数据划分已改变",
    )
    cfg = manifest["config"]
    require(
        cfg["scope"] == "development_cost_calibration_not_fitness"
        and cfg["dimensions"] == [500, 1000]
        and cfg["colonies"] == [1, 32]
        and cfg["instances_per_scale"] == 16
        and cfg["ants"] == 32
        and cfg["fixed_batches"] == 6
        and cfg["profile_order"] == [False, True, True, False]
        and cfg["elapsed_ratio"] == 0.5,
        "本版审计要求完整预定矩阵，不能把缩小样本当作全部条件",
    )
    conditions = {v["name"]: v for v in manifest["conditions"]}
    expected_names = {
        f"{restart}-r{region}-m{2 << level}"
        for restart in ("keep", "restart")
        for region in range(4)
        for level in range(4)
    } | {"feedback-program", "maximum-63-node-program"}
    require(set(conditions) == expected_names, "预定动作/树条件有遗漏或重复")
    for name, condition in conditions.items():
        program = Program.from_dict(condition["program"])
        if name.startswith(("keep-", "restart-")):
            prefix, region, mne = name.split("-")
            level = (int(mne[1:]).bit_length() - 1) - 1
            action = 4 * int(region[1:]) + level
            mask = (1 << action) | ((1 << (action + 16)) if prefix == "restart" else 0)
            require(
                condition["mask"] == mask and program.opcode == (0,) and program.operand == (4,),
                "固定动作掩码或偏好重启评分不符",
            )
        else:
            require(condition["mask"] == 0xFFFFFFFF, "开发程序未开放完整动作集合")
            require(
                len(program.opcode) == (63 if name.startswith("maximum") else 9),
                "开发程序树大小不符",
            )
    files = {p.stem: p for p in (directory / "conditions").glob("*.json")}
    require(set(files) == set(state["completed"]), "条件文件与完成表不一致")
    expected_count = len(cfg["dimensions"]) * 17 * len(conditions)
    require(len(files) == expected_count, "成本矩阵未覆盖全部条件")
    groups = defaultdict(lambda: defaultdict(list))
    condition_groups = defaultdict(lambda: defaultdict(list))
    cycle_totals = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    registrations, seen, single, full = {}, set(), {}, {}
    route_count = 0
    maximum_error = 0.0
    max_stage_excess = 0.0
    profile_batches = 0
    cycle_records = 0
    native_seconds = 0.0
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as data:
        ids = {n: data.record_ids("development", n)[:16] for n in cfg["dimensions"]}
        problems = {name: data.load_instance(name) for names in ids.values() for name in names}
        require(set(problems) == set(state["fees"]), "费用表与开发池32成员不一致")
        for name, problem in problems.items():
            require(
                state["fees"][name]["coordinate_sha256"] == coordinate_hash(problem),
                "冻结费用坐标摘要不符",
            )
        for key, path in sorted(files.items()):
            record = load_checkpoint(path)
            require(
                record["status"] == "completed"
                and record["key"] == key
                and content_hash(record) == state["completed"][key],
                "条件记录状态或内容摘要不符",
            )
            d = record["description"]
            require(content_hash(d) == key and d["run_id"] == state["identity"], "条件身份不符")
            n, colonies, name = d["dimension"], d["colonies"], d["condition"]["name"]
            require(d["condition"] == conditions[name], "实际条件与预定IR/动作掩码不符")
            group_ids = [v[0] for v in d["problems"]]
            require(n in ids and colonies in cfg["colonies"], "出现非预定规模/形状")
            require(
                (colonies == 1 and len(group_ids) == 1 and group_ids[0] in ids[n])
                or (colonies == 32 and group_ids == ids[n]),
                "实际面板与预定开发实例不符",
            )
            expected_replicas = [
                [instance, seed]
                for instance in group_ids
                for seed in (cfg["solver_seeds"][:1] if colonies == 1 else cfg["solver_seeds"])
            ]
            require(d["replicas"] == expected_replicas, "求解seed或副本顺序不符")
            require(
                d["problems"] == [[v, coordinate_hash(problems[v])] for v in group_ids],
                "坐标身份与面板不符",
            )
            occurrence = n, colonies, tuple(group_ids), name
            require(occurrence not in seen, "重复条件替代了缺失条件")
            seen.add(occurrence)
            group = groups[(n, colonies)]
            condition_group = condition_groups[(n, colonies, name)]
            charge = sum(
                state["fees"][v]["cheap_seconds"] + state["fees"][v]["preparation_seconds"]
                for v in group_ids
            )
            require(set(record["registration"]) == set(group_ids), "实际准备记录缺失")
            for instance, registration in record["registration"].items():
                reg_key = record["pid"], n, colonies, instance
                if reg_key in registrations:
                    require(registrations[reg_key] == registration, "同Engine准备记录改变")
                registrations[reg_key] = registration
                phases = registration["phases"]
                require(set(phases) == set(PREPARATION_PHASES), "CPU准备计时项不符")
                require(
                    all(v >= 0 for v in phases.values())
                    and math.isclose(
                        sum(phases.values()),
                        registration["measured"]["preparation_seconds"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    "CPU准备分项与实际总费用不守恒",
                )
            samples = record["samples"]
            require([v["profile_enabled"] for v in samples] == cfg["profile_order"], "ABBA不符")
            baseline_signature = signature(samples[0]["result"])
            for sample in samples:
                result, enabled = sample["result"], sample["profile_enabled"]
                require(signature(result) == baseline_signature, "插桩或重复改变最终轨迹/工作")
                require(
                    result["scope"] == "fixed_batch_diagnostic_not_fitness"
                    and result["completed_batches"] == result["launched_batches"] == 6
                    and result["discarded_batches"] == 0
                    and result["preparation_completed"]
                    and result["preparation_mode"] == "cached_charged"
                    and result["budget_seconds"] == cfg["guard_budget_seconds"]
                    and result["overrun_seconds"] == 0,
                    "诊断入口、费用模式或完整批次不符",
                )
                require(
                    math.isclose(result["charged_seconds"], charge, abs_tol=1e-12)
                    and math.isclose(
                        result["elapsed_seconds"],
                        result["actual_seconds"] + result["charged_seconds"],
                        abs_tol=1e-12,
                    ),
                    "有效预算费用未按冻结表计入",
                )
                require(len(result["items"]) == colonies, "输出不覆盖全部colony")
                for (instance, _), item in zip(expected_replicas, result["items"], strict=True):
                    require(
                        item["has_incumbent"]
                        and 0 <= item["completed_seconds"] <= result["budget_seconds"],
                        "合法截止前incumbent缺失",
                    )
                    error = abs(tour_cost(problems[instance], item["tour"]) - item["cost"])
                    require(error < 1e-8, "独立重算的原目标成本不符")
                    maximum_error = max(maximum_error, error)
                    route_count += 1
                metric = "profiled_native_seconds" if enabled else "unprofiled_native_seconds"
                group[metric].append(result["actual_seconds"])
                condition_group[metric].append(result["actual_seconds"])
                group["production_device_bytes"].append(result["allocated_device_bytes"])
                native_seconds += result["actual_seconds"]
                profile = result["profile"]
                if not enabled:
                    require(
                        not profile["batches"] and profile["diagnostic_device_bytes"] == 0,
                        "未插桩对照产生了设备计时数组",
                    )
                    continue
                require(
                    profile["profile_version"] == 1
                    and profile["initialization_includes_uploads"] is True
                    and profile["diagnostic_device_bytes"] == colonies * cfg["ants"] * 32
                    and len(profile["batches"]) == 6,
                    "profile版本或缓冲/批次大小不符",
                )
                for metric in (
                    "setup_seconds",
                    "host_pack_seconds",
                    "upload_call_seconds",
                    "initialization_gpu_milliseconds",
                    "initialization_download_seconds",
                    "diagnostic_device_bytes",
                ):
                    group[metric].append(profile[metric])
                for batch_id, batch in enumerate(profile["batches"]):
                    require(batch["batch"] == batch_id and batch["committed"], "批次顺序/提交不符")
                    require(set(batch["gpu_milliseconds"]) == set(GPU_PHASES), "GPU计时项不符")
                    gpu_sum = sum(batch["gpu_milliseconds"].values()) * 1e-3
                    max_stage_excess = max(max_stage_excess, gpu_sum - batch["wall_seconds"])
                    require(gpu_sum <= batch["wall_seconds"] + 2e-4, "GPU分项超出主机批区间")
                    for stage, duration in batch["gpu_milliseconds"].items():
                        require(duration >= 0, "负GPU计时")
                        group[f"gpu_ms/{stage}"].append(duration)
                        condition_group[f"gpu_ms/{stage}"].append(duration)
                    for metric in (
                        "wall_seconds",
                        "download_seconds",
                        "verification_seconds",
                        "collection_seconds",
                    ):
                        require(batch[metric] >= 0, "负主机计时")
                        group[f"batch/{metric}"].append(batch[metric])
                        condition_group[f"batch/{metric}"].append(batch[metric])
                    group["batch/uncategorized_seconds"].append(
                        batch["wall_seconds"]
                        - gpu_sum
                        - batch["download_seconds"]
                        - batch["verification_seconds"]
                        - batch["collection_seconds"]
                    )
                    cycles = np.asarray(batch["ant_phase_cycles"], dtype=np.uint64)
                    require(
                        cycles.shape == (colonies * cfg["ants"], 4) and bool((cycles > 0).all()),
                        "每ant四阶段周期计数不完整",
                    )
                    cycle_totals[(n, colonies)] += cycles.sum(axis=0, dtype=np.uint64)
                    cycle_records += len(cycles)
                    profile_batches += 1
            unprofiled = np.mean(
                [v["result"]["actual_seconds"] for v in samples if not v["profile_enabled"]]
            )
            profiled = np.mean(
                [v["result"]["actual_seconds"] for v in samples if v["profile_enabled"]]
            )
            for target in (group, condition_group):
                target["paired_profile_overhead_fraction"].append(float(profiled / unprofiled - 1))
            result = samples[0]["result"]
            condition_group["colony_restarts"].append(
                sum(v["restarts"] for v in result["control_states"])
            )
            condition_group["completed_construction_steps"].append(
                result["completed_construction_steps"]
            )
            condition_group["completed_ls_evaluations"].append(result["completed_ls_evaluations"])
            for (instance, seed), item in zip(expected_replicas, result["items"], strict=True):
                if seed == cfg["solver_seeds"][0]:
                    (single if colonies == 1 else full)[(n, name, instance)] = (
                        item["tour"],
                        item["cost"],
                    )
        require(single == full and len(single) == 1088, "单实例与32-colony完整对应轨迹不符")

    preparation = defaultdict(lambda: defaultdict(list))
    for (_, n, colonies, _), value in registrations.items():
        target = preparation[(n, colonies)]
        for field, duration in value["measured"].items():
            target[field].append(duration)
        for field, duration in value["phases"].items():
            target[field].append(duration)
    shape_reports = {}
    proposals = {}
    for (n, colonies), values in sorted(groups.items()):
        report = {metric: describe(data) for metric, data in sorted(values.items())}
        cycles = cycle_totals[(n, colonies)]
        report["ant_block_cycle_fraction"] = dict(
            zip(CYCLE_PHASES, (cycles / cycles.sum()).tolist(), strict=True)
        )
        report["cpu_preparation"] = {k: describe(v) for k, v in preparation[(n, colonies)].items()}
        shape_reports[f"{n}/{colonies}"] = report
        # 插桩批次p99是预算候选依据，不冒充生产批次p99；下一步必须实际校准正常deadline。
        batch_p99 = report["batch/wall_seconds"]["p99"]
        charges = [
            state["fees"][v]["cheap_seconds"] + state["fees"][v]["preparation_seconds"]
            for v in ids[n]
        ]
        fee = sum(charges) if colonies == 32 else max(charges)
        profile_initialization = max(
            values["host_pack_seconds"][i]
            + values["initialization_gpu_milliseconds"][i] * 1e-3
            + values["initialization_download_seconds"][i]
            for i in range(len(values["host_pack_seconds"]))
        )
        rule = cfg["provisional_budget_rule"]
        short = (
            math.ceil(
                (
                    fee
                    + profile_initialization
                    + max(
                        batch_p99 * rule["minimum_short_search_batches"],
                        batch_p99 / rule["maximum_last_batch_fraction"],
                    )
                )
                * 10
            )
            / 10
        )
        proposals[f"{n}/{colonies}"] = {
            "profiled_batch_p99_seconds": batch_p99,
            "frozen_preparation_seconds": fee,
            "profiled_initialization_max_seconds": profile_initialization,
            "short_seconds": short,
            "medium_seconds": short * rule["medium_multiple_of_short"],
            "long_seconds": short * rule["long_multiple_of_short"],
            "status": "candidate_only_requires_production_deadline_and_Static_Rule_calibration",
        }
    invocations = state["invocations"]
    require(invocations, "完整过程资源记录缺失")
    costs = {
        name: sum(v["costs_seconds"][name] for v in invocations)
        for name in invocations[0]["costs_seconds"]
    }
    process_wall = sum(v["process_wall_seconds"] for v in invocations)
    return {
        "version": 1,
        "scope": cfg["scope"],
        "source_directory": str(directory.relative_to(PROJECT)),
        "identity": state["identity"],
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "checkpoint_sha256": file_hash(directory / "checkpoint.json"),
        "binary_sha256": manifest["binary_sha256"],
        "source_evidence": source_evidence,
        "audit_script_sha256": file_hash(Path(__file__)),
        "device": manifest["device"],
        "config": cfg,
        "audit": {
            "conditions": len(files),
            "measured_native_calls": len(files) * 4,
            "warmup_native_calls": len(files),
            "independently_verified_routes": route_count,
            "maximum_cost_absolute_error": maximum_error,
            "cross_shape_identical_trajectories": len(single),
            "profiled_batches": profile_batches,
            "ant_phase_cycle_records": cycle_records,
            "unique_preparation_measurements": len(registrations),
            "maximum_gpu_sum_above_host_batch_seconds": max_stage_excess,
            "failed_attempt_files": {
                str(p.relative_to(directory)): file_hash(p)
                for p in sorted((directory / "failures").glob("*.json"))
            },
        },
        "resources": {
            "invocations": invocations,
            "exclusive_coordinator_seconds": costs,
            "process_wall_seconds": process_wall,
            "process_uncategorized_seconds": process_wall - sum(costs.values()),
            "measured_native_seconds_excluding_warmup": native_seconds,
            "cpu_user_seconds": sum(v["cpu_user_seconds"] for v in invocations),
            "cpu_system_seconds": sum(v["cpu_system_seconds"] for v in invocations),
            "max_rss_kib": max(v["max_rss_kib"] for v in invocations),
            "boundary_sampled_peak_gpu_mib": max(
                v["boundary_sampled_peak_gpu_mib"] for v in invocations
            ),
        },
        "shapes": shape_reports,
        "conditions": {
            f"{n}/{colonies}/{name}": {metric: describe(v) for metric, v in values.items()}
            for (n, colonies, name), values in sorted(condition_groups.items())
        },
        "budget_candidates": proposals,
        "interpretation": [
            "所有预算仅为待生产deadline核验的候选；没有读取正式测试或任何解标签。",
            "CUDA event区间包含流中的主机提交空隙；初始化包含H2D，upload_call_seconds与之重叠。",
            "每ant线程0的clock64周期是含同步等待的block阶段工作分布，不是并发GPU墙钟占比。",
            "batch/wall_seconds是插桩批次区间，止于计时数据下载后；未包括随后incumbent提交记账。",
            "unprofiled_native_seconds是六批次完整调用，不将其均值冒充单批次p99。",
            "峰值显存为条件边界采样，可能低于瞬时峰值；显式设备缓冲与CUDA上下文/事件资源分列。",
            "本矩阵含cold事件首次分配，ABBA每条件均保留；setup_seconds另列，不删除慢样本。",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    require(directory.is_relative_to(PROJECT), "仅审计项目内产物")
    report = summarize(directory)
    timing = directory.with_suffix(".time.json")
    if timing.exists():
        report["resources"]["outer_gnu_time"] = json.loads(timing.read_text())
        report["resources"]["outer_gnu_time_sha256"] = file_hash(timing)
        require(report["resources"]["outer_gnu_time"]["exit_code"] == 0, "外部运行未成功退出")
    atomic_json(args.output, report)
    print(
        json.dumps(
            {"audit": report["audit"], "budget_candidates": report["budget_candidates"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
