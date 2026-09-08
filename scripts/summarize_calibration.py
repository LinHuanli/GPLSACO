#!/usr/bin/env python3
"""逐条核验生产截止矩阵，按规模/形状/费用模式报告预定预算门槛。"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402
from summarize_profiling import check_sources, describe, require  # noqa: E402


def expected_cases(cfg, names, conditions, identity, hashes):
    """独立重建矩阵；不根据已成功的文件决定期望成员。"""
    result = {}
    for n in cfg["dimensions"]:
        for colonies in cfg["colonies"]:
            panels = [[name] for name in names[n]] if colonies == 1 else [names[n]]
            for index, panel in enumerate(panels):
                for level in ("short", "medium", "long"):
                    if level == "short":
                        if colonies == 32:
                            selected = list(conditions)
                        else:
                            selected = [
                                ("restart" if (index // 4) % 2 else "keep")
                                + f"-r{(index + degree) % 4}-m{2 << degree}"
                                for degree in range(4)
                            ]
                    elif colonies == 32 or index == 0:
                        selected = cfg["longer_budget_conditions"]
                    else:
                        continue
                    for condition in selected:
                        for mode in cfg["preparation_modes"]:
                            d = {
                                "run_id": identity,
                                "dimension": n,
                                "colonies": colonies,
                                "replicas": [
                                    [name, seed]
                                    for name in panel
                                    for seed in (
                                        cfg["solver_seeds"][:1]
                                        if colonies == 1
                                        else cfg["solver_seeds"]
                                    )
                                ],
                                "condition": conditions[condition],
                                "preparation_mode": mode,
                                "budget_level": level,
                                "budget_seconds": cfg["budgets"][f"{n}/{colonies}"][
                                    f"{level}_seconds"
                                ],
                                "problems": [[name, hashes[name]] for name in sorted(panel)],
                            }
                            result[content_hash(d)] = d
    require(len(result) == 424, "本轮预定矩阵应为424任务")
    return result


def summarize(directory):
    state = load_checkpoint(directory / "checkpoint.json")
    require(state["status"] == "complete" and not state.get("error"), "生产校准尚未完整结束")
    manifest = load_checkpoint(directory / "manifest.json")
    require(
        manifest == state["manifest"] and content_hash(manifest) == state["identity"],
        "运行身份不符",
    )
    cfg = manifest["config"]
    require(
        cfg["scope"] == "development_production_deadline_calibration_not_fitness"
        and cfg["normal_elapsed"] is True
        and cfg["diagnostics"] is False,
        "校准入口声明不符",
    )
    source_evidence = check_sources(directory, manifest)
    require(
        file_hash(directory / "runtime/gp_faco_ext.so") == manifest["binary_sha256"], "二进制改变"
    )
    require(
        file_hash(directory / "runtime/calibrate_budget.py") == cfg["entrypoint_sha256"],
        "校准入口快照改变",
    )
    origin = PROJECT / cfg["profiling_directory"]
    require(
        file_hash(origin / "checkpoint.json") == cfg["profiling_checkpoint_sha256"],
        "费用/计时来源改变",
    )
    profiled = load_checkpoint(origin / "checkpoint.json")
    require(
        profiled["status"] == "complete"
        and profiled["identity"] == cfg["profiling_identity"]
        and profiled["fees"] == state["fees"]
        and profiled["manifest"]["conditions"] == manifest["conditions"]
        and profiled["manifest"]["binary_sha256"] == manifest["binary_sha256"]
        and profiled["manifest"]["device"] == manifest["device"],
        "生产校准与成本矩阵的费用、设备或二进制不一致",
    )
    report_path = directory / "runtime/profiling_report.json"
    require(file_hash(report_path) == cfg["profiling_report_sha256"], "成本报告改变")
    require(
        json.loads(report_path.read_text())["budget_candidates"] == cfg["budgets"], "候选预算改变"
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    require(file_hash(database) == manifest["database_sha256"], "索引改变")
    require(
        file_hash(PROJECT / "provenance/splits.v1.json") == manifest["split_sha256"], "划分改变"
    )
    files = {p.stem: p for p in (directory / "conditions").glob("*.json")}
    groups = defaultdict(lambda: defaultdict(list))
    strata = defaultdict(list)
    completed_batches = discarded_batches = members = 0
    maximum_error = charged_seconds = actual_seconds = 0.0
    single_coverage = Counter()
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as data:
        names = {n: data.record_ids("development", n)[:16] for n in cfg["dimensions"]}
        problems = {name: data.load_instance(name) for panel in names.values() for name in panel}
        hashes = {name: coordinate_hash(p) for name, p in problems.items()}
        require(set(hashes) == set(state["fees"]), "开发面板与完整费用表不符")
        require(
            all(v["coordinate_sha256"] == hashes[name] for name, v in state["fees"].items()),
            "费用坐标摘要改变",
        )
        expected = expected_cases(
            cfg, names, {v["name"]: v for v in manifest["conditions"]}, state["identity"], hashes
        )
        require(
            set(expected) == set(files) == set(state["completed"]), "完整任务集合有缺失或额外任务"
        )
        for key, path in sorted(files.items()):
            record = load_checkpoint(path)
            d = record["description"]
            require(
                record["status"] == "completed"
                and record["key"] == key
                and content_hash(record) == state["completed"][key]
                and d == expected[key],
                "任务身份或完成产物摘要改变",
            )
            r = record["result"]
            n, colonies, mode, level = (
                d[v] for v in ("dimension", "colonies", "preparation_mode", "budget_level")
            )
            budget = d["budget_seconds"]
            require("profile" not in r and "scope" not in r, "生产输出出现诊断字段")
            require(
                r["preparation_mode"] == mode
                and r["budget_seconds"] == budget
                and r["preparation_completed"]
                and r["launched_batches"] == r["completed_batches"] + r["discarded_batches"]
                and r["discarded_batches"] in (0, 1),
                "准备状态或完整批次提交记账不符",
            )
            charge = (
                sum(
                    state["fees"][name]["cheap_seconds"]
                    + state["fees"][name]["preparation_seconds"]
                    for name, _ in d["problems"]
                )
                if mode == "cached_charged"
                else 0.0
            )
            require(
                math.isclose(r["charged_seconds"], charge, abs_tol=1e-12)
                and math.isclose(
                    r["elapsed_seconds"], r["charged_seconds"] + r["actual_seconds"], abs_tol=1e-12
                )
                and math.isclose(
                    r["overrun_seconds"], max(0.0, r["elapsed_seconds"] - budget), abs_tol=1e-12
                ),
                "冻结扣费、实际时间、超限关系不符",
            )
            if r["discarded_batches"]:
                require(r["last_batch_completed_seconds"] > budget, "丢弃批次没有晚于截止")
            require(
                len(r["items"]) == len(r["control_states"]) == colonies, "输出成员或控制状态缺失"
            )
            error_max = 0.0
            for (name, _), item in zip(d["replicas"], r["items"], strict=True):
                require(
                    item["has_incumbent"] and 0 <= item["completed_seconds"] <= budget,
                    "迟到解或初解缺失",
                )
                error = abs(tour_cost(problems[name], item["tour"]) - item["cost"])
                require(error < 1e-8, "独立路线核验失败")
                error_max = max(error_max, error)
                members += 1
            require(
                error_max == record["maximum_cost_absolute_error"], "独立复算与归档核验误差不符"
            )
            maximum_error = max(maximum_error, error_max)
            completed_batches += r["completed_batches"]
            discarded_batches += r["discarded_batches"]
            charged_seconds += r["charged_seconds"]
            actual_seconds += r["actual_seconds"]
            group = groups[(n, colonies, mode, level)]
            for field in (
                "actual_seconds",
                "charged_seconds",
                "overrun_seconds",
                "completed_batches",
                "discarded_batches",
                "allocated_device_bytes",
                "completed_ls_evaluations",
                "completed_construction_steps",
            ):
                group[field].append(r[field])
            group["overrun_fraction"].append(r["overrun_seconds"] / budget)
            group["discarded_batch_fraction"].append(
                r["discarded_batches"] / max(r["launched_batches"], 1)
            )
            group["colony_restarts"].append(sum(v["restarts"] for v in r["control_states"]))
            strata[(n, colonies, mode)].append(
                {
                    "key": key,
                    "short": level == "short",
                    "batches": r["completed_batches"],
                    "overrun_fraction": r["overrun_seconds"] / budget,
                }
            )
            if level == "short" and colonies == 1:
                single_coverage[(n, mode, d["condition"]["name"])] += 1
    require(
        len(single_coverage) == 128 and set(single_coverage.values()) == {2},
        "单实例短预算未均衡覆盖32动作",
    )
    acceptance = {}
    rule = cfg["acceptance"]
    for key, samples in sorted(strata.items()):
        short = [v for v in samples if v["short"]]
        insufficient = [
            v["key"] for v in short if v["batches"] < rule["minimum_short_completed_batches"]
        ]
        over_limit = [
            v["key"] for v in samples if v["overrun_fraction"] > rule["maximum_overrun_fraction"]
        ]
        enough_fraction, bounded_fraction = (
            1 - len(insufficient) / len(short),
            1 - len(over_limit) / len(samples),
        )
        acceptance["/".join(map(str, key))] = {
            "short_calls": len(short),
            "all_calls": len(samples),
            "fraction_short_with_at_least_20_batches": enough_fraction,
            "fraction_overrun_at_most_5_percent": bounded_fraction,
            "below_batch_minimum_tasks": insufficient,
            "above_overrun_limit_tasks": over_limit,
            "passed": enough_fraction >= rule["required_short_fraction_meeting_batch_minimum"]
            and bounded_fraction >= rule["required_fraction_meeting_overrun_limit"],
        }
    invocations = state["invocations"]
    wall = sum(v["process_wall_seconds"] for v in invocations)
    costs = {
        name: sum(v["costs_seconds"][name] for v in invocations)
        for name in invocations[0]["costs_seconds"]
    }
    return {
        "version": 1,
        "scope": cfg["scope"],
        "source_directory": str(directory.relative_to(PROJECT)),
        "identity": state["identity"],
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "checkpoint_sha256": file_hash(directory / "checkpoint.json"),
        "binary_sha256": manifest["binary_sha256"],
        "source_evidence": source_evidence,
        "entrypoint_sha256": cfg["entrypoint_sha256"],
        "audit_script_sha256": file_hash(Path(__file__)),
        "device": manifest["device"],
        "config": cfg,
        "audit": {
            "calls": len(files),
            "independently_verified_routes": members,
            "maximum_cost_absolute_error": maximum_error,
            "completed_batches": completed_batches,
            "discarded_batches": discarded_batches,
            "failed_attempt_files": {
                str(p.relative_to(directory)): file_hash(p)
                for p in sorted((directory / "failures").glob("*.json"))
            },
        },
        "resources": {
            "invocations": invocations,
            "process_wall_seconds": wall,
            "exclusive_coordinator_seconds": costs,
            "process_uncategorized_seconds": wall - sum(costs.values()),
            "actual_native_seconds": actual_seconds,
            "frozen_charged_seconds": charged_seconds,
            "cpu_user_seconds": sum(v["cpu_user_seconds"] for v in invocations),
            "cpu_system_seconds": sum(v["cpu_system_seconds"] for v in invocations),
            "max_rss_kib": max(v["max_rss_kib"] for v in invocations),
            "boundary_sampled_peak_gpu_mib": max(
                v["boundary_sampled_peak_gpu_mib"] for v in invocations
            ),
        },
        "groups": {
            "/".join(map(str, key)): {field: describe(v) for field, v in group.items()}
            for key, group in sorted(groups.items())
        },
        "acceptance": acceptance,
        "all_calibration_strata_passed": all(v["passed"] for v in acceptance.values()),
        "interpretation": [
            "只以开发池时间/提交数决定候选预算；没有读取标签或优化gap。",
            "此矩阵验证正常生产elapsed，不要求不同budget或费用模式产生同一轨迹。",
            "overrun是实际结束超过有效预算的时间，不是整个丢弃批次的时长。",
            "每个调用可能损失最后一整批；迟到的结果与控制反馈均不进入输出。",
            "预算是否满足本次门槛均如实保留；通过仍不替代Static/Rule调参与正式费用表冻结。",
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
    print(json.dumps({"audit": report["audit"], "acceptance": report["acceptance"]}, indent=2))


if __name__ == "__main__":
    main()
