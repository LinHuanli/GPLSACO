#!/usr/bin/env python3
"""不启动CUDA，重建全部预登记任务、重算路线/选择/次数曲线并核查真实恢复。"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.configuration_search import SearchData, SearchSettings  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evaluation_run import json_value  # noqa: E402
from gp_faco.fitness import score_panel  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    BaselineTask,
    SolverSettings,
    WorkerProtocol,
    content_hash,
    file_hash,
)
from search_baselines import policy_grid  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summarize(directory, paused=None):
    manifest = load_checkpoint(directory / "manifest.json")
    state = load_checkpoint(directory / "checkpoint.json")
    plan = load_checkpoint(directory / "plan.json")
    require(state["phase"] in ("complete", "failed"), "运行尚未结束")
    require(state["pending"] is None and state["active_worker"] is None, "尚有活动任务")
    require(content_hash(manifest) == state["run_id"], "运行身份改变")
    require(content_hash(plan) == manifest["plan_sha256"], "冻结计划摘要改变")
    for name, fingerprint in manifest["sources"].items():
        require(file_hash(PROJECT / name) == fingerprint, f"源码改变，须在原快照重放: {name}")
    p = manifest["worker_protocol"]
    protocol = WorkerProtocol(
        **{
            f.name: SolverSettings(**p[f.name]) if f.name == "settings" else p[f.name]
            for f in fields(WorkerProtocol)
            if f.init
        }
    )
    require(json_value(protocol.manifest()) == p, "worker实现身份改变")
    require(
        file_hash(PROJECT / protocol.extension_directory / "gp_faco_ext.so")
        == protocol.binary_sha256,
        "原生二进制身份改变",
    )
    require(
        manifest["wall_clock_limit"] is None
        and manifest["budget_kind"] == "search_tour_evaluations",
        "主预算并非纯FE",
    )
    settings = SearchSettings(**manifest["settings"])
    identity = manifest["data"]["identity"]
    config_path = PROJECT / identity["config_path"]
    require(file_hash(config_path) == identity["config_sha256"], "预登记配置改变")
    require(
        file_hash(PROJECT / "scripts/search_baselines.py") == identity["entrypoint_sha256"],
        "配置展开/数据选取入口改变",
    )
    config = json.loads(config_path.read_text())
    require(asdict(SearchSettings(**config["search"])) == asdict(settings), "次数/选择配置不符")
    policies = (
        tuple(BaselinePolicy.from_dict(v) for v in config["policies"])
        if "policies" in config
        else policy_grid(config["policy_grid"])
    )
    by_sha = {v.sha256: v for v in sorted(policies, key=lambda v: v.sha256)}
    require(
        len(by_sha) == len(policies)
        and [v.to_dict() for v in by_sha.values()] == manifest["policies"],
        "配置族覆盖不符",
    )
    require(
        config["scope"] == identity["scope"] and "development" in config["scope"],
        "研究范围不是预定development",
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    require(file_hash(database) == identity["database_sha256"], "索引身份改变")
    require(
        file_hash(PROJECT / "provenance/splits.v1.json") == identity["split_sha256"], "划分身份改变"
    )
    files = {path.stem: path for path in (directory / "tasks").glob("*.json")}
    require(set(files) == set(state["completed"]), "原始记录与完成表的覆盖不符")
    histories = state["worker_history"]
    require(
        histories
        and all(
            v["host"] == protocol.execution_host
            and v["protocol_sha256"] == protocol.sha256
            and v["start_method"] == "spawn"
            for v in histories
        ),
        "worker身份不符",
    )
    known_pids = {v["pid"] for v in histories}
    admitted = Counter(v["task_key"] for v in state["admission"] if v["foreign_processes"] == 0)
    records, phase_records, failures = {}, defaultdict(list), []
    grouped, curves, work_curves, initial = {}, defaultdict(dict), defaultdict(dict), {}
    seen, rows = set(), []
    max_error, members, prefix_pairs, initial_pairs = 0.0, 0, 0, 0

    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        pools, expected_panels = {}, {}
        for role, selection in config["data_pools"].items():
            pools[role], expected_panels[role] = {}, []
            for n in protocol.dimensions:
                ids = sorted(source.record_ids("development", n))
                selected = ids[selection["offset"] : selection["offset"] + selection["count"]]
                require(len(selected) == selection["count"], "development池不足")
                pools[role][n] = selected
                width = settings.instances_per_panel
                require(len(selected) % width == 0, "未使用完整面板")
                expected_panels[role].extend(
                    {"dimension": n, "ids": selected[i : i + width]}
                    for i in range(0, len(selected), width)
                )
        require(json_value(pools) == manifest["data"]["pools"], "冻结数据池不符")
        require(expected_panels == plan["panels"], "实例/规模面板覆盖不符")
        data = SearchData(source, pools, identity)
        rng, expected_search = random.Random(settings.order_seed), []
        for limit in settings.evaluation_limits:
            order = list(by_sha)
            rng.shuffle(order)
            expected_search.extend(
                {"policy_sha256": sha, "limit": limit, "panel": panel}
                for sha in order
                for panel in range(len(expected_panels["search"]))
            )
        require(expected_search == plan["search"], "预登记随机顺序不能重放")

        def phase_scores(phase, jobs):
            nonlocal max_error, members, initial_pairs
            buckets = defaultdict(list)
            for index, job in enumerate(jobs):
                panel = expected_panels[phase][job["panel"]]
                problems = data.problems(panel["ids"])
                task = BaselineTask(
                    f"{state['run_id']}:{phase}:case{index}",
                    by_sha[job["policy_sha256"]],
                    problems,
                    tuple(
                        (v.instance_id, seed) for v in problems for seed in settings.solver_seeds
                    ),
                    job["limit"],
                    settings.preparation_mode,
                    settings.experiment_mask,
                )
                description = json_value(task.manifest(protocol))
                key = content_hash(
                    {"run_id": state["run_id"], "kind": "solve", "description": description}
                )
                require(key not in seen and key in files, "任务重复或缺失")
                seen.add(key)
                record = load_checkpoint(files[key])
                require(
                    content_hash(record) == state["completed"][key]
                    and record["description"] == description
                    and record["key"] == key
                    and record["kind"] == "solve",
                    "任务原始记录身份改变",
                )
                require("program_sha256" not in description, "基线混入GP替身身份")
                checked = score_panel(task, protocol, record["outcome"], data.labels(task))
                require(json_value(asdict(checked)) == record["checked"], "外部复核结果不符")
                records[key] = record
                phase_records[phase].append(record)
                native = record["outcome"].get("native_result", {})
                if checked.failed:
                    failures.append({"key": key, "error": checked.error})
                else:
                    require(record["outcome"]["worker_pid"] in known_pids, "返回worker未知")
                    require(admitted[key] >= len(record["attempts"]), "缺少提交前空闲记录")
                    require(record["attempts"][-1]["status"] == "returned", "返回非真实终态")
                    members += len(checked.members)
                    by_name = {v.instance_id: v for v in problems}
                    for (name, seed), item in zip(task.replicas, native["items"], strict=True):
                        actual = tour_cost(by_name[name], item["tour"])
                        max_error = max(max_error, abs(actual - item["cost"]))
                        if settings.purpose == "calibration":
                            curves[(task.policy.sha256, name, seed)][job["limit"]] = actual
                            if job["limit"] == 0:
                                marker = (item["tour"], item["cost"])
                                if (name, seed) in initial:
                                    require(
                                        initial[name, seed] == marker, "不同配置的零FE初始化不同"
                                    )
                                    initial_pairs += 1
                                initial[name, seed] = marker
                    if settings.purpose == "calibration":
                        work_curves[(task.policy.sha256, job["panel"])][job["limit"]] = {
                            k: native[k]
                            for k in ("completed_construction_steps", "completed_ls_evaluations")
                        }
                buckets[f"{job['limit']}/{task.policy.sha256}"].append((key, checked, native))
            scores = {}
            for k, entries in buckets.items():
                scale_values = {}
                for n in protocol.dimensions:
                    scale = [entry for entry in entries if entry[1].dimension == n]
                    require(
                        len(scale) == len(pools[phase][n]) // settings.instances_per_panel,
                        "候选未覆盖该规模全部面板",
                    )
                    values = defaultdict(list)
                    for _, checked, _ in scale:
                        for name, _, gap in checked.members:
                            values[name].append(gap)
                    scale_values[str(n)] = (
                        None
                        if any(v[1].failed for v in scale)
                        else statistics.mean(statistics.mean(v) for v in values.values())
                    )
                scores[k] = {
                    "fitness": None
                    if None in scale_values.values()
                    else statistics.mean(scale_values.values()),
                    "per_dimension": scale_values,
                    "task_keys": [v[0] for v in entries],
                }
                limit, sha = k.split("/")
                rows.append(
                    {
                        "phase": phase,
                        "limit": int(limit),
                        "policy_sha256": sha,
                        "kind": by_sha[sha].kind,
                        **{f"gap_{n}": v for n, v in scale_values.items()},
                        "macro_gap": scores[k]["fitness"],
                        "native_seconds": sum(v[2].get("actual_seconds", 0) for v in entries),
                        "ls_move_checks": sum(
                            v[2].get("completed_ls_evaluations", 0) for v in entries
                        ),
                    }
                )
            grouped[phase] = scores
            return scores

        search_scores = phase_scores("search", expected_search)
        require(search_scores == state["search_scores"], "搜索宏平均或候选覆盖不符")
        if settings.purpose == "tuning":
            limit, shortlist = settings.evaluation_limits[0], []
            for kind in ("static", "rule"):
                ranked = sorted(
                    (v["fitness"], k.split("/")[1])
                    for k, v in search_scores.items()
                    if v["fitness"] is not None and by_sha[k.split("/")[1]].kind == kind
                )
                if not ranked:
                    require(
                        state["phase"] == "failed" and not state["selected"], "全失败基线仍被选择"
                    )
                    shortlist = []
                    break
                shortlist.extend(sha for _, sha in ranked[: settings.validation_shortlist_per_kind])
            require(shortlist == state["shortlist"], "验证短名单不符")
            validation_plan = [
                {"policy_sha256": sha, "limit": limit, "panel": panel}
                for sha in shortlist
                for panel in range(len(expected_panels["validation"]))
            ]
            require(validation_plan == state["validation_plan"], "统一验证遗漏候选或面板")
            if validation_plan:
                scores = phase_scores("validation", validation_plan)
                require(scores == state["validation_scores"], "固定验证汇总不符")
                selected = {}
                for kind in ("static", "rule"):
                    ranked = sorted(
                        (v["fitness"], k.split("/")[1])
                        for k, v in scores.items()
                        if v["fitness"] is not None and by_sha[k.split("/")[1]].kind == kind
                    )
                    if not ranked:
                        selected = {}
                        break
                    _, sha = ranked[0]
                    selected[kind] = {
                        "policy": by_sha[sha].to_dict(),
                        "policy_sha256": sha,
                        **scores[f"{limit}/{sha}"],
                    }
                require(selected == state["selected"], "导出不是固定验证的逐类最优")
                if selected:
                    exported = load_checkpoint(directory / "selected_baselines.json")
                    require(
                        exported
                        == {
                            "run_id": state["run_id"],
                            "manifest": manifest,
                            "selected": selected,
                            "validation_scores_sha256": content_hash(scores),
                        },
                        "独立导出身份不符",
                    )
        else:
            require(
                not state["selected"] and not state["shortlist"] and not state["validation_plan"],
                "校准不应静默选择正式基线",
            )

    require(seen == set(files), "存在未预登记评价或计划遗漏")
    incomplete_curves = 0
    for values in curves.values():
        if set(values) != set(settings.evaluation_limits):
            require(failures, "FE曲线缺少档位")
            incomplete_curves += 1
        ordered = [v for _, v in sorted(values.items())]
        for a, b in zip(ordered[:-1], ordered[1:], strict=True):
            require(b <= a + 1e-10, "无progress基线的FE前缀最佳路线反而变差")
            prefix_pairs += 1
    for values in work_curves.values():
        ordered = [v for _, v in sorted(values.items())]
        for a, b in zip(ordered[:-1], ordered[1:], strict=True):
            require(all(b[k] >= a[k] for k in a), "FE前缀累计工作量减少")
    costs = state["costs"]
    require(
        costs["solve_jobs"] == len(records)
        and costs["valid_members"] == members
        and costs["failed_solves"] == len(failures)
        and costs["preparation_jobs"] == 0,
        "任务/成员/失败账目不平",
    )
    for field, source_field, native_field in (
        ("worker_seconds", "worker_seconds", False),
        ("registration_seconds", "registration_seconds", False),
        ("native_actual_seconds", "actual_seconds", True),
        ("search_tour_evaluations", "total_tour_evaluations", True),
        ("charged_seconds", "charged_seconds", True),
        ("overrun_seconds", "overrun_seconds", True),
        *[
            (k, k, True)
            for k in (
                "completed_batches",
                "completed_construction_steps",
                "completed_ls_evaluations",
            )
        ],
    ):
        actual = sum(
            (r["outcome"].get("native_result", {}) if native_field else r["outcome"]).get(
                source_field, 0
            )
            for r in records.values()
        )
        require(
            math.isclose(costs[field], actual, rel_tol=1e-12, abs_tol=1e-10),
            f"资源账目不符: {field}",
        )
    require(
        math.isclose(
            costs["evaluator_seconds"],
            sum(r["evaluation_seconds"] for r in records.values()),
            rel_tol=1e-12,
            abs_tol=1e-10,
        ),
        "外部评分资源账目不符",
    )
    recovery = None
    if paused:
        old = load_checkpoint(paused)
        require(
            old["run_id"] == state["run_id"]
            and old["pending"] is None
            and old["active_worker"] is None,
            "暂停快照不是已关闭任务边界",
        )
        require(
            all(state["completed"].get(k) == v for k, v in old["completed"].items()),
            "恢复改写已完成记录",
        )
        require(
            old["worker_history"] == histories[: len(old["worker_history"])]
            and len(histories) > len(old["worker_history"]),
            "未实际重建worker",
        )
        recovery = {
            "preserved_records": len(old["completed"]),
            "paused_workers": len(old["worker_history"]),
            "total_workers": len(histories),
        }
    after = [r["outcome"].get("gpu_boundary_after", {}) for r in records.values()]
    occupancy = {
        "before_foreign_samples": sum(v["foreign_processes"] > 0 for v in state["admission"]),
        "after_foreign_samples": sum((v.get("foreign_processes") or 0) > 0 for v in after),
        "after_unknown_samples": sum(v.get("foreign_processes") is None for v in after),
        "interpretation": "task boundaries only; not a continuous exclusivity claim",
    }
    expected_evaluations = sum(
        r["description"]["evaluation_limit_per_colony"] * protocol.colonies
        for r in records.values()
    )
    if not failures:
        require(
            costs["search_tour_evaluations"] == expected_evaluations, "完成FE未覆盖完整预定任务"
        )
    return {
        "status": "passed" if not failures else "completed_with_failures",
        "scope": identity["scope"],
        "purpose": settings.purpose,
        "run_id": state["run_id"],
        "policies": dict(Counter(p.kind for p in policies)),
        "limits": settings.evaluation_limits,
        "phase_jobs": {k: len(v) for k, v in phase_records.items()},
        "routes": members,
        "expected_tour_evaluations": expected_evaluations,
        "costs": costs,
        "max_route_cost_error": max_error,
        "failures": failures,
        "gpu_occupancy": occupancy,
        "worker_history": histories,
        "recovery": recovery,
        "prefix_pairs": prefix_pairs,
        "common_initial_pairs": initial_pairs,
        "incomplete_curves": incomplete_curves,
        "selected": state["selected"],
        "curves": sorted(rows, key=lambda v: (v["phase"], v["limit"], v["policy_sha256"])),
        "identity": {
            "manifest_sha256": file_hash(directory / "manifest.json"),
            "checkpoint_sha256": file_hash(directory / "checkpoint.json"),
            "binary_sha256": protocol.binary_sha256,
            "sources": manifest["sources"],
        },
        "resource_note": "Registration/native timings nest inside worker; repeated verification "
        "and recovery aggregation are additionally covered by whole-CLI resource records.",
        "evidence_limit": "Development only. Reference labels are user supplied, not independently "
        "certified. E1-E4 and final G4 freeze are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--paused-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory.resolve(), args.paused_checkpoint)
    atomic_json(args.output, report)
    if args.csv:
        if not args.csv.resolve().is_relative_to(PROJECT):
            parser.error("CSV产物须位于GPLSACO内")
        with args.csv.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["curves"][0]))
            writer.writeheader()
            writer.writerows(report["curves"])
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "purpose",
                    "policies",
                    "phase_jobs",
                    "routes",
                    "costs",
                    "prefix_pairs",
                    "recovery",
                )
            },
            ensure_ascii=False,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
