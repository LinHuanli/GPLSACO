#!/usr/bin/env python3
"""流式核验完整基线搜索计划的已完成前缀；不选择候选，不启动GPU。"""

import argparse
import json
import math
import random
import sys
from collections import Counter
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


def audit(directory):
    manifest = load_checkpoint(directory / "manifest.json")
    state = load_checkpoint(directory / "checkpoint.json")
    plan = load_checkpoint(directory / "plan.json")
    require(state["phase"] == "search", "只接受尚未汇总选择的搜索前缀")
    require(
        state["active_worker"] is None
        and state["pending"]
        and state["pending"]["attempts"] == []
        and state["admission"][-1]["foreign_processes"] > 0
        and state["admission"][-1]["task_key"] == state["pending"]["key"],
        "缺少未提交资源暂停证据",
    )
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
    seen, failures = set(), []
    max_error, members, post_foreign, post_unknown = 0.0, 0, 0, 0
    totals = Counter()
    require(
        all(
            not state[k]
            for k in (
                "search_scores",
                "validation_scores",
                "validation_plan",
                "shortlist",
                "selected",
            )
        ),
        "前缀已有选择状态",
    )
    require(
        state["cursor"] == len(state["completed"]) == state["costs"]["solve_jobs"],
        "搜索cursor与完成账目不符",
    )

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

        require(
            len(policies) == 776
            and len(expected_search) == 6208
            and 0 < state["cursor"] < len(expected_search),
            "完整预登记搜索范围或前缀错误",
        )
        for index, job in enumerate(expected_search[: state["cursor"] + 1]):
            panel = expected_panels["search"][job["panel"]]
            problems = data.problems(panel["ids"])
            task = BaselineTask(
                f"{state['run_id']}:search:case{index}",
                by_sha[job["policy_sha256"]],
                problems,
                tuple((p.instance_id, seed) for p in problems for seed in settings.solver_seeds),
                job["limit"],
                settings.preparation_mode,
                settings.experiment_mask,
            )
            description = json_value(task.manifest(protocol))
            key = content_hash(
                {"run_id": state["run_id"], "kind": "solve", "description": description}
            )
            if index == state["cursor"]:
                require(
                    state["pending"]["key"] == key
                    and state["pending"]["description"] == description
                    and key not in files,
                    "待提交任务不是原计划的下一个位置",
                )
                break
            require(key not in seen and key in files, "前缀任务重复或缺失")
            seen.add(key)
            record = load_checkpoint(files[key])
            require(
                record["key"] == key
                and record["kind"] == "solve"
                and record["description"] == description
                and content_hash(record) == state["completed"][key],
                "原任务身份或不可变记录改变",
            )
            checked = score_panel(task, protocol, record["outcome"], data.labels(task))
            require(json_value(asdict(checked)) == record["checked"], "独立fitness与记录不符")
            require(
                record["attempts"]
                and record["attempts"][-1]["status"] == "returned"
                and admitted[key] >= len(record["attempts"])
                and record["outcome"]["worker_pid"] in known_pids,
                "提交/返回/worker证据不完整",
            )
            if checked.failed:
                failures.append({"key": key, "error": checked.error})
            else:
                members += len(checked.members)
                by_name = {p.instance_id: p for p in problems}
                for (name, _), item in zip(
                    task.replicas, record["outcome"]["native_result"]["items"], strict=True
                ):
                    max_error = max(
                        max_error, abs(tour_cost(by_name[name], item["tour"]) - item["cost"])
                    )
            native = record["outcome"].get("native_result", {})
            for field in (
                "search_tour_evaluations",
                "completed_batches",
                "completed_construction_steps",
                "completed_ls_evaluations",
            ):
                origin = "total_tour_evaluations" if field == "search_tour_evaluations" else field
                totals[field] += native.get(origin, 0)
            for field, value in (
                ("worker_seconds", record["outcome"].get("worker_seconds", 0)),
                ("evaluator_seconds", record["evaluation_seconds"]),
                ("registration_seconds", record["outcome"].get("registration_seconds", 0)),
                ("native_actual_seconds", native.get("actual_seconds", 0)),
                ("charged_seconds", native.get("charged_seconds", 0)),
                ("overrun_seconds", native.get("overrun_seconds", 0)),
            ):
                totals[field] += value
            boundary = record["outcome"].get("gpu_boundary_after", {})
            post_foreign += (
                type(boundary.get("foreign_processes")) is int and boundary["foreign_processes"] > 0
            )
            post_unknown += boundary.get("foreign_processes") is None
    require(
        seen == set(files)
        and members == state["costs"]["valid_members"]
        and len(failures) == state["costs"]["failed_solves"],
        "完整前缀覆盖/失败账目不符",
    )
    for field, value in totals.items():
        require(
            math.isclose(value, state["costs"].get(field, 0), rel_tol=1e-12, abs_tol=1e-7),
            "资源/FE账目不平",
        )
    report = {
        "status": "snapshot_passed",
        "scope": "full baseline prefix only; no candidate selection",
        "run_id": state["run_id"],
        "calls": len(seen),
        "valid_tours": members,
        "FE": totals["search_tour_evaluations"],
        "failed_solves": len(failures),
        "frozen_policies": 776,
        "frozen_search_calls": 6208,
        "max_independent_cost_error": max_error,
        "post_foreign_samples": post_foreign,
        "post_unknown_samples": post_unknown,
        "costs": dict(totals),
        "snapshot_checkpoint_sha256": file_hash(directory / "checkpoint.json"),
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "formal_test_released": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.input.resolve(), args.output.resolve()
    require(
        directory.is_relative_to(PROJECT)
        and output.is_relative_to(PROJECT)
        and not output.exists(),
        "审计必须使用项目内静止快照和全新输出",
    )
    report = audit(directory)
    atomic_json(output, report)
    print(
        json.dumps({k: report[k] for k in ("status", "run_id", "calls", "FE", "failed_solves")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
