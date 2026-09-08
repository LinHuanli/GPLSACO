#!/usr/bin/env python3
"""不启动CUDA地重读四条件worker原始结果，核验完整任务集合、身份、路线与重放。"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from check_graph_worker import require, semantic_result  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.fitness import score_panel  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    BaselineTask,
    SolverSettings,
    SolveTask,
    WorkerProtocol,
    content_hash,
    coordinate_hash,
    file_hash,
    freeze_problems,
)


def audit(directory, database, dataset_root):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["status"] == "passed", "真实CLI尚未完成")
    require(file_hash(database) == manifest["database_sha256"], "数据库身份改变")
    require(
        all(file_hash(PROJECT / name) == digest for name, digest in manifest["sources"].items()),
        "运行源码身份改变",
    )
    catalog_path = PROJECT / manifest["catalog"]
    require(file_hash(catalog_path) == manifest["catalog_sha256"], "图目录改变")
    require(manifest["wall_clock_limit"] is None, "设置了墙钟上限")
    catalog = json.loads(catalog_path.read_text())
    records = {row["path"]: row["sha256"] for row in manifest["records"]}
    require(len(records) == len(manifest["records"]) == 48, "准备/求解任务集合不完整")
    require(
        set(records) == {p.name for p in directory.glob("*.json") if p.name != "manifest.json"},
        "存在未归集或缺失的任务文件",
    )
    require(
        all(file_hash(directory / name) == digest for name, digest in records.items()),
        "原始返回记录被修改",
    )
    program = Program.from_dict(manifest["program"])
    baseline = BaselinePolicy(**manifest["static"])
    sequence = [
        ["gp", 0, "cached"],
        ["gp", 128, "cached"],
        ["static", 64, "cached"],
        ["gp", 128, "end_to_end"],
        ["gp", 128, "cached"],
    ]
    require(manifest["sequence"] == sequence, "开发矩阵序列被缩减或改变")
    protocols, runtimes = {}, {}
    for value in manifest["workers"]:
        spec = value["protocol"]
        protocol = WorkerProtocol(
            **{
                f.name: SolverSettings(**spec[f.name]) if f.name == "settings" else spec[f.name]
                for f in fields(WorkerProtocol)
                if f.init
            }
        )
        require(json.loads(json.dumps(protocol.manifest())) == spec, "重建worker协议不符")
        require(
            file_hash(PROJECT / protocol.extension_directory / "gp_faco_ext.so")
            == protocol.binary_sha256,
            "原生二进制改变",
        )
        key = protocol.constraint_mode, protocol.graph_prior_kind
        require(key not in protocols, "重复条件worker")
        runtime = value["runtime"]
        device = [v.strip() for v in next(csv.reader([runtime["device_before_start"]]))]
        require(
            runtime["host"] == protocol.execution_host
            and runtime["start_method"] == "spawn"
            and runtime["protocol_sha256"] == protocol.sha256
            and device[0] == protocol.gpu_uuid == manifest["gpu_uuid"]
            and device[1] == protocol.gpu_model
            and device[4] == protocol.driver_version,
            "实际运行硬件/启动身份不符",
        )
        protocols[key], runtimes[key] = protocol, runtime
    require(
        set(protocols)
        == {(mode, kind) for mode in ("hard", "escape") for kind in ("ALPHA", "POPMUSIC")},
        "四条件未覆盖",
    )
    solves, routes, evaluations, hard_edges, max_error, replays = 0, 0, 0, 0, 0.0, 0
    reservoirs, escape_counts = {}, {}
    with IndexedDataset(database, dataset_root) as source:
        for (constraint, kind), protocol in protocols.items():
            for n in (500, 1000):
                ids = sorted(
                    row["instance_id"] for row in catalog["entries"] if row["dimension"] == n
                )
                require(
                    len(ids) == 16 and set(ids) <= set(source.record_ids("development", n)),
                    "真实开发实例集合不符",
                )
                problems = freeze_problems(tuple(source.load_instance(name) for name in ids))
                problem_map = {p.instance_id: p for p in problems}
                labels = {name: source.load_label(name) for name in ids}
                replicas = tuple((name, seed) for name in ids for seed in (17, 29))
                previous = []
                for i in ("prepare", *range(5)):
                    record = json.loads(
                        (directory / f"{constraint}-{kind}-{n}-{i}.json").read_text()
                    )
                    for boundary in (record["before"], record["after"]):
                        require(
                            type(boundary["foreign_processes"]) is int
                            and boundary["foreign_processes"] == 0,
                            "GPU边界出现外来/未知进程",
                        )
                    outcome = record["outcome"]
                    require(
                        outcome["worker_pid"] == runtimes[constraint, kind]["pid"]
                        and outcome["status"] == "completed"
                        and outcome["engine_generation"] == 0,
                        "同PID完整任务或缓存状态不符",
                    )
                    if i == "prepare":
                        description = {
                            "protocol_sha256": protocol.sha256,
                            "dimension": n,
                            "problems": [(p.instance_id, coordinate_hash(p)) for p in problems],
                            **protocol.graph_identity(problems),
                        }
                        require(
                            outcome["preparation_id"] == content_hash(description)
                            and json.loads(json.dumps(description)) == record["description"],
                            "准备身份未包含实际图",
                        )
                        require(
                            set(outcome["registration_fees"]) == set(ids)
                            and all(
                                type(v) in (int, float) and math.isfinite(v) and v >= 0
                                for fees in outcome["registration_fees"].values()
                                for v in fees.values()
                            ),
                            "实际准备资源缺失或无效",
                        )
                        continue
                    controller, limit, mode = sequence[i]
                    occurrence = f"graph-worker-v1:{constraint}:{kind}:{n}:{i}"
                    task = (
                        SolveTask(
                            occurrence,
                            program,
                            problems,
                            replicas,
                            preparation_mode=mode,
                            evaluation_limit_per_colony=limit,
                        )
                        if controller == "gp"
                        else BaselineTask(occurrence, baseline, problems, replicas, limit, mode)
                    )
                    require(
                        json.loads(json.dumps(task.manifest(protocol))) == record["description"],
                        "实际任务/先验/FE/随机seed身份不符",
                    )
                    checked = score_panel(task, protocol, outcome, labels)
                    require(
                        not checked.failed
                        and json.loads(json.dumps(asdict(checked))) == record["checked"],
                        "外部fitness或图可行性复核失败",
                    )
                    result = outcome["native_result"]
                    previous.append(semantic_result(result))
                    reservoirs[constraint, kind, n] = (
                        result["allocated_device_bytes"],
                        result["reserved_escape_device_bytes"],
                    )
                    for (name, _), item in zip(replicas, result["items"], strict=True):
                        max_error = max(
                            max_error,
                            abs(tour_cost(problem_map[name], item["tour"]) - item["cost"]),
                        )
                        hard_edges += n if constraint == "hard" else 0
                    for field, value in result.get("escape_counters", {}).items():
                        require(type(value) is int and value >= 0, "非法Escape整数计数")
                        escape_counts[field] = escape_counts.get(field, 0) + value
                    solves += 1
                    routes += len(replicas)
                    evaluations += limit * len(replicas)
                require(
                    previous[1] == previous[3] == previous[4],
                    "GP/Static切换及end_to_end精确重放失败",
                )
                replays += 2
    for kind in ("ALPHA", "POPMUSIC"):
        for n in (500, 1000):
            require(reservoirs["hard", kind, n] == reservoirs["escape", kind, n], "容量不匹配")
    require((solves, routes, evaluations, replays) == (40, 1280, 114688, 16), "完整矩阵计数不符")
    return {
        "status": "passed",
        "scope": "engineering_development; no formal E3 inference",
        "solve_jobs": solves,
        "preparation_jobs": 8,
        "routes": routes,
        "search_tour_evaluations": evaluations,
        "hard_tour_edges_checked": hard_edges,
        "same_pid_replay_pairs": replays,
        "max_independent_cost_error": max_error,
        "escape_counters": escape_counts,
        "workers": manifest["workers"],
        "source_manifest_sha256": file_hash(directory / "manifest.json"),
        "catalog_sha256": file_hash(catalog_path),
        "audit_source_sha256": file_hash(Path(__file__)),
        "limits": [
            "Full per-move Escape permission checked by separate native oracle tests",
            "External LKH/matching already cached; not total end-to-end timing",
            "Formal four-condition independent training/validation remains pending",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(PROJECT), "审计产物必须在工作树内")
    report = audit(args.input.resolve(), args.database.resolve(), args.dataset_root.resolve())
    atomic_json(args.output, report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "solve_jobs", "routes", "search_tour_evaluations")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
