#!/usr/bin/env python3
"""真实匹配v2面板的四条件常驻worker验收；固定次数、同PID复用及外部图可行性。"""

import argparse
import csv
import fcntl
import json
import subprocess
import sys
from concurrent.futures import TimeoutError
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.configuration_search import gpu_boundary  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.fitness import score_panel  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    BaselineTask,
    PersistentGpuWorker,
    SolveTask,
    WorkerProtocol,
    content_hash,
    coordinate_hash,
    file_hash,
    freeze_problems,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def semantic_result(value):
    """剔除实际计时；路线、完整控制状态、工作计数与Escape使用量必须精确重放。"""
    fields = (
        "completed_batches",
        "discarded_batches",
        "completed_construction_steps",
        "completed_ls_evaluations",
        "completed_constraint_rejections",
        "total_tour_evaluations",
        "control_states",
        "allocated_device_bytes",
        "reserved_escape_device_bytes",
        "constraint_mode",
        "graph_spec_id",
        "graph_edges_per_colony",
    )
    result = {key: value[key] for key in fields}
    result["incumbents"] = [
        {key: item[key] for key in ("has_incumbent", "tour", "cost")} for item in value["items"]
    ]
    if value["constraint_mode"] == "escape":
        result.update(
            {
                key: value[key]
                for key in ("escape_spec_id", "escape_edge_capacity_per_ant", "escape_counters")
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog, output = args.catalog.resolve(), args.output.resolve()
    require(all(p.is_relative_to(PROJECT) for p in (catalog, output)), "目录/产物须在工作树内")
    output.mkdir(parents=True, exist_ok=False)
    device = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu_uuid}",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    )
    model, driver = [value.strip() for value in next(csv.reader(device.splitlines()))]
    spec = json.loads(catalog.read_text())
    ids = {
        n: sorted(row["instance_id"] for row in spec["entries"] if row["dimension"] == n)
        for n in (500, 1000)
    }
    require(
        len(spec["entries"]) == 32 and all(len(v) == 16 for v in ids.values()),
        "验收固定使用32个已匹配开发实例",
    )
    program = Program((0,), (4,), feature_spec_id=2)
    baseline = BaselinePolicy(
        mne_level=3, max_mne_level=3, region=3, restart_mode="bernoulli", restart_probability=1.0
    )
    manifest = {
        "status": "running",
        "scope": "engineering_development; untrained programs",
        "catalog": str(catalog.relative_to(PROJECT)),
        "catalog_sha256": file_hash(catalog),
        "database_sha256": file_hash(args.database),
        "gpu_uuid": args.gpu_uuid,
        "sources": {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in [Path(__file__), *sorted((PROJECT / "python/gp_faco").glob("*.py"))]
        },
        "sequence": [
            ["gp", 0, "cached"],
            ["gp", 128, "cached"],
            ["static", 64, "cached"],
            ["gp", 128, "end_to_end"],
            ["gp", 128, "cached"],
        ],
        "program": program.to_dict(),
        "static": baseline.to_dict(),
        "wall_clock_limit": None,
        "workers": [],
        "records": [],
        "preparation_scope": "external LKH/matching already cached; "
        "native preparation separately recorded",
    }
    atomic_json(output / "manifest.json", manifest)
    timeouts, solves, routes, evaluations, replays = 0, 0, 0, 0, 0
    reservations = {}

    def wait(future):
        nonlocal timeouts
        while True:
            try:
                return future.result(timeout=0.001 if timeouts == 0 else 5)
            except TimeoutError:
                timeouts += 1
                print(
                    json.dumps({"event": "waiting_same_future", "timeouts": timeouts}), flush=True
                )

    with IndexedDataset(args.database, args.dataset_root) as source:
        panels = {}
        for n in (500, 1000):
            require(set(ids[n]) <= set(source.record_ids("development", n)), "图面板越过开发池")
            panels[n] = freeze_problems(tuple(source.load_instance(name) for name in ids[n]))
        for constraint in ("hard", "escape"):
            for kind in ("ALPHA", "POPMUSIC"):
                protocol = WorkerProtocol(
                    args.gpu_uuid,
                    model,
                    driver,
                    file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
                    maximum_registered_per_dimension=32,
                    constraint_mode=constraint,
                    graph_prior_kind=kind,
                    graph_catalog_path=str(catalog.relative_to(PROJECT)),
                    graph_catalog_sha256=file_hash(catalog),
                )
                with PersistentGpuWorker(protocol) as worker:
                    while True:
                        try:
                            runtime = worker.ready(timeout=5)
                            break
                        except TimeoutError:
                            print(json.dumps({"event": "waiting_worker_startup"}), flush=True)
                    manifest["workers"].append(
                        {"protocol": protocol.manifest(), "runtime": runtime}
                    )
                    atomic_json(output / "manifest.json", manifest)
                    # 子进程持有主工作树UUID锁；另一个文件描述符必须抢占失败。
                    with Path(runtime["gpu_lock_path"]).open("a") as competing:
                        try:
                            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            raise RuntimeError("活跃worker没有持有共享UUID锁")

                    def save_operation(
                        name,
                        description,
                        submit,
                        protocol=protocol,
                        runtime=runtime,
                        constraint=constraint,
                        kind=kind,
                    ):
                        before = gpu_boundary(protocol, runtime["pid"])
                        require(before["foreign_processes"] == 0, "目标GPU已有外来计算进程")
                        outcome = wait(submit())
                        record = {"description": description, "before": before, "outcome": outcome}
                        path = output / f"{constraint}-{kind}-{name}.json"
                        atomic_json(path, record)
                        record["after"] = gpu_boundary(protocol, runtime["pid"])
                        atomic_json(path, record)
                        require(
                            record["after"]["foreign_processes"] == 0, "任务结束时GPU有外来进程"
                        )
                        require(outcome["worker_pid"] == runtime["pid"], "worker PID改变")
                        require(outcome["status"] == "completed", str(outcome.get("error")))
                        return path, record

                    for n, problems in panels.items():
                        description = {
                            "protocol_sha256": protocol.sha256,
                            "dimension": n,
                            "problems": [(p.instance_id, coordinate_hash(p)) for p in problems],
                            **protocol.graph_identity(problems),
                        }
                        path, record = save_operation(
                            f"{n}-prepare", description, lambda p=problems: worker.prepare(p)
                        )
                        require(
                            record["outcome"]["preparation_id"] == content_hash(description),
                            "准备任务丢失图身份",
                        )
                        manifest["records"].append({"path": path.name, "sha256": file_hash(path)})
                        replicas = tuple(
                            (p.instance_id, seed) for p in problems for seed in (17, 29)
                        )
                        previous = []
                        for i, (controller, limit, mode) in enumerate(manifest["sequence"]):
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
                                else BaselineTask(
                                    occurrence, baseline, problems, replicas, limit, mode
                                )
                            )
                            path, record = save_operation(
                                f"{n}-{i}", task.manifest(protocol), lambda t=task: worker.submit(t)
                            )
                            checked = score_panel(
                                task,
                                protocol,
                                record["outcome"],
                                {p.instance_id: source.load_label(p.instance_id) for p in problems},
                            )
                            record["checked"] = asdict(checked)
                            atomic_json(path, record)
                            require(not checked.failed, str(checked.error))
                            result = record["outcome"]["native_result"]
                            require(record["outcome"]["engine_generation"] == 0, "发生意外缓存替换")
                            require(record["outcome"]["registered_problems"] == 16, "注册数量不符")
                            previous.append(semantic_result(result))
                            reservations[constraint, kind, n] = (
                                result["allocated_device_bytes"],
                                result["reserved_escape_device_bytes"],
                            )
                            solves += 1
                            routes += len(checked.members)
                            evaluations += result["total_tour_evaluations"]
                            manifest["records"].append(
                                {"path": path.name, "sha256": file_hash(path)}
                            )
                            atomic_json(output / "manifest.json", manifest)
                            print(
                                json.dumps(
                                    {
                                        "event": "verified",
                                        "constraint": constraint,
                                        "prior": kind,
                                        "dimension": n,
                                        "index": i,
                                    }
                                ),
                                flush=True,
                            )
                        require(
                            previous[1] == previous[3] == previous[4],
                            "同PID GP/Static切换后重放不符",
                        )
                        replays += 2
    for kind in ("ALPHA", "POPMUSIC"):
        for n in (500, 1000):
            require(
                reservations["hard", kind, n] == reservations["escape", kind, n],
                "Hard/Escape设备容量不匹配",
            )
    require(solves == 40 and routes == 1280 and evaluations == 114688, "完整次数矩阵未完成")
    require(
        all(file_hash(PROJECT / name) == digest for name, digest in manifest["sources"].items()),
        "运行期间Python源码改变",
    )
    manifest.update(
        status="passed",
        solve_jobs=solves,
        routes=routes,
        tour_evaluations=evaluations,
        same_pid_replay_pairs=replays,
        observer_timeouts=timeouts,
        registered_graphs=128,
        preparation_jobs=8,
    )
    atomic_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "solve_jobs",
                    "routes",
                    "tour_evaluations",
                    "same_pid_replay_pairs",
                    "observer_timeouts",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
