#!/usr/bin/env python3
"""真实spawn worker的状态/失败检查及500/1K同GPU外部fitness闭环。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import TimeoutError
from dataclasses import asdict, replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.data import Instance, Label  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.fitness import aggregate_panels, score_panel  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    PersistentGpuWorker,
    SolverSettings,
    SolveTask,
    WorkerProtocol,
    file_hash,
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def flat(name: str, n: int, shift: int) -> Instance:
    return Instance(name, tuple((float(shift + (i == 0)), 0.0) for i in range(n)))


def synthetic_task(name: str, offset: int = 0, program: Program | None = None) -> SolveTask:
    a, b = flat(f"a{offset}", 31, offset), flat(f"b{offset}", 31, offset + 2)
    return SolveTask(
        name,
        program or Program((0,), (4,)),
        (a, b),
        tuple((p.instance_id, seed) for p in (a, b) for seed in (17, 29)),
        0.2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/gpu/worker-engine/smoke"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("所有产物必须位于GPLSACO内")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("请使用新的运行目录，不能覆盖已存在的完整或部分任务")
    output.mkdir(parents=True, exist_ok=True)
    require("gp_faco_ext" not in sys.modules, "协调进程意外导入CUDA扩展")
    row = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={args.gpu_uuid}",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).splitlines()
        )
    )
    model, driver = [v.strip() for v in row]
    binary_hash = file_hash(PROJECT / "build/cuda/gp_faco_ext.so")
    records = []
    protocols, runtimes = {}, {}
    started = time.perf_counter()

    def save(task, protocol, result):
        identity = task.task_id(protocol)
        require(result["task_id"] == identity, "返回了别的任务")
        path = output / f"{identity}.json"
        path.write_text(
            json.dumps(
                {"task": task.manifest(protocol), "result": result},
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        records.append(
            {
                "task_id": identity,
                "occurrence_id": task.occurrence_id,
                "status": result["status"],
                "worker_seconds": result["worker_seconds"],
            }
        )
        print(task.occurrence_id, result["status"], flush=True)
        return result

    protocol = WorkerProtocol(
        args.gpu_uuid,
        model,
        driver,
        binary_hash,
        dimensions=(31,),
        colonies=4,
        settings=SolverSettings(ants=4),
        maximum_registered_per_dimension=4,
    )
    protocols["state_checks"] = protocol.manifest()
    with PersistentGpuWorker(protocol) as worker:
        runtime = worker.ready(timeout=30)
        runtimes["state_checks"] = runtime
        require(
            runtime["start_method"] == "spawn" and runtime["pid"] != os.getpid(),
            "没有使用真实spawn子进程",
        )
        a = replace(synthetic_task("state:A0"), budget_seconds=0.8)
        b = synthetic_task("state:B", program=Program((0,), (5,)))
        future = worker.submit(a)
        try:
            future.result(timeout=0)
            raise RuntimeError("未覆盖活动Future超时检查")
        except TimeoutError:
            pass
        try:
            worker.submit(b)
            raise AssertionError("活动任务期间接受了第二个任务")
        except RuntimeError:
            pass
        first = save(a, protocol, future.result(timeout=30))
        middle = save(b, protocol, worker.submit(b).result(timeout=30))
        again_task = replace(a, occurrence_id="state:A1")
        again = save(again_task, protocol, worker.submit(again_task).result(timeout=30))
        for result, restart in ((first, True), (middle, False), (again, True)):
            require(
                result["status"] == "completed" and result["worker_pid"] == runtime["pid"],
                "没有在同一worker完成A-B-A",
            )
            native = result["native_result"]
            for item, state in zip(native["items"], native["control_states"], strict=True):
                require(
                    item["cost"] == 2 and sorted(item["tour"]) == list(range(31)),
                    "等成本fixture输出非法",
                )
                require(
                    state["stagnant_batches"] == native["completed_batches"],
                    "跨任务保留旧stagnation",
                )
                require((state["restarts"] > 0) == restart, "跨任务保留旧重启计数")
        zero_task = replace(a, occurrence_id="state:zero", budget_seconds=0)
        zero = save(zero_task, protocol, worker.submit(zero_task).result(timeout=30))
        labels = {p.instance_id: Label(tuple(range(31)), 2) for p in a.problems}
        missing = score_panel(zero_task, protocol, zero, labels)
        require(
            missing.failed
            and math.isinf(aggregate_panels((zero_task,), protocol, (missing,), (31,))),
            "缺失截止前解没有保留为失败任务",
        )
        require(
            all(s["archive_size"] == 0 for s in zero["native_result"]["control_states"]),
            "零预算返回旧控制状态",
        )
        for index, offset in enumerate((10, 20)):
            task = synthetic_task(f"state:cache{index}", offset)
            result = save(task, protocol, worker.submit(task).result(timeout=30))
        require(
            result["engine_generation"] == 1 and result["registered_problems"] == 2,
            "静态缓存没有在任务边界按容量重建",
        )
        bad_instance = Instance("zero-length", ((0.0, 0.0),) * 31)
        bad = SolveTask(
            "state:invalid-problem",
            a.program,
            (bad_instance,),
            tuple((bad_instance.instance_id, seed) for seed in range(4)),
            0.2,
        )
        failed = save(bad, protocol, worker.submit(bad).result(timeout=30))
        require(failed["status"] == "failed" and "error" in failed, "worker丢失实际准备失败")
        recovery = replace(a, occurrence_id="state:after-failure", budget_seconds=0.2)
        recovered = save(recovery, protocol, worker.submit(recovery).result(timeout=30))
        require(
            recovered["status"] == "completed" and recovered["worker_pid"] == runtime["pid"],
            "可恢复输入错误污染后续worker任务",
        )

    # 同一个32-colony、32-ant协议完成两个真实开发规模，标签只留在此协调进程。
    protocol = WorkerProtocol(args.gpu_uuid, model, driver, binary_hash)
    protocols["development_panel"] = protocol.manifest()
    tasks, scores, panels = [], [], []
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as data:
        with PersistentGpuWorker(protocol) as worker:
            runtimes["development_panel"] = worker.ready(timeout=30)
            for n, budget in ((500, 0.6), (1000, 1.2)):
                ids = data.record_ids("development", n)[:16]
                problems = tuple(data.load_instance(record_id) for record_id in ids)
                task = SolveTask(
                    f"development:individual0:scale{n}",
                    Program((0,), (4,)),
                    problems,
                    tuple((p.instance_id, seed) for p in problems for seed in (17, 29)),
                    budget,
                )
                result = save(task, protocol, worker.submit(task).result(timeout=30))
                labels = {p.instance_id: data.load_label(p.instance_id) for p in problems}
                score = score_panel(task, protocol, result, labels)
                require(not score.failed, f"开发面板外部核验失败: {score.error}")
                tasks.append(task)
                scores.append(score)
                panels.append(
                    {
                        "task_id": task.task_id(protocol),
                        "dimension": n,
                        "score": asdict(score),
                        "native_summary": {
                            k: v for k, v in result["native_result"].items() if k != "items"
                        },
                    }
                )
    fitness = aggregate_panels(tuple(tasks), protocol, tuple(reversed(scores)), (500, 1000))
    require(math.isfinite(fitness), "真实两规模fitness不有限")
    require("gp_faco_ext" not in sys.modules, "协调进程在评价后导入了CUDA扩展")
    report = {
        "status": "passed",
        "scope": "spawn worker and external fitness; DEAP training/E1-E4 pending",
        "coordinator_pid": os.getpid(),
        "coordinator_loaded_native_extension": False,
        "timeout_kept_same_future": True,
        "busy_submission_rejected": True,
        "a_b_a_state_checks": True,
        "bounded_registry_rebuilt_at_boundary": True,
        "preparation_failure_recorded_and_recovered": True,
        "missing_incumbent_fitness_is_infinite": True,
        "protocols": protocols,
        "runtimes": runtimes,
        "records": records,
        "development_panels": panels,
        "development_untrained_macro_reference_gap_percent": fitness,
        "coordinator_wall_seconds": time.perf_counter() - started,
        "inputs": {
            name: file_hash(PROJECT / name)
            for name in (
                "python/gp_faco/worker.py",
                "python/gp_faco/fitness.py",
                "scripts/worker_smoke.py",
                "tests/python/test_worker_protocol.py",
                "provenance/splits.v1.json",
            )
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print("两规模开发fitness:", fitness, flush=True)


if __name__ == "__main__":
    main()
