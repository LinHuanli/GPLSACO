#!/usr/bin/env python3
"""真实32-colony开发面板上的Static–GP–Rule复用；每次返回先落盘再核验。"""

import argparse
import csv
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
    file_hash,
    freeze_problems,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("产物必须位于GPLSACO内")
    output.mkdir(parents=True, exist_ok=False)
    model, driver = next(
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
    protocol = WorkerProtocol(
        args.gpu_uuid,
        model.strip(),
        driver.strip(),
        file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
        maximum_registered_per_dimension=128,
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    static = BaselinePolicy(
        mne_level=1, max_mne_level=1, region=2, restart_mode="bernoulli", restart_probability=0.5
    )
    rule = BaselinePolicy(
        kind="rule",
        max_mne_level=3,
        region=3,
        stagnation_step=1,
        restart_stagnation=1,
        restart_cooldown=1,
    )
    atomic_json(
        output / "manifest.json",
        {
            "scope": "engineering_development",
            "protocol": protocol.manifest(),
            "database_sha256": file_hash(database),
            "sources": {
                str(p.relative_to(PROJECT)): file_hash(p)
                for p in sorted((PROJECT / "python/gp_faco").glob("*.py"))
            },
            "entrypoint_sha256": file_hash(Path(__file__)),
            "sequence_per_dimension": ["static64", "gp64", "rule128", "static64", "rule128"],
            "static": static.to_dict(),
            "rule": rule.to_dict(),
            "wall_clock_limit": None,
        },
    )
    records, timeouts = [], 0

    def wait(future):
        nonlocal timeouts
        while True:
            try:
                return future.result(timeout=5)
            except TimeoutError:
                timeouts += 1
                print(json.dumps({"event": "waiting_same_future"}), flush=True)

    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        with PersistentGpuWorker(protocol) as worker:
            while True:
                try:
                    runtime = worker.ready(timeout=5)
                    break
                except TimeoutError:
                    print(json.dumps({"event": "waiting_worker_startup"}), flush=True)
            atomic_json(output / "runtime.json", runtime)
            for n in protocol.dimensions:
                ids = sorted(source.record_ids("development", n))[:16]
                problems = freeze_problems(tuple(source.load_instance(name) for name in ids))
                replicas = tuple((p.instance_id, seed) for p in problems for seed in (17, 29))
                pairs = []
                for i, (kind, limit) in enumerate(
                    [("static", 64), ("gp", 64), ("rule", 128), ("static", 64), ("rule", 128)]
                ):
                    occurrence = f"baseline-worker-v1:{n}:{i}"
                    if kind == "gp":
                        task = SolveTask(
                            occurrence,
                            Program((0,), (4,), feature_spec_id=2),
                            problems,
                            replicas,
                            preparation_mode="cached",
                            evaluation_limit_per_colony=limit,
                        )
                    else:
                        task = BaselineTask(
                            occurrence,
                            static if kind == "static" else rule,
                            problems,
                            replicas,
                            limit,
                        )
                    before = gpu_boundary(protocol, runtime["pid"])
                    if before["foreign_processes"]:
                        raise RuntimeError("目标GPU有外来计算进程，任务未提交")
                    outcome = wait(worker.submit(task))
                    path = output / f"{n}-{i}.json"
                    record = {"task": task.manifest(protocol), "before": before, "outcome": outcome}
                    atomic_json(path, record)  # 保留真实返回，包括失败。
                    record["after"] = gpu_boundary(protocol, runtime["pid"])
                    checked = score_panel(
                        task,
                        protocol,
                        outcome,
                        {p.instance_id: source.load_label(p.instance_id) for p in problems},
                    )
                    record["checked"] = asdict(checked)
                    atomic_json(path, record)
                    if checked.failed:
                        raise RuntimeError(checked.error)
                    assert outcome["worker_pid"] == runtime["pid"]
                    assert record["after"]["foreign_processes"] == 0
                    pairs.append(outcome["native_result"])
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "event": "verified",
                                "dimension": n,
                                "index": i,
                                "kind": kind,
                                "members": len(checked.members),
                            }
                        ),
                        flush=True,
                    )
                for left, right in ((pairs[0], pairs[3]), (pairs[2], pairs[4])):
                    assert left["control_states"] == right["control_states"]
                    assert [(v["tour"], v["cost"]) for v in left["items"]] == [
                        (v["tour"], v["cost"]) for v in right["items"]
                    ]
                    for field in ("completed_construction_steps", "completed_ls_evaluations"):
                        assert left[field] == right[field]
    summary = {
        "status": "passed",
        "solve_jobs": len(records),
        "worker": runtime,
        "route_count": sum(len(r["checked"]["members"]) for r in records),
        "tour_evaluations": sum(
            r["outcome"]["native_result"]["total_tour_evaluations"] for r in records
        ),
        "observer_timeouts": timeouts,
        "same_controller_repeat_pairs": 4,
        "record_sha256": {str(p.name): file_hash(p) for p in sorted(output.glob("*-*.json"))},
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
