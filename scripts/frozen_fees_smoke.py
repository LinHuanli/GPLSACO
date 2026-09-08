#!/usr/bin/env python3
"""在真实worker重建前后复用同一准备费用和任务身份，核验独立路线与扣费。"""

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.fitness import score_panel  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    PersistentGpuWorker,
    SolverSettings,
    SolveTask,
    WorkerProtocol,
    file_hash,
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/gpu/gp-training/frozen-fees"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("产物必须在GPLSACO内")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("必须使用新的输出目录")
    output.mkdir(parents=True, exist_ok=True)
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
    protocol = WorkerProtocol(
        args.gpu_uuid,
        row[0].strip(),
        row[1].strip(),
        file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
        dimensions=(500,),
        colonies=4,
        settings=SolverSettings(ants=4),
    )
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as data:
        problems = tuple(
            data.load_instance(record) for record in data.record_ids("development", 500)[:2]
        )
        labels = {p.instance_id: data.load_label(p.instance_id) for p in problems}
    base = SolveTask(
        "fee-recovery:same-task",
        Program((0,), (5,)),
        problems,
        tuple((p.instance_id, seed) for p in problems for seed in (17, 29)),
        0.25,
    )
    runs, preparations, runtimes = [], [], []
    task = None
    for attempt in range(2):
        with PersistentGpuWorker(protocol) as worker:
            runtimes.append(worker.ready(timeout=30))
            prepared = worker.prepare(problems).result(timeout=30)
            require(prepared["status"] == "completed", "实际准备失败")
            preparations.append(prepared)
            if task is None:
                task = replace(
                    base,
                    preparation_charges=tuple(
                        (name, fee["cheap_seconds"], fee["preparation_seconds"])
                        for name, fee in prepared["registration_fees"].items()
                    ),
                )
            result = worker.submit(task).result(timeout=30)
            score = score_panel(task, protocol, result, labels)
            require(
                not score.failed and result["native_result"]["preparation_completed"],
                f"重建后任务/费用/路线不符: {score.error}",
            )
            expected = math.fsum(a + b for _, a, b in task.preparation_charges)
            require(
                math.isclose(result["native_result"]["charged_seconds"], expected, abs_tol=1e-12),
                "重新测量改变了原有扣费",
            )
            runs.append({"result": result, "score": asdict(score)})
            if attempt == 0:
                omitted = worker.submit(base).result(timeout=30)
                require(omitted["status"] == "failed", "省略冻结表后发生隐式费用复用")
                changed = list(task.preparation_charges)
                name, cheap, full = changed[0]
                changed[0] = name, cheap, full + 0.001
                conflict = worker.submit(replace(task, preparation_charges=tuple(changed))).result(
                    timeout=30
                )
                require(conflict["status"] == "failed", "同实例冻结费用被修改")
    require(runtimes[0]["pid"] != runtimes[1]["pid"], "未实际重建worker")
    require(runs[0]["result"]["task_id"] == runs[1]["result"]["task_id"], "恢复改变了任务身份")
    report = {
        "status": "passed",
        "protocol": protocol.manifest(),
        "task": task.manifest(protocol),
        "runtimes": runtimes,
        "preparations": preparations,
        "runs": runs,
        "omitted_table_rejected": True,
        "conflicting_table_rejected": True,
        "same_task_and_charges_after_worker_restart": True,
        "scope": "real frozen-fee recovery prerequisite; GP loop verification separate",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("真实worker重建后任务与扣费一致，8条开发路线独立核验通过", flush=True)


if __name__ == "__main__":
    main()
