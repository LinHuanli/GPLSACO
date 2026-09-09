#!/usr/bin/env python3
"""预先登记的数值质量面板或 24 控制器 FE 曲线；一条轨迹只运行一次。"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.experiment_v2 import (  # noqa: E402
    CHECKPOINTS,
    GPU_BASELINE,
    batches,
    choose_horizon,
    predeclare,
    representative_program,
    write_once,
)
from gp_faco.gpu_session import GpuSession  # noqa: E402
from gp_faco.result_journal import ResultJournal  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("quality", "curves"))
    parser.add_argument("--backend", choices=("exact", "fp32", "fp32_fast"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--directory", type=Path, default=PROJECT / "artifacts/v2/protocol")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("运行目录必须在项目内")
    output = directory / args.stage / args.backend
    output.mkdir(parents=True, exist_ok=True)
    source = IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    )
    declaration = predeclare(source, directory)
    policies = (
        {GPU_BASELINE: None, "gp_representative": representative_program().to_dict()}
        if args.stage == "quality"
        else {
            f"policy-{i + 1:02d}": p
            for i, p in enumerate(
                json.loads((PROJECT / "configs/controller_families_v2.json").read_text())[
                    "calibration_policies"
                ]
            )
        }
    )
    panels = declaration[args.stage]
    checkpoints = CHECKPOINTS if args.stage == "curves" else (5000,)
    manifest = {
        "experiment_version": 2,
        "stage": args.stage,
        "backend": args.backend,
        "panels": panels,
        "controllers": policies,
        "iterations": 5000,
        "checkpoints": checkpoints,
        "ants": "64*ceil(sqrt(n)/16)",
        "wall_clock_limit": None,
    }
    write_once(output / "manifest.json", manifest)
    journal = ResultJournal(output / "results.jsonl")
    completed = {record["job_id"]: record for _, record in journal.trailing(0)}
    session = None
    try:
        for panel in panels:
            n = panel["dimension"]
            problems = {name: source.load_instance(name) for name in panel["ids"]}
            # 标签不传入 GpuSession；每个最终/曲线 tour 在本进程评分一次。
            references = {name: source.load_label(name).cost for name in panel["ids"]}
            for controller, policy in policies.items():
                for index, pairs in enumerate(batches(panel)):
                    job = f"n{n}-{controller}-batch{index + 1:02d}"
                    if job in completed:
                        continue
                    if session is None:
                        session = GpuSession(args.gpu, args.backend)
                    started = time.perf_counter()
                    try:
                        result, timing = session.solve(
                            problems, pairs, 5000, policy, checkpoints=checkpoints
                        )
                    except Exception as error:
                        # 求解失败也是预登记面板的结果，不能删除后重抽更好的案例。
                        record = {
                            "job_id": job,
                            "rows": [
                                {
                                    "dimension": n,
                                    "controller": controller,
                                    "instance_id": name,
                                    "seed": seed,
                                    "iterations": it,
                                    "status": "failed",
                                    "error": str(error),
                                }
                                for it in checkpoints
                                for name, seed in pairs
                            ],
                            "native_result": None,
                            "timing": {},
                            "device": session.device,
                            "end_to_end_seconds": time.perf_counter() - started,
                        }
                        journal.append(record)
                        completed[job] = record
                        print(
                            json.dumps({"job_id": job, "status": "failed", "error": str(error)}),
                            flush=True,
                        )
                        continue
                    rows = []
                    for checkpoint in result["checkpoints"]:
                        for (name, seed), tour in zip(pairs, checkpoint["tours"], strict=True):
                            try:
                                cost = tour_cost(problems[name], tour)
                                row = {
                                    "status": "completed",
                                    "cost": cost,
                                    "gap_percent": 100 * (cost / references[name] - 1),
                                }
                            except (ValueError, TypeError) as error:
                                row = {"status": "failed", "error": str(error)}
                            rows.append(
                                {
                                    "dimension": n,
                                    "controller": controller,
                                    "instance_id": name,
                                    "seed": seed,
                                    "iterations": checkpoint["iterations"],
                                    **row,
                                }
                            )
                    record = {
                        "job_id": job,
                        "rows": rows,
                        "native_result": result,
                        "timing": timing,
                        "end_to_end_seconds": time.perf_counter() - started,
                        "device": session.device,
                    }
                    journal.append(record)
                    completed[job] = record
                    # 精简进度不含完整 tour；恢复直接读追加日志。
                    print(
                        json.dumps(
                            {
                                "job_id": job,
                                "completed_jobs": len(completed),
                                "seconds": record["end_to_end_seconds"],
                            }
                        ),
                        flush=True,
                    )
                    atomic_json(
                        output / "progress.json",
                        {"status": "running", "completed_jobs": len(completed)},
                    )
        rows = [row for record in completed.values() for row in record["rows"]]
        summary = {
            "status": "complete",
            "backend": args.backend,
            "stage": args.stage,
            "completed_jobs": len(completed),
            "rows": rows,
            "total_seconds": sum(v["end_to_end_seconds"] for v in completed.values()),
        }
        if args.stage == "curves":
            summary["selection"] = choose_horizon(
                rows, controllers=list(policies), expected_panels=panels
            )
        atomic_json(output / "summary.json", summary)
        atomic_json(
            output / "progress.json", {"status": "complete", "completed_jobs": len(completed)}
        )
    finally:
        if session:
            session.close()
        journal.close()
        source.close()


if __name__ == "__main__":
    main()
