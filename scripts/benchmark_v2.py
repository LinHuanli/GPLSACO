#!/usr/bin/env python3
"""固定评价次数的 A5000 测量：一次预热、五次完整求解、独立最终评分。"""

import argparse
import fcntl
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
import numpy as np  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import faco_ants  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="build/v2-exact")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    build = (PROJECT / args.build).resolve()
    if not build.is_relative_to(PROJECT) or not args.output.resolve().is_relative_to(PROJECT):
        raise ValueError("构建和输出必须在项目内")
    lease = (PROJECT / ".tmp" / f"worker-{args.gpu}.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX)
    info = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu}",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    )
    if not info.startswith("NVIDIA RTX A5000,") or args.gpu in apps:
        raise RuntimeError("测量需要空闲 RTX A5000")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_CACHE_PATH"] = str(PROJECT / ".cache/cuda")
    sys.path.insert(0, str(build))
    cold = time.perf_counter()
    import gp_faco_ext as native

    report = {
        "host": platform.node(),
        "device": info,
        "gpu_uuid": args.gpu,
        "build": args.build,
        "iterations": args.iterations,
        "warmups": 1,
        "repeats": 5,
        "colonies": 32,
        "import_seconds": time.perf_counter() - cold,
        "rows": [],
    }
    dataset = IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    )
    policies = [
        ("static_mne8_region3", BaselinePolicy(mne_level=2, max_mne_level=2, region=3)),
        (
            "gp_representative",
            Program((0, 0, 2), (3, 8, 0), feature_spec_id=2, program_id="representative-01"),
        ),
    ]
    for n in (500, 1000):
        start = time.perf_counter()
        settings = native.FixedFacoSettings()
        settings.ants = faco_ants(n)
        engine = native.FacoBatchEngine(n, 32, settings)
        problems = [
            dataset.load_instance(name) for name in dataset.record_ids("development", n)[96:112]
        ]
        for problem in problems:
            engine.register_problem(
                problem.numeric_id, np.asarray(problem.coordinates, dtype=np.float64)
            )
        registration = time.perf_counter() - start
        keys = np.asarray([p.numeric_id for p in problems for _ in (17, 29)], dtype=np.uint64)
        seeds = np.asarray([seed for _ in problems for seed in (17, 29)], dtype=np.uint64)
        reference = [dataset.load_label(p.instance_id).cost for p in problems for _ in (17, 29)]
        for name, policy in policies:
            durations, native_durations = [], []
            for repeat in range(6):
                start = time.perf_counter()
                method = (
                    engine.evaluate_baseline_evaluations
                    if isinstance(policy, BaselinePolicy)
                    else engine.evaluate_program_evaluations
                )
                result = method(
                    keys, seeds, settings.ants * args.iterations, policy.to_dict(), "cached"
                )
                costs = [
                    tour_cost(p, item["tour"])
                    for p, item in zip(
                        [p for p in problems for _ in (17, 29)], result["items"], strict=True
                    )
                ]
                duration = time.perf_counter() - start
                if repeat == 0:
                    cold_seconds = registration + duration
                else:
                    durations.append(duration)
                    native_durations.append(result["actual_seconds"])
            row = {
                "dimension": n,
                "controller": name,
                "ants": settings.ants,
                "registration_seconds": registration,
                "cold_seconds": cold_seconds,
                "seconds": durations,
                "native_seconds": native_durations,
                "median_seconds": statistics.median(durations),
                "fe_per_second": result["total_tour_evaluations"] / statistics.median(durations),
                "device_bytes": result["allocated_device_bytes"],
                "construction_steps": result["completed_construction_steps"],
                "ls_evaluations": result["completed_ls_evaluations"],
                "tour_evaluations": result["total_tour_evaluations"],
                "gap_percent": [
                    100 * (cost / ref - 1) for cost, ref in zip(costs, reference, strict=True)
                ],
                "final_tours": [item["tour"] for item in result["items"]],
            }
            report["rows"].append(row)
            atomic_json(args.output, report)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("final_tours", "gap_percent")
                    }
                ),
                flush=True,
            )
        del engine
    dataset.close()
    lease.close()


if __name__ == "__main__":
    main()
