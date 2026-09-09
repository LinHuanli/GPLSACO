#!/usr/bin/env python3
# ruff: noqa: E402
"""回放预实验真实种群，比较分片、生产计时和新旧构建的直接结果。"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read, source
from gp_faco.checkpoint import atomic_json
from gp_faco.data import tour_cost
from gp_faco.gpu_session import GpuSession


def signature(result):
    return {
        "items": [{"tour": r["tour"], "cost": r["cost"]} for r in result["items"]],
        **{
            k: result[k]
            for k in (
                "control_states",
                "completed_construction_steps",
                "completed_ls_evaluations",
                "total_tour_evaluations",
            )
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    p.add_argument("--runs", nargs="+", default=["joint_single-s1103", "conditional_three-s1103"])
    p.add_argument("--generations", type=int, nargs="+", default=[1, 10])
    p.add_argument("--iterations", type=int, nargs="+", default=[100])
    p.add_argument("--chunks", type=int, nargs="+", default=[128])
    p.add_argument("--measures", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        p.error("输出必须位于项目内")
    output.parent.mkdir(parents=True, exist_ok=True)
    pilot = PROJECT / "artifacts/v3/gp-representation-pilot"
    session = GpuSession(args.gpu, "exact", colonies=16, extension_directory=args.build)
    report = {"status": "running", "build": args.build, "device": session.device, "rows": []}
    reference = read(args.compare) if args.compare else None
    try:
        with source() as data:
            for run in args.runs:
                for generation in args.generations:
                    job = read(
                        pilot
                        / "jobs"
                        / f"population-{run}-training-g{generation:02d}-p000-c000/job.json"
                    )
                    pairs, controllers = job["pairs"], job["controllers"]
                    problems = {name: data.load_instance(name) for name, _ in pairs}
                    labels = {name: data.load_label(name).cost for name in problems}
                    for iterations in args.iterations:
                        case = f"{run}-g{generation:02d}-h{iterations}"
                        expected = None
                        if reference:
                            old = next(r for r in reference["rows"] if r["case"] == case)
                            expected = read(Path(old["raw"]))
                        for chunk in args.chunks:
                            timings = []
                            for _measure in range(args.measures + 1 if iterations <= 100 else 1):
                                start = time.perf_counter()
                                native, registration = [], 0.0
                                for left in range(0, len(controllers), chunk):
                                    batch, timing = session.solve_population(
                                        problems,
                                        pairs,
                                        iterations,
                                        controllers[left : left + chunk],
                                    )
                                    native.extend(batch)
                                    registration += timing["registration_seconds"]
                                solve = time.perf_counter() - start
                                start = time.perf_counter()
                                gaps = [
                                    100
                                    * (tour_cost(problems[name], item["tour"]) / labels[name] - 1)
                                    for r in native
                                    for (name, _), item in zip(pairs, r["items"], strict=True)
                                ]
                                score = time.perf_counter() - start
                                timings.append(
                                    {
                                        "seconds": solve + score,
                                        "solve_seconds": solve,
                                        "scoring_seconds": score,
                                        "registration_seconds": registration,
                                    }
                                )
                            actual = [signature(r) for r in native]
                            if expected is not None and actual != expected:
                                raise AssertionError(
                                    f"{case}/chunk{chunk}: 路线、控制状态或工作量改变"
                                )
                            expected = actual
                            raw = output.parent / f"{output.stem}-{case}.raw.json"
                            atomic_json(raw, actual)
                            row = {
                                "case": case,
                                "chunk": chunk,
                                "measurements": timings,
                                "median_seconds": statistics.median(
                                    t["seconds"] for t in (timings[1:] or timings)
                                ),
                                "gap_percent": statistics.mean(gaps),
                                "raw": str(raw),
                                "total_fe": sum(r["total_tour_evaluations"] for r in native),
                                "device_bytes": max(r["allocated_device_bytes"] for r in native),
                                "same_reference": reference is not None,
                            }
                            if args.profile:
                                stages, checked = {}, []
                                keys = np.asarray(
                                    [problems[name].numeric_id for name, _ in pairs],
                                    dtype=np.uint64,
                                )
                                seeds = np.asarray([seed for _, seed in pairs], dtype=np.uint64)
                                engine = session.population_engines[500, chunk]
                                for left in range(0, len(controllers), chunk):
                                    batch = engine.evaluate_population_evaluations(
                                        keys,
                                        seeds,
                                        128 * iterations,
                                        controllers[left : left + chunk],
                                        profile_events_only=True,
                                    )
                                    checked.extend(signature(r) for r in batch)
                                    for key, ms in batch[0][
                                        "production_kernel_milliseconds"
                                    ].items():
                                        stages[key] = stages.get(key, 0) + ms
                                if checked != actual:
                                    raise AssertionError("生产布局event测量改变结果")
                                row["stages_ms"] = stages
                            report["rows"].append(row)
                            atomic_json(output, report)
                            print(
                                json.dumps({k: v for k, v in row.items() if k != "measurements"}),
                                flush=True,
                            )
        report["status"] = "passed"
        atomic_json(output, report)
    finally:
        session.close()


if __name__ == "__main__":
    main()
