#!/usr/bin/env python3
"""真实首代种群、固定 FE 的生产路径测量；原始返回供新旧构建直接比较。"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import source
from gp_faco.checkpoint import atomic_json
from gp_faco.data import tour_cost
from gp_faco.evolution import Evolution, EvolutionSettings
from gp_faco.gpu_session import GpuSession
from gp_faco.program_ir import export_tree
from gp_faco.worker import faco_ants


def signature(native):
    return {
        "items": [{"tour": r["tour"], "cost": r["cost"]} for r in native["items"]],
        **{k: native[k] for k in ("control_states", "completed_construction_steps",
                                  "completed_ls_evaluations", "total_tour_evaluations")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--chunks", type=int, nargs="+", default=[128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[500])
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--measures", type=int, default=3)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT):
        parser.error("产物须在项目内")
    evolution = Evolution(EvolutionSettings(feature_spec_id=2), 1103)
    evolution.initialize()
    programs = [export_tree(p).to_dict() for p in evolution.population[:args.population]]
    reference = json.loads(args.compare.read_text()) if args.compare else None
    session = GpuSession(args.gpu, "exact", colonies=args.instances * len(args.seeds),
                         extension_directory=args.build)
    report = {"status": "running", "device": session.device, "population": len(programs),
              "instances": args.instances, "aco_seeds": args.seeds, "rows": []}
    try:
        with source() as data:
            for n in args.dimensions:
                names = data.record_ids("development", n)[64:64 + args.instances]
                problems = {name: data.load_instance(name) for name in names}
                labels = {name: data.load_label(name).cost for name in names}
                pairs = [(name, seed) for name in names for seed in args.seeds]
                for iterations in args.iterations:
                    expected = None
                    for chunk in args.chunks:
                        measurements = []
                        outputs = None
                        # 长预算只测一次；短预算一个热身和指定次数。全部测量固定 FE。
                        for measure in range(args.measures + 1 if iterations <= 100 else 1):
                            started = time.perf_counter()
                            outputs = []
                            registration = 0
                            for left in range(0, len(programs), chunk):
                                native, timing = session.solve_population(
                                    problems, pairs, iterations, programs[left:left + chunk])
                                outputs.extend(native)
                                registration += timing["registration_seconds"]
                            solve = time.perf_counter() - started
                            before_score = time.perf_counter()
                            gaps = [[100 * (tour_cost(problems[name], r["tour"]) / labels[name] - 1)
                                     for (name, _), r in zip(pairs, native["items"], strict=True)]
                                    for native in outputs]
                            score = time.perf_counter() - before_score
                            measurements.append({"seconds": solve + score, "solve_seconds": solve,
                                                 "scoring_seconds": score, "registration_seconds": registration})
                        actual = [signature(r) for r in outputs]
                        if expected is not None and actual != expected:
                            raise AssertionError("分片改变最终路线、状态或工作量")
                        expected = actual
                        key = f"n{n}-h{iterations}"
                        raw_path = args.output.with_name(args.output.stem + f"-{key}.raw.json")
                        if reference:
                            old_path = Path(next(r["raw"] for r in reference["rows"]
                                                 if r["dimension"] == n and r["iterations"] == iterations))
                            if json.loads(old_path.read_text()) != actual:
                                raise AssertionError(f"构建改变 {key} 的轨迹或工作量")
                        atomic_json(raw_path, actual)
                        warm = measurements[1:] if len(measurements) > 1 else measurements
                        row = {"dimension": n, "iterations": iterations, "chunk": chunk,
                               "measurements": measurements,
                               "median_seconds": statistics.median(r["seconds"] for r in warm),
                               "development_mean_gap_percent": statistics.mean(v for values in gaps for v in values),
                               "total_fe": sum(r["total_tour_evaluations"] for r in outputs),
                               "device_bytes": max(r["allocated_device_bytes"] for r in outputs),
                               "raw": str(raw_path.resolve()), "same_reference": bool(reference)}
                        if args.profile and iterations == min(args.iterations) and chunk == args.chunks[0]:
                            engine = session.population_engines[n, chunk]
                            keys = np.asarray([problems[name].numeric_id for name, _ in pairs], dtype=np.uint64)
                            seeds = np.asarray([seed for _, seed in pairs], dtype=np.uint64)
                            stages = {}
                            profiled = []
                            for left in range(0, len(programs), chunk):
                                batch = engine.evaluate_population_evaluations(
                                    keys, seeds, faco_ants(n) * iterations, programs[left:left + chunk],
                                    profile_events_only=True)
                                profiled.extend(signature(r) for r in batch)
                                for stage, ms in batch[0]["production_kernel_milliseconds"].items():
                                    stages[stage] = stages.get(stage, 0) + ms
                            if profiled != actual:
                                raise AssertionError("生产 event 插桩改变求解")
                            row["stages_ms"] = stages
                        report["rows"].append(row)
                        atomic_json(args.output, report)
                        print(json.dumps(row), flush=True)
            report["status"] = "passed"
            atomic_json(args.output, report)
    finally:
        session.close()


if __name__ == "__main__":
    main()
