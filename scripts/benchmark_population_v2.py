#!/usr/bin/env python3
"""同一 A5000 上测完整首代种群；计时包含求解和一次外部最终路线评分。"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import Evolution, EvolutionSettings  # noqa: E402
from gp_faco.gpu_session import GpuSession  # noqa: E402
from gp_faco.program_ir import export_tree  # noqa: E402
from gp_faco.worker import faco_ants  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--measures", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or args.iterations < 1 or args.measures < 1:
        parser.error("产物必须位于项目内，评价次数和测量次数必须为正")
    evolution = Evolution(EvolutionSettings(feature_spec_id=2), 1103)
    evolution.initialize()
    programs = [
        export_tree(individual).to_dict() for individual in evolution.population[: args.population]
    ]
    if len(programs) != args.population or args.population < 2:
        parser.error("种群大小应在 2..128")
    session = GpuSession(args.gpu, "exact", extension_directory="build/v2-population-exact")
    report = {
        "status": "running",
        "host": platform.node(),
        "device": session.device,
        "population": len(programs),
        "replicas": 32,
        "ants": 128,
        "iterations": args.iterations,
        "warmups": 1,
        "measures": args.measures,
        "evolution_seed": 1103,
        "programs": programs,
        "rows": [],
    }
    atomic_json(output, report)
    try:
        with IndexedDataset(
            PROJECT / "artifacts/data/main-index-v1/instances.sqlite",
            PROJECT.parent / "Datasets/TSP",
        ) as source:
            for n in (500, 1000):
                names = source.record_ids("development", n)[64:80]
                if len(names) != 16:
                    raise ValueError("测速开发面板需要完整的 16 个实例")
                problems = [source.load_instance(name) for name in names]
                panel = [problem for problem in problems for _ in (17, 29)]
                keys = np.asarray([problem.numeric_id for problem in panel], dtype=np.uint64)
                seeds = np.asarray([seed for _ in problems for seed in (17, 29)], dtype=np.uint64)
                references = {
                    p.instance_id: source.load_label(p.instance_id).cost for p in problems
                }
                settings = session.native.FixedFacoSettings()
                settings.ants = faco_ants(n)
                serial = session.native.FacoBatchEngine(n, 32, settings)
                expanded = session.native.FacoBatchEngine(
                    n, 32, settings, population_size=len(programs)
                )
                registration = time.perf_counter()
                for engine in (serial, expanded):
                    for problem in problems:
                        engine.register_problem(
                            problem.numeric_id, np.asarray(problem.coordinates, dtype=np.float64)
                        )
                row = {
                    "dimension": n,
                    "instances": names,
                    "seeds": [17, 29],
                    "registration_seconds": time.perf_counter() - registration,
                    "total_fe": len(programs) * 32 * faco_ants(n) * args.iterations,
                    "serial_seconds": [],
                    "population_seconds": [],
                }
                report["rows"].append(row)

                def solve(
                    parallel,
                    engines=(serial, expanded),
                    keys=keys,
                    seeds=seeds,
                    n=n,
                    references=references,
                    panel=panel,
                ):
                    before = time.perf_counter()
                    if parallel:
                        results = engines[1].evaluate_population_evaluations(
                            keys, seeds, faco_ants(n) * args.iterations, programs
                        )
                    else:
                        results = [
                            engines[0].evaluate_program_evaluations(
                                keys, seeds, faco_ants(n) * args.iterations, program
                            )
                            for program in programs
                        ]
                    gaps = [
                        [
                            100
                            * (
                                tour_cost(problem, item["tour"]) / references[problem.instance_id]
                                - 1
                            )
                            for problem, item in zip(panel, result["items"], strict=True)
                        ]
                        for result in results
                    ]
                    return time.perf_counter() - before, results, gaps

                for label, parallel in (("serial", False), ("population", True)):
                    cold, results, gaps = solve(parallel)
                    row[f"{label}_cold_seconds"] = cold
                    row[f"{label}_device_bytes"] = results[0]["allocated_device_bytes"]
                    if not parallel:
                        serial_results, serial_gaps = results, gaps
                    else:
                        for expected, actual in zip(serial_results, results, strict=True):
                            for field in (
                                "control_states",
                                "completed_construction_steps",
                                "completed_ls_evaluations",
                                "total_tour_evaluations",
                            ):
                                if expected[field] != actual[field]:
                                    raise AssertionError(f"population mismatch: n={n}, {field}")
                            if [(v["tour"], v["cost"]) for v in expected["items"]] != [
                                (v["tour"], v["cost"]) for v in actual["items"]
                            ]:
                                raise AssertionError(f"population tour/cost mismatch: n={n}")
                        if serial_gaps != gaps:
                            raise AssertionError("外部评分改变")
                        row["equivalence"] = {
                            "programs": len(programs),
                            "tours": len(programs) * 32,
                            "all_tours_costs_feedback_work_fe_equal": True,
                        }
                    row["construction_steps"] = sum(
                        r["completed_construction_steps"] for r in results
                    )
                    row["ls_evaluations"] = sum(r["completed_ls_evaluations"] for r in results)
                    atomic_json(output, report)
                    print(
                        json.dumps({"dimension": n, "mode": label, "warmup_seconds": cold}),
                        flush=True,
                    )
                    for measure in range(args.measures):
                        seconds, _, _ = solve(parallel)
                        row[f"{label}_seconds"].append(seconds)
                        atomic_json(output, report)
                        print(
                            json.dumps(
                                {
                                    "dimension": n,
                                    "mode": label,
                                    "measure": measure + 1,
                                    "seconds": seconds,
                                }
                            ),
                            flush=True,
                        )
                row["serial_median_seconds"] = statistics.median(row["serial_seconds"])
                row["population_median_seconds"] = statistics.median(row["population_seconds"])
                row["speedup"] = row["serial_median_seconds"] / row["population_median_seconds"]
                row["population_fe_per_second"] = row["total_fe"] / row["population_median_seconds"]
                del solve, serial, expanded, results, serial_results
                atomic_json(output, report)
        report["status"] = "complete"
        report["generation_seconds"] = sum(r["population_median_seconds"] for r in report["rows"])
        report["serial_generation_seconds"] = sum(
            r["serial_median_seconds"] for r in report["rows"]
        )
        report["generation_speedup"] = (
            report["serial_generation_seconds"] / report["generation_seconds"]
        )
        atomic_json(output, report)
    finally:
        session.close()


if __name__ == "__main__":
    main()
