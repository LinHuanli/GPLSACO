#!/usr/bin/env python3
"""原始 2022 FACO 的预登记 TSPLIB 复现；每个实例 30 次、5000 迭代、8 线程。"""

import argparse
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import validate_tour  # noqa: E402
from gp_faco.experiment_v2 import write_once  # noqa: E402
from gp_faco.tsplib import parse_problem  # noqa: E402

INSTANCES = ("att532", "u1817", "pr2392", "fl3795", "fnl4461", "rl5915")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/v2/faco-2022-tsplib")
    parser.add_argument("--binary", type=Path, default=PROJECT / "build/v2-cpu/faco_2022")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or not args.binary.resolve().is_relative_to(PROJECT):
        raise ValueError("产物与构建必须位于项目内")
    output.mkdir(parents=True, exist_ok=True)
    archive = PROJECT / ".tmp/worktrees/e4-transfer/.deps/tsplib/official-v1/ALL_tsp.tar.gz"
    # 参考值只留在协调端评分，原生入口完全不读取此文件。
    references = json.loads((PROJECT.parent / "references/FocusedACO/best-known.json").read_text())
    manifest = {
        "experiment_version": 2,
        "method": "original_faco_2022",
        "host": platform.node(),
        "instances": INSTANCES,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "threads": 8,
        "seed_schedule": "1..repeats",
        "ant_rule": "64*ceil(sqrt(n)/16)",
        "beta": 1,
        "retention": 0.5,
        "p_best": 0.1,
        "source_probability": 0.01,
        "primary_width": 16,
        "backup_width": 64,
        "ls_width": 20,
        "mne": 8,
        "wall_clock_limit": None,
        "algorithm_source": "references/FocusedACO",
        "initialization": "native par_build_initial_routes; omp_get_num_procs routes",
    }
    write_once(output / "manifest.json", manifest)
    rows = []
    with tarfile.open(archive, "r:gz") as bundle:
        for name in INSTANCES:
            text = gzip.decompress(bundle.extractfile(f"{name}.tsp.gz").read()).decode()
            problem = parse_problem(text)
            path = output / f"{name}.tsp"
            if not path.exists():
                path.write_text(text)
            for seed in range(1, args.repeats + 1):
                directory = output / name / f"seed-{seed:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                result_path = directory / "result.json"
                scored = directory / "score.json"
                if scored.exists():
                    row = json.loads(scored.read_text())
                else:
                    if not result_path.exists():
                        with (directory / "native.log").open("w") as log:
                            subprocess.run(
                                [
                                    str(args.binary),
                                    "--problem",
                                    str(path),
                                    "--seed",
                                    str(seed),
                                    "--threads",
                                    "8",
                                    "--iterations",
                                    str(args.iterations),
                                    "--min-new-edges",
                                    "8",
                                    "--beta",
                                    "1",
                                    "--rho",
                                    "0.5",
                                    "--cand-list-size",
                                    "16",
                                    "--backup-list-size",
                                    "64",
                                    "--ls-cand-list-size",
                                    "20",
                                    "--p-best",
                                    "0.1",
                                    "--gbest-as-source-prob",
                                    "0.01",
                                    "--results-dir",
                                    str(directory),
                                ],
                                cwd=PROJECT,
                                env={**os.environ, "TMPDIR": str(PROJECT / ".tmp")},
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                    result = json.loads(result_path.read_text())
                    validate_tour(result["tour"], problem.dimension)
                    cost = sum(
                        problem.distance(result["tour"][i - 1], node)
                        for i, node in enumerate(result["tour"])
                    )
                    row = {
                        "instance": name,
                        "dimension": problem.dimension,
                        "seed": seed,
                        "cost": cost,
                        "reference_cost": references[name],
                        "gap_percent": 100 * (cost / references[name] - 1),
                        **{
                            key: result[key]
                            for key in (
                                "solver_seconds",
                                "total_seconds",
                                "search_tour_evaluations",
                                "initial_route_count",
                            )
                        },
                        "ants": result["args"]["ants"],
                    }
                    atomic_json(scored, row)
                rows.append(row)
                summaries = []
                for instance in INSTANCES:
                    values = [r for r in rows if r["instance"] == instance]
                    if values:
                        summaries.append(
                            {
                                "instance": instance,
                                "repeats": len(values),
                                "ants": values[0]["ants"],
                                "mean_gap_percent": statistics.mean(
                                    r["gap_percent"] for r in values
                                ),
                                "best_gap_percent": min(r["gap_percent"] for r in values),
                                "mean_seconds": statistics.mean(r["total_seconds"] for r in values),
                                "initial_route_count": values[0]["initial_route_count"],
                            }
                        )
                atomic_json(
                    PROJECT / "results/v2/faco_2022_tsplib.json",
                    {
                        "status": "complete"
                        if len(rows) == len(INSTANCES) * args.repeats
                        else "running",
                        "manifest": manifest,
                        "summary": summaries,
                        "runs": rows,
                    },
                )
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
