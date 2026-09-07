#!/usr/bin/env python3
"""在开发样本上验证未改动的原生 FACO 内核；不作正式性能比较。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.data import read_record, tour_cost, write_explicit_tsplib  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=PROJECT / "build/cpu")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/native/smoke")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--variant", choices=["native", "native-edge-guard"], default="native")
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to(PROJECT) or args.iterations < 1:
        parser.error("输出须在项目内，iterations 须为正")
    root.mkdir(parents=True, exist_ok=True)
    if args.variant == "native-edge-guard":
        expected_source = json.loads((PROJECT / "provenance/native_edge_guard.json").read_text())[
            "modified_sha256"
        ]
    else:
        expected_source = json.loads((PROJECT / "provenance/sources.lock.json").read_text())[
            "adaptive_faco"
        ]["files"]["src/local_search.cpp"]["sha256"]
    summaries = []
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "TMPDIR": str(PROJECT / ".tmp")}
    for n, folder, filename in (
        (500, "tsp_500", "tsp500_uniform_16k_1.txt"),
        (1000, "tsp_1k", "tsp1000_uniform_8k_1.txt"),
    ):
        data_path = PROJECT.parent / "Datasets/TSP/train_dataset/tsp" / folder / filename
        instance, label = read_record(data_path)
        input_path = root / f"development-{n}.tsp"
        started = time.perf_counter()
        write_explicit_tsplib(instance, input_path)
        conversion_seconds = time.perf_counter() - started
        input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        for algorithm in ("mfaco", "faco_apt"):
            for seed in (17, 29, 43):
                case = root / f"{n}-{algorithm}-{seed}"
                case.mkdir(parents=True, exist_ok=True)
                parameters = [
                    "--alg",
                    algorithm,
                    "--problem",
                    str(input_path),
                    "--ants",
                    "32",
                    "--threads",
                    "1",
                    "--seed",
                    str(seed),
                    "--iterations",
                    str(args.iterations),
                    "--repeat",
                    "1",
                    "--cand-list-size",
                    "16",
                    "--backup-list-size",
                    "64",
                    "--ls-cand-list-size",
                    "20",
                    "--local-search",
                    "1",
                    "--min-new-edges",
                    "8",
                    "--beta",
                    "1",
                    "--rho",
                    "0.5",
                    "--p-best",
                    "0.1",
                    "--gbest-as-source-prob",
                    "0.01",
                    "--keep-better-ant-sol",
                    "1",
                    "--source-sol-local-update",
                    "1",
                    "--picture",
                    "0",
                    "--count-new-edges",
                    "1",
                    "--mab",
                    "ucb1tuned",
                    "--reward-policy",
                    "proportional",
                    "--results-dir",
                    str(case),
                ]
                command = [str(args.build.resolve() / "native_faco_diagnostic"), *parameters]
                (case / "task.json").write_text(
                    json.dumps(
                        {
                            "command": command,
                            "input_sha256": input_hash,
                            "purpose": "development correctness only",
                        },
                        indent=2,
                    )
                    + "\n"
                )
                started = time.perf_counter()
                with (case / "stdout.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=case,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                    )
                wall_seconds = time.perf_counter() - started
                report = json.loads((case / "diagnostic.json").read_text())
                if report["local_search_source_sha256"] != expected_source:
                    raise RuntimeError("声明的原生/适配版本与二进制编译来源不一致")
                verified = tour_cost(instance, report["tour"])
                if not math.isclose(verified, report["cost"], rel_tol=1e-12, abs_tol=1e-9):
                    raise RuntimeError(f"独立重算与原生增量成本不符: {case}")
                native_cli_cost = None
                if seed == 17:
                    # 同 seed 的原生入口和导出 wrapper 做黑盒核对；不读取最佳值表。
                    with (case / "native-cli.log").open("w") as log:
                        subprocess.run(
                            [str(args.build.resolve() / "native_faco"), *parameters],
                            cwd=case,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=120,
                        )
                    outputs = sorted(case.glob(f"{algorithm}-*.json"))
                    if not outputs:
                        raise RuntimeError("原生 CLI 没有结果文件")
                    cli = json.loads(outputs[-1].read_text())
                    native_cli_cost = cli["executions"][0]["final cost"]
                    if not math.isclose(native_cli_cost, verified, rel_tol=1e-12, abs_tol=1e-9):
                        raise RuntimeError("原生 CLI 与诊断 wrapper 在相同条件下不一致")
                summaries.append(
                    {
                        "variant": args.variant,
                        "local_search_source_sha256": report["local_search_source_sha256"],
                        "initial_route_count": report["initial_route_count"],
                        "n": n,
                        "algorithm": algorithm,
                        "seed": seed,
                        "iterations": args.iterations,
                        "ants": 32,
                        "threads": 1,
                        "cost": report["cost"],
                        "independent_cost": verified,
                        "native_cli_cost": native_cli_cost,
                        "absolute_cost_error": abs(verified - report["cost"]),
                        "reference_cost": label.cost,
                        "reference_gap": (verified - label.cost) / label.cost,
                        "label_certificate_status": label.certificate_status,
                        "solver_seconds": report["solver_seconds"],
                        "process_wall_seconds": wall_seconds,
                        "matrix_conversion_seconds": conversion_seconds,
                        "valid": True,
                        "input_sha256": input_hash,
                    }
                )
                print(f"{n} {algorithm} seed={seed}: cost={verified:.12g}, valid", flush=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "scope": "two development instances, iteration budget; no RQ conclusion",
                "runs": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
