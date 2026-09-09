"""原始 FACO 的连续距离适配：输入只有坐标和求解参数，评分位于调用方。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from gp_faco.experiment_v2 import NATIVE_BASELINE, write_once
from gp_faco.worker import PROJECT


def solve_native(problem, seed: int, iterations: int, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    write_once(
        directory / "parameters.json",
        {
            "method": NATIVE_BASELINE,
            "adaptation_revision": 2,
            "instance_id": problem.instance_id,
            "dimension": problem.dimension,
            "seed": seed,
            "iterations": iterations,
            "ants": "64*ceil(sqrt(n)/16)",
            "threads": 8,
            "mne": 8,
            "distance": "continuous_euclidean_fp64",
            "wall_clock_limit": None,
        },
    )
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    path = directory / "problem.tsp"
    # 十七位有效数字往返原来的 double 坐标；不做坐标量化。
    path.write_text(
        f"NAME: instance-{problem.numeric_id}\nTYPE: TSP\nDIMENSION: {problem.dimension}\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
        + "".join(f"{i + 1} {x:.17g} {y:.17g}\n" for i, (x, y) in enumerate(problem.coordinates))
        + "EOF\n"
    )
    with (directory / "native.log").open("w") as log:
        subprocess.run(
            [
                str(PROJECT / "build/v2-native-continuous/faco_2022"),
                "--problem",
                str(path),
                "--seed",
                str(seed),
                "--threads",
                "8",
                "--iterations",
                str(iterations),
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
    if result["method"] != NATIVE_BASELINE:
        raise ValueError("连续问题误用了 TSPLIB 整数距离的原生构建")
    return result
