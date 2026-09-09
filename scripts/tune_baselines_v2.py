#!/usr/bin/env python3
"""完整 Static/Rule 配置族，开发集筛选后在完整验证集使用 5000 迭代选择。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.baseline_policy import policy_grid  # noqa: E402
from gp_faco.configuration_search import ConfigurationRun, SearchData, SearchSettings  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, faco_ants  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT / "artifacts/v2/protocol")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--constraint-mode", choices=("unrestricted", "hard", "escape"), default="unrestricted"
    )
    parser.add_argument("--graph-prior", choices=("ALPHA", "POPMUSIC"))
    parser.add_argument("--graph-catalog")
    args = parser.parse_args()
    frozen = json.loads((args.directory / "frozen.json").read_text())
    config = json.loads((PROJECT / "configs/controller_families_v2.json").read_text())
    policies = policy_grid(config["tuning_policy_grid"])
    purpose = "tuning"
    if args.constraint_mode != "unrestricted":
        policies = tuple(p for p in policies if p.kind == "static")
        purpose = "static_tuning"
    settings = SearchSettings(
        purpose=purpose,
        evaluation_limits=(faco_ants(500) * frozen["training_iterations"],),
        validation_evaluation_limit=faco_ants(500) * 5000,
    )
    backend = frozen["numeric_backend"]
    build = {"exact": "v2-exact", "fp32": "v2-fp32", "fp32_fast": "v2-fp32-fast"}[backend]
    model, driver = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.gpu}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    protocol = WorkerProtocol(
        args.gpu,
        model,
        driver,
        build,
        numeric_backend=backend,
        extension_directory=f"build/{build}",
        settings=SolverSettings(),
        constraint_mode=args.constraint_mode,
        graph_prior_kind=args.graph_prior,
        graph_catalog_path=args.graph_catalog,
    )
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as source:
        pools = {
            "search": {n: source.record_ids("development", n)[:64] for n in (500, 1000)},
            "validation": {n: source.record_ids("validation", n) for n in (500, 1000)},
        }
        data = SearchData(
            source,
            pools,
            {
                "experiment_version": 2,
                "frozen": frozen,
                "selection": "complete_predeclared_grid_development64_then_validation256",
            },
        )
        result = ConfigurationRun(
            args.output,
            settings,
            protocol,
            policies,
            data,
            resume=args.resume,
            event=lambda row: print(json.dumps(row), flush=True),
        ).run()
    if result["status"] != "complete":
        raise RuntimeError("基线配置选择未完成；保留失败结果")


if __name__ == "__main__":
    main()
