#!/usr/bin/env python3
"""v2 全量训练入口：冻结后端/预算和两类 baseline 后，从新种群训练与完整验证。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import EvolutionSettings  # noqa: E402
from gp_faco.experiment_v2 import (  # noqa: E402
    EVOLUTION_SEEDS,
    GPU_BASELINE,
    NATIVE_BASELINE,
    BaselineCache,
)
from gp_faco.factorial_policy import FactorialPolicy  # noqa: E402
from gp_faco.factorial_training import FactorialTrainingRun  # noqa: E402
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, faco_ants  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT / "artifacts/v2/protocol")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=EVOLUTION_SEEDS, required=True)
    parser.add_argument("--condition", choices=("Full", "NoFeedback", "M10", "M01"), default="Full")
    parser.add_argument("--factorial-policy", type=Path)
    parser.add_argument(
        "--constraint-mode", choices=("unrestricted", "hard", "escape"), default="unrestricted"
    )
    parser.add_argument("--graph-prior", choices=("ALPHA", "POPMUSIC"))
    parser.add_argument("--graph-catalog", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    parser.add_argument("--through", choices=("training", "validation"), default="validation")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT) or not args.output.resolve().is_relative_to(PROJECT):
        parser.error("配置与产物必须在 GPLSACO 内")
    frozen = json.loads((directory / "frozen.json").read_text())
    declaration = json.loads((directory / "panels.json").read_text())
    readiness = json.loads((directory / "baselines_ready.json").read_text())
    if (
        frozen["experiment_version"] != 2
        or readiness["status"] != "complete"
        or readiness["training_iterations"] != frozen["training_iterations"]
        or readiness["manifest"]["numeric_backend"] != frozen["numeric_backend"]
    ):
        parser.error("v2 预算和两类配对 baseline 尚未完成")
    backend = frozen["numeric_backend"]
    build = {"exact": "v2-exact", "fp32": "v2-fp32", "fp32_fast": "v2-fp32-fast"}[backend]
    population_size = 128 if args.constraint_mode == "unrestricted" else 1
    if population_size > 1:
        build = "v2-population-" + backend.replace("_", "-")
    settings = TrainingSettings(
        evolution=EvolutionSettings(feature_spec_id=2, no_feedback=args.condition == "NoFeedback"),
        evolution_seed=args.seed,
        budgets=tuple((n, faco_ants(n) * frozen["training_iterations"]) for n in (500, 1000)),
        validation_budgets=tuple((n, faco_ants(n) * 5000) for n in (500, 1000)),
        monitoring_every=5,
        preparation_mode="cached",
        budget_kind="search_tour_evaluations",
        scope="formal_v2",
        population_batch_size=population_size,
    )
    model, driver = [
        v.strip()
        for v in subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.gpu_uuid}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(",")
    ]
    protocol = WorkerProtocol(
        args.gpu_uuid,
        model,
        driver,
        build,
        numeric_backend=backend,
        extension_directory=f"build/{build}",
        settings=SolverSettings(),
        maximum_registered_per_dimension=1024,
        constraint_mode=args.constraint_mode,
        graph_prior_kind=args.graph_prior,
        graph_catalog_path=str(args.graph_catalog.resolve().relative_to(PROJECT))
        if args.graph_catalog
        else None,
    )
    cache = BaselineCache(directory / "baselines.sqlite", readiness["manifest"], readonly=True)
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as source:
        training = {n: source.record_ids("train", n) for n in (500, 1000)}
        validation = {p["dimension"]: p["ids"] for p in declaration["validation"]}
        monitoring = [
            {**p, "evaluation_limit": faco_ants(p["dimension"]) * 5000}
            for p in declaration["monitor"]
        ]
        # 一次性确认 baseline 覆盖实际成员；每个 GP 个体仍独立完成全部 FE。
        for panels, iterations in [
            (g["panels"], frozen["training_iterations"]) for g in declaration["training"]
        ] + [(declaration["validation"], 5000), (declaration["monitor"], 5000)]:
            for panel in panels:
                for method in (NATIVE_BASELINE, GPU_BASELINE):
                    cache.panel(method, panel, iterations)
        data = TrainingData(
            source,
            training,
            validation,
            {
                "dataset": "main-index-v1",
                "experiment_version": 2,
                "protocol_directory": str(directory.relative_to(PROJECT)),
                "condition": args.condition,
                "frozen": frozen,
            },
            shared_panels=declaration["training"],
            monitoring_panels=monitoring,
            baseline_cache=cache,
        )
        cls, kwargs = TrainingRun, {}
        if args.condition in ("M10", "M01"):
            if not args.factorial_policy:
                parser.error("E2 重训需要已经选定的 Static 析因参数文件")
            policy = FactorialPolicy.from_dict(json.loads(args.factorial_policy.read_text()))
            if policy.variant != args.condition:
                parser.error("析因参数与 condition 不符")
            cls, kwargs = FactorialTrainingRun, {"factorial_policy": policy}
        run = cls(
            args.output,
            settings,
            protocol,
            data,
            resume=args.resume,
            event=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True),
            **kwargs,
        )
        result = run.run(stop_after_tasks=args.stop_after_tasks, through=args.through)
    cache.close()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
