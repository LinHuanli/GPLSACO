#!/usr/bin/env python3
"""按评价次数推进 v2：数值质量 → FE 曲线 → 配对 FACO → 新 GP 训练。"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
ACTIVE_DIRECTORY = None

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.experiment_v2 import (  # noqa: E402
    EVOLUTION_SEEDS,
    paired_quality,
    predeclare,
    write_once,
)


def execute(script, arguments, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.run(
            [str(PROJECT / ".venv/bin/python"), str(PROJECT / "scripts" / script), *arguments],
            cwd=PROJECT,
            env={**os.environ, "TMPDIR": str(PROJECT / ".tmp")},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def speed_comparison(exact, candidate):
    left = {(r["dimension"], r["controller"]): r for r in exact["rows"]}
    right = {(r["dimension"], r["controller"]): r for r in candidate["rows"]}
    if (
        len(left) != 4
        or left.keys() != right.keys()
        or any(len(r["seconds"]) != 5 for r in right.values())
    ):
        raise ValueError("需要两个规模、两种控制器的一次预热和五次完整测速")
    rows = [
        {
            "dimension": key[0],
            "controller": key[1],
            "speedup": left[key]["median_seconds"] / right[key]["median_seconds"],
        }
        for key in left
    ]
    return {
        "passed": all(r["speedup"] >= 1.1 for r in rows),
        "rows": rows,
        "total_median_seconds": sum(r["median_seconds"] for r in candidate["rows"]),
    }


def main():
    global ACTIVE_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--directory", type=Path, default=PROJECT / "artifacts/v2/protocol")
    parser.add_argument(
        "--through", choices=("quality", "curves", "baselines", "training"), default="baselines"
    )
    parser.add_argument(
        "--wait-for-previous",
        action="store_true",
        help="等待同目录当前流程完成后接续；等待期间不占 GPU",
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("运行目录必须在项目内")
    directory.mkdir(parents=True, exist_ok=True)
    lease = (directory / "pipeline.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not args.wait_for_previous:
            raise
        print(
            json.dumps({"stage": "waiting_for_previous_pipeline", "through": args.through}),
            flush=True,
        )
        fcntl.flock(lease, fcntl.LOCK_EX)
        previous = directory / "pipeline_status.json"
        if previous.exists() and json.loads(previous.read_text()).get("stage") == "failed":
            raise RuntimeError("前置流程失败；保留其错误记录，后续训练不启动") from None
    ACTIVE_DIRECTORY = directory
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as source:
        predeclare(source, directory)
    write_once(
        directory / "selection_rules.json",
        {
            "version": 2,
            "quality": (
                "one_sided_paired_student_t_95_percent_upper_"
                "at_most_0.01pp_per_scale_and_controller"
            ),
            "speed": "all_four_complete_solve_medians_at_least_1.1x",
            "candidate_order": (
                "increasing_sum_of_four_medians; test_next_only_if_faster_candidate_fails"
            ),
            "horizon": (
                "shortest_checkpoint_with_95_percent_improvement_and_spearman_0.90_on_both_scales"
            ),
            "training": "50_evaluated_generations_128_population_no_fitness_cache",
            "wall_clock_limit": None,
            "test_labels_before_method_freeze": False,
        },
    )

    def status(stage, **values):
        atomic_json(
            directory / "pipeline_status.json",
            {"stage": stage, "gpu_uuid": args.gpu, "pid": os.getpid(), **values},
        )
        print(json.dumps({"stage": stage, **values}), flush=True)
        execute("summarize_v2.py", ["--directory", str(directory)], directory / "logs/report.log")

    numeric_path = directory / "numeric_selection.json"
    if not numeric_path.exists():
        status("benchmark")
        reports = {}
        for backend, suffix in (("exact", "exact"), ("fp32", "fp32"), ("fp32_fast", "fp32-fast")):
            path = PROJECT / f"artifacts/v2/gpu-checks/final-v2-{suffix}.json"
            if not path.exists() or len(json.loads(path.read_text())["rows"]) != 4:
                execute(
                    "benchmark_v2.py",
                    ["--gpu", args.gpu, "--build", f"build/v2-{suffix}", "--output", str(path)],
                    directory / "logs" / f"benchmark-{suffix}.log",
                )
            reports[backend] = json.loads(path.read_text())
        candidates = {
            name: speed_comparison(reports["exact"], reports[name])
            for name in ("fp32", "fp32_fast")
        }
        ordered = sorted(
            (name for name in candidates if candidates[name]["passed"]),
            key=lambda name: candidates[name]["total_median_seconds"],
        )
        selected = "exact"
        if ordered:
            status("quality", backend="exact")
            execute(
                "calibrate_v2.py",
                ["quality", "--backend", "exact", "--gpu", args.gpu, "--directory", str(directory)],
                directory / "logs/quality-exact.log",
            )
            exact = json.loads((directory / "quality/exact/summary.json").read_text())
            if any(row["status"] != "completed" for row in exact["rows"]):
                raise RuntimeError("精确后端质量面板存在失败，保留全部案例，修复前不能启动训练")
            for name in ordered:
                status("quality", backend=name)
                execute(
                    "calibrate_v2.py",
                    [
                        "quality",
                        "--backend",
                        name,
                        "--gpu",
                        args.gpu,
                        "--directory",
                        str(directory),
                    ],
                    directory / "logs" / f"quality-{name}.log",
                )
                candidate = json.loads((directory / "quality" / name / "summary.json").read_text())
                candidates[name]["quality"] = paired_quality(exact["rows"], candidate["rows"])
                if candidates[name]["quality"]["passed"]:
                    selected = name
                    break
        atomic_json(
            numeric_path,
            {
                "numeric_backend": selected,
                "candidates": candidates,
                "selection": (
                    "fastest_speed_qualified_candidate_passing_complete_quality_panel_else_exact"
                ),
            },
        )
    numeric = json.loads(numeric_path.read_text())
    if args.through == "quality":
        status("quality_complete", **numeric)
        return
    frozen_path = directory / "frozen.json"
    if not frozen_path.exists():
        status("curves", backend=numeric["numeric_backend"])
        execute(
            "calibrate_v2.py",
            [
                "curves",
                "--backend",
                numeric["numeric_backend"],
                "--gpu",
                args.gpu,
                "--directory",
                str(directory),
            ],
            directory / "logs/curves.log",
        )
        curve = json.loads(
            (directory / "curves" / numeric["numeric_backend"] / "summary.json").read_text()
        )
        atomic_json(
            frozen_path,
            {
                "experiment_version": 2,
                "numeric_backend": numeric["numeric_backend"],
                "training_iterations": curve["selection"]["iterations"],
                "validation_iterations": 5000,
                "test_iterations": 5000,
                "ants": "64*ceil(sqrt(n)/16)",
                "numeric_selection": numeric,
                "horizon_selection": curve["selection"],
                "wall_clock_limit": None,
            },
        )
    if args.through == "curves":
        status("curves_complete")
        return
    if not (directory / "baselines_ready.json").exists():
        status("baselines")
        execute(
            "cache_baselines_v2.py",
            ["--gpu", args.gpu, "--directory", str(directory)],
            directory / "logs/baselines.log",
        )
    if args.through == "baselines":
        status("baselines_complete_training_ready")
        return
    tuning = PROJECT / "artifacts/v2/baseline-tuning"
    tuned = tuning / "summary.json"
    if not tuned.exists() or json.loads(tuned.read_text())["status"] != "complete":
        status("baseline_tuning")
        arguments = ["--directory", str(directory), "--gpu", args.gpu, "--output", str(tuning)]
        if (tuning / "checkpoint.json").exists():
            arguments.append("--resume")
        execute("tune_baselines_v2.py", arguments, directory / "logs/baseline-tuning.log")
    # 单卡队列不预占其他 GPU；所有 Full/NoFeedback 使用同一份已经准备好的面板和对照。
    for condition in ("Full", "NoFeedback"):
        for seed in EVOLUTION_SEEDS:
            output = PROJECT / "artifacts/v2/training" / f"e1-{condition}-seed{seed}"
            if (output / "summary.json").exists() and json.loads(
                (output / "summary.json").read_text()
            )["status"] == "complete":
                continue
            status("training", condition=condition, seed=seed)
            arguments = [
                "--directory",
                str(directory),
                "--gpu-uuid",
                args.gpu,
                "--output",
                str(output),
                "--condition",
                condition,
                "--seed",
                str(seed),
            ]
            if (output / "checkpoint.json").exists():
                arguments.append("--resume")
            execute("train_gp.py", arguments, directory / "logs" / f"e1-{condition}-{seed}.log")
    status("e1_gp_training_and_validation_complete_test_still_sealed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ACTIVE_DIRECTORY is not None:
            path = ACTIVE_DIRECTORY / "pipeline_status.json"
            previous = json.loads(path.read_text()) if path.exists() else {}
            atomic_json(
                path,
                {
                    **previous,
                    "stage": "failed",
                    "failed_stage": previous.get("stage"),
                    "error": str(error),
                },
            )
            # 失败时也更新读者看到的状态；报告工具失败不能掩盖原始异常。
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT / "scripts/summarize_v2.py"),
                    "--directory",
                    str(ACTIVE_DIRECTORY),
                ],
                check=False,
            )
        raise
