#!/usr/bin/env python3
# ruff: noqa: E402
"""在六组完整小流程中暂停/恢复，对比未中断运行的训练、选模和测试结果。"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.remote import environment
from gp_faco.representation_campaign import RUN_IDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    root = args.directory.resolve()
    if args.verify_only:
        verify(root, args.reference)
        return
    if (root / "campaign.json").exists():
        raise ValueError("恢复检查必须选择全新的工程目录，不能暂停既有实验")
    root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT / "scripts/run_representation_campaign.py"),
        "--directory",
        str(root),
        "--build",
        args.build,
        "--engineering",
        "--shard-size",
        "4",
        "--hosts",
        "cuda02",
        "cuda04",
        "cuda10",
        "cuda11",
        "piccolo",
    ]
    with (root / "resume_check.log").open("a") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT, env=environment(), stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            completed = [
                path for path in (root / "jobs").glob("population-*-training-*/result.json")
            ]
            if completed:
                atomic_json(
                    root / "pause.json",
                    {
                        "reason": "工程恢复检查",
                        "completed_jobs": [p.parent.name for p in completed],
                    },
                )
                break
            time.sleep(1)
        if process.wait() != 0 or read(root / "status.json")["stage"] != "paused":
            raise RuntimeError("工程流程未正常进入paused")
        receipt = read(root / "pause.json")
        atomic_json(root / "pause_observation.json", receipt)
        subprocess.run(
            [*command, "--resume"],
            cwd=PROJECT,
            env=environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    verify(root, args.reference)


def verify(root, reference):
    if read(root / "status.json")["stage"] != "complete":
        raise ValueError("恢复后的完整流程没有结束")
    for run in RUN_IDS:
        a = read(root / "runs" / run / "training_summary.json")
        b = read(reference / "runs" / run / "training_summary.json")
        assert [m["population_fitness"] for m in a["history"]] == [
            m["population_fitness"] for m in b["history"]
        ]
        assert [m["winner"] for m in a["history"]] == [m["winner"] for m in b["history"]]
        assert a["final_population"] == b["final_population"]
        assert (
            read(root / "runs" / run / "selected_controller.json")["controller"]
            == read(reference / "runs" / run / "selected_controller.json")["controller"]
        )

    def scores(directory):
        return sorted(
            (r["method"], r["dimension"], r["instance_id"], r["seed"], r["cost"], r["gap_percent"])
            for r in read(directory / "test_scores.json")["rows"]
            if r["method"] in RUN_IDS
        )

    assert scores(root) == scores(reference)
    assert all(
        read(path).get("attempt", 1) == 1
        for path in (root / "jobs").glob("*/assigned.json")
        if read(path.parent / "job.json")["resource"] == "gpu"
    )
    atomic_json(
        root / "resume_verification.json",
        {
            "status": "passed",
            "runs": 6,
            "same_population_fitness": True,
            "same_winners": True,
            "same_selected_controllers": True,
            "same_gp_test_scores": True,
            "all_gpu_job_attempts_one": True,
            "paused_after": read(root / "pause_observation.json"),
        },
    )
    print("六组暂停恢复与连续运行结果一致", flush=True)


if __name__ == "__main__":
    main()
