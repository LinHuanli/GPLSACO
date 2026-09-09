#!/usr/bin/env python3
# ruff: noqa: E402
"""汇总本轮独立工程测量，通过后发布50代入口所需的普通配置记录。"""

import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.representation_campaign import RUN_IDS


def main():
    root = PROJECT / "artifacts/v3"
    optimization = root / "representation-optimization"
    measured = {
        name: read(optimization / f"{name}.json") for name in ("reference", "w2", "w4", "w8")
    }
    if not all(value and value["status"] == "passed" for value in measured.values()):
        raise ValueError("同卡布局比较未全部通过")
    chosen = min(
        ("w2", "w4", "w8"), key=lambda k: sum(r["median_seconds"] for r in measured[k]["rows"])
    )
    if chosen != "w2":
        raise ValueError("最终构建需要与实际最快布局一致")
    rows = []
    for before, after in zip(measured["reference"]["rows"], measured[chosen]["rows"], strict=True):
        if before["case"] != after["case"] or not after["same_reference"]:
            raise ValueError("不同布局未对齐同一真实种群")
        rows.append(
            {
                "case": before["case"],
                "original_seconds": before["median_seconds"],
                "optimized_seconds": after["median_seconds"],
                "time_reduction_percent": 100
                * (1 - after["median_seconds"] / before["median_seconds"]),
                "total_fe": after["total_fe"],
                "identical_outputs_and_work": True,
            }
        )
    pool = read(root / "representation-pool-w2/performance.json")
    full = read(root / "representation-full-budget-final/performance.json")
    checks = read(optimization / "final_tests.json")
    resume = read(root / "representation-full-resume-01/resume_verification.json")
    reporting = read(optimization / "reporting_verification.json")
    if not all(
        value and value["status"] == "passed" for value in (pool, full, checks, resume, reporting)
    ):
        raise ValueError("分片、完整H1000、检查或恢复流程尚未全部通过")
    if (
        full["iterations"] != 1000
        or full["selected_shard_size"] != pool["selected_shard_size"]
        or checks["build"] != full["build"]
    ):
        raise ValueError("完整预算检查必须使用所选构建和分片")
    if not full["identical_across_shards"] or full["measurements"][0]["total_fe"] != 1572864000:
        raise ValueError("六个真实末代种群的完整FE或输出检查不符")
    if any(
        read(p)["execution"]["attempt"] != 1
        for p in (root / "representation-full-budget-final/jobs").glob("*/result.json")
    ):
        raise ValueError("完整预算检查中出现再次求解，需要单独核对FE")
    engineering = []
    for name in (
        "representation-engineering-01",
        "representation-full-resume-01",
        "representation-engineering-fast-io",
    ):
        status = read(root / name / "status.json")
        if status["stage"] != "complete" or status["failed"]:
            raise ValueError("完整工程流程尚未成功结束")
        engineering.append({"directory": name, "jobs": status["jobs"]})
    fast_root = root / "representation-engineering-fast-io"
    reference_root = root / "representation-engineering-01"
    for run in RUN_IDS:
        fast = read(fast_root / "runs" / run / "training_summary.json")
        reference = read(reference_root / "runs" / run / "training_summary.json")
        if (
            any(
                [m[field] for m in fast["history"]] != [m[field] for m in reference["history"]]
                for field in ("population_fitness", "winner")
            )
            or fast["final_population"] != reference["final_population"]
        ):
            raise ValueError("调度修改后训练结果不一致")
        if (
            read(fast_root / "runs" / run / "selected_controller.json")["controller"]
            != read(reference_root / "runs" / run / "selected_controller.json")["controller"]
        ):
            raise ValueError("调度修改后选模结果不一致")

    def gp_scores(directory):
        return sorted(
            (r["method"], r["dimension"], r["instance_id"], r["seed"], r["cost"])
            for r in read(directory / "test_scores.json")["rows"]
            if r["method"] in RUN_IDS
        )

    if gp_scores(fast_root) != gp_scores(reference_root) or any(
        read(path)["attempt"] != 1
        for path in (fast_root / "jobs").glob("*/assigned.json")
        if read(path.parent / "job.json")["resource"] == "gpu"
    ):
        raise ValueError("调度修改后测试结果不一致或GPU分片重复派单")
    log = (optimization / "final-tests.log").read_text()
    ctest = re.search(r"100% tests passed, 0 tests failed out of (\d+)", log)
    pytest = re.search(r"(\d+) passed, (\d+) skipped in", log)
    if not ctest or not pytest:
        raise ValueError("最终测试日志不完整")
    scheduler_test = re.search(
        r"(\d+) passed in", (optimization / "final-scheduler-tests.log").read_text()
    )
    if not scheduler_test:
        raise ValueError("最新调度回归检查尚未通过")
    performance = {
        "status": "passed",
        "build": full["build"],
        "warps_per_block": 2,
        "shard_size": pool["selected_shard_size"],
        "same_card_h100": rows,
        "same_card_device": measured["reference"]["device"],
        "layout_medians": {
            k: [r["median_seconds"] for r in v["rows"]] for k, v in measured.items()
        },
        "six_gpu_shard_h100": pool,
        "six_gpu_full_h1000": full,
        "full_h1000_reference": "gp-representation-pilot: all six generation-10 populations",
        "full_wall_includes_worker_startup": True,
        "full_wall_includes_coordinator_repair": False,
        "full_jobs_all_execution_attempt_one": True,
        "superseded_full_check": "artifacts/v3/representation-full-budget/verification_notes.json",
    }
    atomic_json(PROJECT / "results/v3/representation_performance.json", performance)
    atomic_json(
        PROJECT / "results/v3/representation_full_verification.json",
        {
            "status": "passed",
            "build": full["build"],
            "shard_size": pool["selected_shard_size"],
            "ctest_passed": int(ctest[1]),
            "python_passed": int(pytest[1]),
            "python_skipped": int(pytest[2]),
            "latest_scheduler_suite_passed": int(scheduler_test[1]),
            "engineering": engineering,
            "final_scheduler_same_results_and_no_gpu_retry": True,
            "resume": resume,
            "reporting": reporting,
            "performance": "results/v3/representation_performance.json",
            "ordinary_identifiers_only": True,
        },
    )
    print(f"Verified {full['build']} with {pool['selected_shard_size']} individuals per shard")


if __name__ == "__main__":
    main()
