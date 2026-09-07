#!/usr/bin/env python3
"""归集多实例Engine、截止提交与开发池计时证据，不推断E1–E4结论。"""

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts/gpu/batch-engine"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_log(path: Path, pattern: str) -> dict:
    match = re.search(pattern, path.read_text())
    if match is None:
        raise RuntimeError(f"缺少完成证据: {path}")
    return {"result": match.group(), "sha256": digest(path)}


def check_diagnostic(value: dict) -> None:
    if (
        value["status"] != "passed"
        or value["panel_configurations"] != 4
        or value["single_colony_comparisons"] != 16
        or not value["real_late_improvement_discarded"]
        or value["late_batch_completion_seconds"] <= value["late_budget_seconds"]
    ):
        raise RuntimeError("并发/迟到诊断不完整")


def main() -> None:
    diagnostic = read(PROJECT / "build/cuda/batch_engine_results.json")
    check_diagnostic(diagnostic)
    logs = {
        "cpu_ctest": checked_log(
            PROJECT / "artifacts/environment/ctest-deadline-cpu.log",
            r"100% tests passed, 0 tests failed out of 2",
        ),
        "gpu_ctest": checked_log(
            ARTIFACTS / "ctest.log", r"100% tests passed, 0 tests failed out of 4"
        ),
        "gpu_pytest": checked_log(ARTIFACTS / "pytest.log", r"49 passed in [\d.]+s"),
    }
    sanitizers = {}
    for name in ("memcheck", "synccheck", "racecheck"):
        sanitizers[name] = read(ARTIFACTS / f"{name}.json")
        # 每次运行的真实时长不同；检查同一面板/不变量，不要求时间值相等。
        check_diagnostic(sanitizers[name])
        pattern = (
            r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
            if name == "racecheck"
            else r"ERROR SUMMARY: 0 errors"
        )
        logs[name] = checked_log(ARTIFACTS / f"{name}.log", pattern)

    smoke = read(ARTIFACTS / "smoke/summary.json")
    manifest = read(ARTIFACTS / "smoke/manifest.json")
    if smoke["status"] != "passed" or len(smoke["runs"]) != 16:
        raise RuntimeError("开发池面板不完整")
    if manifest["split_sha256"] != digest(PROJECT / "provenance/splits.v1.json"):
        raise RuntimeError("split身份变化")
    panels, signatures = [], set()
    for run in smoke["runs"]:
        n, mode, budget = run["dimension"], run["preparation_mode"], run["budget_seconds"]
        signature = (n, mode, budget)
        if signature in signatures:
            raise RuntimeError("重复面板")
        signatures.add(signature)
        items = run["checked_items"]
        expected = {(record, seed) for record in manifest["records"][str(n)] for seed in (17, 29)}
        if len(items) != 32 or {(r["record_id"], r["seed"]) for r in items} != expected:
            raise RuntimeError("面板遗漏或替换了实例/seed")
        present = [item for item in items if item["has_incumbent"]]
        if any(
            not math.isfinite(item["cost"])
            or item["absolute_cost_error"] > 1e-8
            or not 0 <= item["completed_seconds"] <= budget
            or item["reference_gap_percent"] < -1e-8
            for item in present
        ):
            raise RuntimeError("存在成本/时间/标签核验失败")
        if not math.isclose(
            run["elapsed_seconds"], run["actual_seconds"] + run["charged_seconds"], abs_tol=1e-12
        ):
            raise RuntimeError("实际时间与扣费账目不符")
        if (
            run["completed_batches"] + run["discarded_batches"] != run["launched_batches"]
            or run["discarded_batches"] > 1
            or (mode == "end_to_end" and run["charged_seconds"] != 0)
        ):
            raise RuntimeError("批次或模式账目不符")
        total_fee = sum(
            fee["cheap_seconds"] + fee["preparation_seconds"]
            for fee in manifest["registration_fees"][str(n)].values()
        )
        if mode == "cached_charged" and run["preparation_completed"]:
            if not math.isclose(run["charged_seconds"], total_fee, abs_tol=1e-12):
                raise RuntimeError("完成准备的缓存费用不符")
        panel = {key: value for key, value in run.items() if key != "checked_items"}
        panel.update(
            {
                "tasks": len(items),
                "feasible_on_time": len(present),
                "missing_incumbents": len(items) - len(present),
                "maximum_absolute_cost_error": max(
                    (r["absolute_cost_error"] for r in present), default=0
                ),
                "mean_reference_gap_percent_development_only": (
                    statistics.mean(r["reference_gap_percent"] for r in present)
                    if present
                    else None
                ),
                "actual_wall_overrun_seconds": max(0, run["actual_seconds"] - budget),
                "discarded_fraction": (
                    run["discarded_batches"] / run["launched_batches"]
                    if run["launched_batches"]
                    else 0
                ),
            }
        )
        panels.append(panel)

    sources = [
        "CMakeLists.txt",
        "cpp/include/gp_faco/batch_engine.hpp",
        "cpp/include/gp_faco/deadline.hpp",
        "cpp/include/gp_faco/prepared_problem.hpp",
        "cpp/src/prepared_problem.cpp",
        "cpp/src/bindings.cpp",
        "cuda/batch_engine.cu",
        "cuda/colony_state.cuh",
        "cuda/faco_device.cuh",
        "cuda/faco_choices.cuh",
        "cuda/fixed_faco.cu",
        "tests/cpp/test_deadline.cpp",
        "tests/cpp/test_batch_engine.cpp",
        "tests/python/test_batch_engine.py",
        "scripts/run_batch_engine_checks.sh",
        "scripts/batch_engine_smoke.py",
        "scripts/summarize_batch_engine.py",
        "provenance/splits.v1.json",
    ]
    report = {
        "scope": (
            "fixed-shape deadline Engine engineering pilot; GP/restart/Hard/Escape/E1-E4 pending"
        ),
        "differential_and_deadline": diagnostic,
        "sanitizers": sanitizers,
        "logs": logs,
        "device_record": (ARTIFACTS / "device.csv").read_text().strip(),
        "development_pilot": {"manifest": manifest, "panels": panels},
        "artifact_hashes": {
            path: digest(ARTIFACTS / path)
            for path in ("smoke/summary.json", "smoke/manifest.json", "smoke/device.csv")
        },
        "inputs": {path: digest(PROJECT / path) for path in sources},
    }
    (PROJECT / "docs/reports/batch_engine_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"已归集{len(panels)}个固定面板、{sum(p['tasks'] for p in panels)}个任务结果")


if __name__ == "__main__":
    main()
