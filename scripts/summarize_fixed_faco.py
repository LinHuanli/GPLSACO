#!/usr/bin/env python3
"""归集固定迭代FACO的CPU重放、GPU检查和开发池路线核验证据。"""

import hashlib
import json
import re
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts/gpu/fixed-faco"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_log(path: Path, pattern: str) -> dict:
    match = re.search(pattern, path.read_text())
    if match is None:
        raise RuntimeError(f"缺少完成证据: {path}")
    return {"result": match.group(), "sha256": digest(path)}


def main() -> None:
    differential = read(PROJECT / "build/cuda/fixed_faco_results.json")
    common_kernel = read(PROJECT / "build/cuda/cuda_faco_results.json")
    smoke = read(ARTIFACTS / "smoke/summary.json")
    manifest = read(ARTIFACTS / "smoke/manifest.json")
    if any(value["status"] != "passed" for value in (differential, common_kernel, smoke)):
        raise RuntimeError("存在未通过的检查")
    if differential["ant_batch_cases"] != 1280 or common_kernel["cases"] != 2160:
        raise RuntimeError("当前检查面板不完整")
    if len(smoke["runs"]) != 48 or not all(r["valid"] for r in smoke["runs"]):
        raise RuntimeError("开发池试跑不完整")
    if smoke["negative_reference_gaps"]:
        raise RuntimeError("存在负reference gap，需先核对距离/标签")
    logs = {
        "gpu_ctest": checked_log(ARTIFACTS / "ctest.log", r"100% tests passed, 0 tests failed"),
        "gpu_pytest": checked_log(ARTIFACTS / "pytest.log", r"39 passed in [\d.]+s"),
        "cpu_native_ctest": checked_log(
            PROJECT / "artifacts/environment/ctest-cpu-fixed.log",
            r"100% tests passed, 0 tests failed",
        ),
    }
    sanitized = {}
    for name in ("memcheck", "synccheck", "racecheck"):
        sanitized[name] = read(ARTIFACTS / f"{name}.json")
        if sanitized[name]["status"] != "passed" or sanitized[name]["ant_batch_cases"] != 240:
            raise RuntimeError(f"sanitizer面板不完整: {name}")
        pattern = (
            r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
            if name == "racecheck"
            else r"ERROR SUMMARY: 0 errors"
        )
        logs[name] = checked_log(ARTIFACTS / f"{name}.log", pattern)
    if (
        sanitized["memcheck"] != sanitized["synccheck"]
        or sanitized["memcheck"] != sanitized["racecheck"]
    ):
        raise RuntimeError("sanitizer结果不一致")
    by_scale = {}
    for dimension in (500, 1000):
        runs = [run for run in smoke["runs"] if run["dimension"] == dimension]
        by_scale[str(dimension)] = {
            "instances": len({r["record_id"] for r in runs}),
            "runs": len(runs),
            "maximum_absolute_cost_error": max(r["absolute_cost_error"] for r in runs),
            "improved_over_initial_runs": sum(r["cost"] < r["initial_cost"] for r in runs),
            "median_preparation_seconds": statistics.median(r["preparation_seconds"] for r in runs),
            "median_fixed_iteration_seconds": statistics.median(r["solve_seconds"] for r in runs),
            "allocated_device_bytes": sorted({r["allocated_device_bytes"] for r in runs}),
            "mean_reference_gap_percent_development_only": statistics.mean(
                r["reference_gap_percent"] for r in runs
            ),
        }
    sources = [
        "CMakeLists.txt",
        "cpp/include/gp_faco/fixed_faco_gpu.hpp",
        "cpp/src/bindings.cpp",
        "cuda/faco_device.cuh",
        "cuda/faco_choices.cuh",
        "cuda/fixed_faco.cu",
        "cuda/faco_operations.cu",
        "tests/cpp/test_fixed_faco.cpp",
        "tests/cpp/test_native_semantics.cpp",
        "tests/python/test_fixed_faco_gpu.py",
        "scripts/run_fixed_faco_checks.sh",
        "scripts/fixed_faco_smoke.py",
        "provenance/splits.v1.json",
    ]
    report = {
        "scope": (
            "fixed-iteration GPU FACO development pipeline; wall-clock GP Engine and E1-E4 pending"
        ),
        "differential": differential,
        "shared_kernel_regression": common_kernel,
        "sanitizers": sanitized,
        "logs": logs,
        "device_record": (ARTIFACTS / "device.csv").read_text().strip(),
        "development_pilot": {"manifest": manifest, "by_scale": by_scale, "runs": smoke["runs"]},
        "inputs": {path: digest(PROJECT / path) for path in sources},
    }
    (PROJECT / "docs/reports/fixed_faco_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print("已生成固定迭代FACO开发报告")


if __name__ == "__main__":
    main()
