#!/usr/bin/env python3
"""归集真实GP控制、档案、重启与开发计时证据，不推断训练或E1–E4结论。"""

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts/gpu/control-engine"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_log(path: Path, pattern: str) -> dict:
    match = re.search(pattern, path.read_text())
    if match is None:
        raise RuntimeError(f"缺少完成证据: {path}")
    return {"result": match.group(), "sha256": digest(path)}


def check_diagnostic(value: dict, small: bool) -> None:
    batches = 108 if small else 252
    ants = batches * (4 if small else 8)
    expected = {
        "status": "passed",
        "panel_configurations": 9,
        "colony_batches": batches,
        "ants": ants,
        "features": batches * 384,
        "actual_region_starts": ants,
        "full_identity_comparisons": ants,
        "forced_collision_colony_batches": 12 if small else 28,
        "real_late_improvement_discarded": True,
    }
    if any(value.get(key) != result for key, result in expected.items()):
        raise RuntimeError("控制诊断的面板/事务/碰撞/迟到证据不完整")
    if (
        value["restart_transactions"] <= 0
        or value["epoch_sources"] <= 0
        or value["iteration_sources"] <= 0
        or value["maximum_feature_error"] > 1e-6
        or value["maximum_cost_error"] > 1e-8
    ):
        raise RuntimeError("控制诊断分支覆盖或误差不符")


def main() -> None:
    cpu = read(PROJECT / "build/cpu/control_cpu_results.json")
    if (
        cpu != read(PROJECT / "build/cpu-sanitize/control_cpu_results.json")
        or cpu["status"] != "passed"
        or cpu["remaining_rank_checks"] != 2750840
        or cpu["region_checks"] != 2240
        or cpu["identity_checks"] != 34
    ):
        raise RuntimeError("CPU独立枚举/手算或sanitizer证据不一致")
    diagnostic = read(PROJECT / "build/cuda/control_engine_results.json")
    check_diagnostic(diagnostic, False)
    logs = {
        "cpu_ctest": checked_log(
            PROJECT / "artifacts/environment/ctest-control-cpu.log",
            r"100% tests passed, 0 tests failed out of 4",
        ),
        "cpu_asan_ubsan": checked_log(
            PROJECT / "artifacts/environment/ctest-control-cpu-sanitize.log",
            r"100% tests passed, 0 tests failed out of 4",
        ),
        "gpu_ctest": checked_log(
            ARTIFACTS / "ctest.log", r"100% tests passed, 0 tests failed out of 8"
        ),
        "gpu_pytest": checked_log(ARTIFACTS / "pytest.log", r"62 passed in [\d.]+s"),
    }
    sanitized = {}
    for name in ("memcheck", "synccheck", "racecheck"):
        sanitized[name] = read(ARTIFACTS / f"{name}.json")
        check_diagnostic(sanitized[name], True)
        pattern = (
            r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
            if name == "racecheck"
            else r"ERROR SUMMARY: 0 errors"
        )
        logs[name] = checked_log(ARTIFACTS / f"{name}.log", pattern)

    smoke = read(ARTIFACTS / "smoke/summary.json")
    manifest = read(ARTIFACTS / "smoke/manifest.json")
    if smoke["status"] != "passed" or len(smoke["runs"]) != 24:
        raise RuntimeError("开发计时面板不完整")
    if manifest["split_sha256"] != digest(PROJECT / "provenance/splits.v1.json"):
        raise RuntimeError("split身份变化")
    panels, signatures = [], set()
    for run in smoke["runs"]:
        n, mode, budget = run["dimension"], run["preparation_mode"], run["budget_seconds"]
        signature = (n, mode, budget, run["program_name"])
        if signature in signatures:
            raise RuntimeError("重复计时面板")
        signatures.add(signature)
        items, states = run["checked_items"], run["control_states"]
        expected = {(record, seed) for record in manifest["records"][str(n)] for seed in (17, 29)}
        if (
            len(items) != 32
            or len(states) != 32
            or {(r["record_id"], r["seed"]) for r in items} != expected
        ):
            raise RuntimeError("面板遗漏或替换了实例/seed")
        present = [item for item in items if item["has_incumbent"]]
        if any(
            not math.isfinite(item["cost"])
            or item["absolute_cost_error"] > 1e-8
            or not 0 <= item["completed_seconds"] <= budget
            for item in present
        ):
            raise RuntimeError("存在成本/时间核验失败")
        if not math.isclose(
            run["elapsed_seconds"], run["actual_seconds"] + run["charged_seconds"], abs_tol=1e-12
        ):
            raise RuntimeError("实际时间与扣费账目不符")
        completed = run["completed_batches"]
        if (
            completed + run["discarded_batches"] != run["launched_batches"]
            or run["discarded_batches"] > 1
            or (mode == "end_to_end" and run["charged_seconds"] != 0)
        ):
            raise RuntimeError("批次或模式账目不符")
        for state in states:
            if (
                not 0 <= state["restarts"] <= max(0, completed - 1)
                or not 0 <= state["epoch_batches"] <= completed
                or not 0 <= state["stagnant_batches"] <= completed
                or any(not 0 <= state[name] <= 1 for name in ("return_rate", "ls_work"))
                or (run["preparation_completed"] and not 1 <= state["archive_size"] <= 4)
                or (not run["preparation_completed"] and state["archive_size"] != 0)
                or (run["program_name"] == "keep_mne16" and state["restarts"] != 0)
            ):
                raise RuntimeError("对外控制状态违反完成批次计数/动作约束")
        total_fee = sum(sum(fee.values()) for fee in manifest["registration_fees"][str(n)].values())
        if mode == "cached_charged" and run["preparation_completed"]:
            if not math.isclose(run["charged_seconds"], total_fee, abs_tol=1e-12):
                raise RuntimeError("完成准备的缓存费用不符")
        panel = {k: v for k, v in run.items() if k not in ("checked_items", "control_states")}
        panel.update(
            {
                "tasks": len(items),
                "feasible_on_time": len(present),
                "missing_incumbents": len(items) - len(present),
                "maximum_absolute_cost_error": max(
                    (r["absolute_cost_error"] for r in present), default=0
                ),
                "mean_reference_gap_percent_untrained_development_only": (
                    statistics.mean(r["reference_gap_percent"] for r in present)
                    if present
                    else None
                ),
                "restarts_across_colonies": sum(s["restarts"] for s in states),
                "mean_archive_size": statistics.mean(s["archive_size"] for s in states),
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
        "cpp/include/gp_faco/control_ops.hpp",
        "cpp/include/gp_faco/control_cpu.hpp",
        "cpp/include/gp_faco/control_trace.hpp",
        "cpp/include/gp_faco/batch_engine.hpp",
        "cpp/include/gp_faco/prepared_problem.hpp",
        "cpp/include/gp_faco/faco_cuda_diagnostic.hpp",
        "cpp/src/control_cpu.cpp",
        "cpp/src/prepared_problem.cpp",
        "cpp/src/bindings.cpp",
        "cuda/control_state.cuh",
        "cuda/batch_engine.cu",
        "cuda/colony_state.cuh",
        "cuda/faco_device.cuh",
        "cuda/faco_choices.cuh",
        "cuda/gp_score.cu",
        "cuda/gp_score.cuh",
        "tests/cpp/test_control_cpu.cpp",
        "tests/cpp/test_control_engine.cpp",
        "tests/python/test_control_engine.py",
        "scripts/run_control_checks.sh",
        "scripts/control_engine_smoke.py",
        "scripts/summarize_control_engine.py",
        "docs/planning/09_control_contract.md",
        "provenance/splits.v1.json",
    ]
    report = {
        "scope": "main GP control engineering validation; worker, learning and E1-E4 pending",
        "cpu_reference": cpu,
        "gpu_differential": diagnostic,
        "sanitizers": sanitized,
        "logs": logs,
        "device_record": (ARTIFACTS / "device.csv").read_text().strip(),
        "development_manifest_sha256": digest(ARTIFACTS / "smoke/manifest.json"),
        "development_programs": manifest["programs"],
        "development_panels": panels,
        "inputs": {path: digest(PROJECT / path) for path in sources},
    }
    (PROJECT / "docs/reports/control_engine_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"已归集控制/事务证据及{len(panels)}个开发计时面板；正式训练仍未执行")


if __name__ == "__main__":
    main()
