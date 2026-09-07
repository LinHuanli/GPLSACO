#!/usr/bin/env python3
"""从实际产物生成可提交的启动报告，剔除完整tour与其他用户资源信息。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def read_json(name: str) -> dict:
    return json.loads((PROJECT / name).read_text())


def main() -> None:
    inventory = read_json("provenance/data_inventory.json")
    native = read_json("artifacts/native/edge-guard-smoke/summary.json")
    probe = read_json("artifacts/native/initialization-probe/summary.json")
    scoring = read_json("artifacts/gpu/program-scores.json")
    checks = {}
    for name, path, marker in (
        ("cpu_pytest", "artifacts/environment/pytest-cpu.log", "27 passed"),
        ("cuda_pytest", "artifacts/gpu/pytest.log", "29 passed"),
        ("cuda_memcheck", "artifacts/gpu/memcheck.log", "ERROR SUMMARY: 0 errors"),
        ("cuda_synccheck", "artifacts/gpu/synccheck.log", "ERROR SUMMARY: 0 errors"),
    ):
        text = (PROJECT / path).read_text()
        if marker not in text:
            raise RuntimeError(f"缺少当前验收证据: {name}")
        checks[name] = {"result": marker, "log_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if len(native["runs"]) != 12 or not all(run["valid"] for run in native["runs"]):
        raise RuntimeError("原生适配诊断不完整")
    if not probe["confirmed_same_edge_false_improvement"]:
        raise RuntimeError("未确认初始化问题，不能沿用历史说明")
    environment = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "compiler": subprocess.check_output(
            ["/usr/bin/g++-15", "--version"], text=True
        ).splitlines()[0],
        "cuda_compiler": subprocess.check_output(
            ["/opt/cuda/bin/nvcc", "--version"], text=True
        ).splitlines()[-2:],
        "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
        "gpu": scoring["device"],
        "gpu_driver_record": (PROJECT / "artifacts/gpu/device.csv").read_text().strip(),
        "cmake_build_type": "Release",
        "cxx_standard": 17,
        "cuda_standard": 17,
        "cuda_architecture": "86",
        "gp_fma_contraction": False,
        "native_flags_additions": ["-mavx2", "support objects: -include cstdint"],
        "python_lock_sha256": hashlib.sha256(
            (PROJECT / "provenance/python.lock.txt").read_bytes()
        ).hexdigest(),
    }
    (PROJECT / "provenance/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    summary = {
        "scope": "research bootstrap and correctness diagnostics; E1-E4 not executed",
        "data": {
            "files": len(inventory["files"]),
            "total_bytes": inventory["total_bytes"],
            "raw_records_including_duplicates": inventory["raw_instances_including_duplicates"],
            "validated_records": sum(f["validated_records"] for f in inventory["files"]),
            "duplicate_groups": inventory["duplicate_groups"],
        },
        "checks": checks,
        "program_scoring": scoring,
        "native_initialization_probe": probe,
        "native_edge_guard_runs": native["runs"],
        "native_max_absolute_cost_error": max(r["absolute_cost_error"] for r in native["runs"]),
        "native_cli_equivalence_checks": sum(
            r["native_cli_cost"] is not None for r in native["runs"]
        ),
        "input_manifests": {
            name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
            for name in (
                "provenance/data_inventory.json",
                "provenance/sources.lock.json",
                "provenance/native_edge_guard.json",
                "configs/program_spec_v1.json",
            )
        },
    }
    (PROJECT / "docs/reports/bootstrap_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print("已生成环境记录与启动结果摘要")


if __name__ == "__main__":
    main()
