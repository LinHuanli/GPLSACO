#!/usr/bin/env python3
"""从已完成的数据和 CPU 语义检查生成精简证据，不改写历史启动报告。"""

import hashlib
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: str) -> dict:
    return json.loads((PROJECT / path).read_text())


def main() -> None:
    sources = read_json("provenance/sources.lock.json")["adaptive_faco"]
    source_root = PROJECT / sources["root"]
    for path, fingerprint in sources["files"].items():
        if digest(source_root / path) != fingerprint["sha256"]:
            raise RuntimeError(f"外部 FACO 来源变更: {path}")
    data = read_json("artifacts/data/main-index-v1/verification.json")
    cpu = read_json("build/cpu/native_semantics_results.json")
    sanitized = read_json("build/cpu-sanitize/native_semantics_results.json")
    if any(result["status"] != "passed" for result in (data, cpu, sanitized)):
        raise RuntimeError("存在未通过的检查")
    if cpu != sanitized:
        raise RuntimeError("普通与 sanitizer 构建的语义结果不同")
    logs = {}
    for path, pattern in (
        ("artifacts/environment/pytest-core-data.log", r"31 passed in [\d.]+s"),
        ("artifacts/environment/ctest-core-semantics.log", r"100% tests passed, 0 tests failed"),
        ("artifacts/environment/ctest-cpu-sanitize.log", r"100% tests passed, 0 tests failed"),
    ):
        text = (PROJECT / path).read_text()
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"缺少通过证据: {path}")
        logs[path] = {"sha256": digest(PROJECT / path), "result": match.group()}
    summary = {
        "scope": "main data split and CPU FACO operations; CUDA solver and E1-E4 pending",
        "data": data,
        "cpu_native_differential": cpu,
        "cpu_sanitizers": {
            "address": "passed",
            "undefined_behavior": "passed",
            "leak_detection": True,
            "same_results_as_release": True,
            "build_type": "RelWithDebInfo",
            "compiler": "/usr/bin/g++-15",
        },
        "logs": logs,
        "external_source_fingerprints_unchanged": len(sources["files"]),
        "inputs": {
            path: digest(PROJECT / path)
            for path in (
                "provenance/splits.v1.json",
                "configs/data_split_v1.json",
                "provenance/sources.lock.json",
                "cpp/include/gp_faco/faco_cpu.hpp",
                "cpp/src/faco_cpu.cpp",
                "tests/cpp/test_native_semantics.cpp",
                "python/gp_faco/dataset_index.py",
                "scripts/build_main_dataset.py",
                "scripts/verify_main_dataset.py",
            )
        },
    }
    (PROJECT / "docs/reports/data_cpu_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print("已生成数据与 CPU 操作语义报告")


if __name__ == "__main__":
    main()
