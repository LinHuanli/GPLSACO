#!/usr/bin/env python3
"""汇总已完成的CUDA构造/LS操作检查，保留首轮竞争发现的证据。"""

import hashlib
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "artifacts/gpu/faco-operations"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_result(path: Path, count: int) -> dict:
    result = json.loads(path.read_text())
    if result["status"] != "passed" or result["cases"] != count:
        raise RuntimeError(f"检查范围或状态不符: {path}")
    return result


def checked_log(path: Path, pattern: str) -> dict:
    match = re.search(pattern, path.read_text())
    if match is None:
        raise RuntimeError(f"缺少通过证据: {path}")
    return {"sha256": fingerprint(path), "result": match.group()}


def main() -> None:
    full = read_result(PROJECT / "build/cuda/cuda_faco_results.json", 2160)
    sanitizers = {}
    logs = {
        "ctest": checked_log(RESULTS / "ctest.log", r"100% tests passed, 0 tests failed"),
        "pytest": checked_log(RESULTS / "pytest.log", r"33 passed in [\d.]+s"),
    }
    for name in ("memcheck", "synccheck", "racecheck"):
        sanitizers[name] = read_result(RESULTS / f"{name}.json", 540)
        pattern = (
            r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
            if name == "racecheck"
            else r"ERROR SUMMARY: 0 errors"
        )
        logs[name] = checked_log(RESULTS / f"{name}.log", pattern)
    if not sanitizers["memcheck"] == sanitizers["synccheck"] == sanitizers["racecheck"]:
        raise RuntimeError("三种sanitizer的相同面板结果不一致")
    initial = PROJECT / "artifacts/gpu/faco-operations-initial-race"
    initial_evidence = None
    if initial.exists():
        initial_evidence = {
            "log": checked_log(initial / "racecheck.log", r"RACECHECK SUMMARY: .*"),
            "source_sha256": fingerprint(initial / "faco_operations.cu"),
            "fix": "block barrier after construction loop condition and before steps increment",
        }
    report = {
        "scope": "CPU/CUDA operation equivalence; full GPU FACO, Engine and E1-E4 pending",
        "device_record": (RESULTS / "device.csv").read_text().strip(),
        "compile": {"cuda_architecture": 86, "fma_contraction": False, "line_info": True},
        "full_differential": full,
        "sanitizers": sanitizers,
        "logs": logs,
        "initial_race_and_fix": initial_evidence,
        "limitations": [
            "explicit symmetric distance matrix only for diagnostic n<=1024",
            "given selection permutations; device roulette and pheromone pending",
            "allocation and synchronization on every diagnostic call; not persistent Engine",
            "no wall-clock deadline, graph legality, restart transaction or GP training tested",
            "synthetic fixtures only; no dataset performance unblinding",
        ],
        "inputs": {
            path: fingerprint(PROJECT / path)
            for path in (
                "CMakeLists.txt",
                "cpp/include/gp_faco/faco_cuda_diagnostic.hpp",
                "cpp/src/faco_cpu.cpp",
                "cuda/faco_operations.cu",
                "tests/cpp/test_cuda_faco.cpp",
                "scripts/run_cuda_faco_checks.sh",
                "docs/reports/data_cpu_results.json",
            )
        },
    }
    (PROJECT / "docs/reports/cuda_faco_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print("已生成CUDA FACO操作检查报告")


if __name__ == "__main__":
    main()
