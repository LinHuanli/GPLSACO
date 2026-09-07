#!/usr/bin/env python3
"""归集Hard逐边不变量与CUDA真实选点证据，保持E3工程/实验边界。"""

import hashlib
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts/gpu/hard-operations"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_log(path: Path, pattern: str) -> dict:
    match = re.search(pattern, path.read_text())
    if match is None:
        raise RuntimeError(f"缺少完成证据: {path}")
    return {"result": match.group(), "sha256": digest(path)}


def check_gpu(result: dict, cases: int, choices: int) -> None:
    if (
        result["status"] != "passed"
        or result["operation_cases"] != cases
        or result["stochastic_choices"] != choices
        or result["ls_accepted"] <= 0
        or result["ls_constraint_rejections"] <= 0
        or result["construction_exhausted"] <= 0
        or len(result["selection_stages"]) != 5
        or any(count <= 0 for count in result["selection_stages"])
        or sum(result["selection_stages"]) != choices
        or result["maximum_absolute_cost_error"] > 1e-8
    ):
        raise RuntimeError("Hard GPU面板/分支检查不完整")


def main() -> None:
    cpu = read(PROJECT / "build/cpu/hard_edge_results.json")
    if (
        cpu["status"] != "passed"
        or cpu["relocations"] != 34404
        or cpu["two_opt_moves"] != 68808
        or cpu["predicate_graph_checks"] != 370416
        or cpu != read(PROJECT / "build/cpu-sanitize/hard_edge_results.json")
    ):
        raise RuntimeError("CPU完整边集oracle或sanitizer结果不符")
    gpu = read(PROJECT / "build/cuda/hard_cuda_results.json")
    check_gpu(gpu, 576, 6144)
    logs = {
        "cpu_ctest": checked_log(
            PROJECT / "artifacts/environment/ctest-hard-cpu.log",
            r"100% tests passed, 0 tests failed out of 3",
        ),
        "cpu_asan_ubsan": checked_log(
            PROJECT / "artifacts/environment/ctest-hard-cpu-sanitize.log",
            r"100% tests passed, 0 tests failed out of 3",
        ),
        "gpu_ctest": checked_log(
            ARTIFACTS / "ctest.log", r"100% tests passed, 0 tests failed out of 6"
        ),
        "gpu_pytest": checked_log(ARTIFACTS / "pytest.log", r"49 passed in [\d.]+s"),
    }
    sanitized = {}
    for name in ("memcheck", "synccheck", "racecheck"):
        sanitized[name] = read(ARTIFACTS / f"{name}.json")
        check_gpu(sanitized[name], 288, 1536)
        pattern = (
            r"RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)"
            if name == "racecheck"
            else r"ERROR SUMMARY: 0 errors"
        )
        logs[name] = checked_log(ARTIFACTS / f"{name}.log", pattern)
    if not sanitized["memcheck"] == sanitized["synccheck"] == sanitized["racecheck"]:
        raise RuntimeError("三个sanitizer的确定性操作结果不一致")
    sources = [
        "CMakeLists.txt",
        "cpp/include/gp_faco/edge_constraints.hpp",
        "cpp/include/gp_faco/sparse_graph.hpp",
        "cpp/include/gp_faco/hard_diagnostic.hpp",
        "cpp/include/gp_faco/faco_cpu.hpp",
        "cpp/include/gp_faco/faco_cuda_diagnostic.hpp",
        "cpp/src/sparse_graph.cpp",
        "cpp/src/faco_cpu.cpp",
        "cuda/faco_device.cuh",
        "cuda/faco_choices.cuh",
        "cuda/faco_operations.cu",
        "cuda/hard_diagnostic.cu",
        "tests/cpp/test_hard_edges.cpp",
        "tests/cpp/test_hard_cuda.cpp",
        "scripts/run_hard_checks.sh",
        "scripts/summarize_hard_operations.py",
    ]
    report = {
        "scope": "Hard operation correctness; Engine integration, Escape, exporters and E3 pending",
        "complete_edge_set_oracle": cpu,
        "gpu_differential": gpu,
        "selection_stage_order": [
            "primary",
            "zero_weight_primary",
            "backup",
            "global",
            "exhausted",
        ],
        "sanitizers": sanitized,
        "logs": logs,
        "device_record": (ARTIFACTS / "device.csv").read_text().strip(),
        "shared_kernel_regression": {
            "explicit_operations": read(PROJECT / "build/cuda/cuda_faco_results.json"),
            "fixed_iterations": read(PROJECT / "build/cuda/fixed_faco_results.json"),
            "batch_deadline": read(PROJECT / "build/cuda/batch_engine_results.json"),
        },
        "inputs": {path: digest(PROJECT / path) for path in sources},
    }
    if any(value["status"] != "passed" for value in report["shared_kernel_regression"].values()):
        raise RuntimeError("共同内核回归未通过")
    (PROJECT / "docs/reports/hard_operations_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print("已归集Hard CPU/CUDA操作证据；E3完整求解仍待实现")


if __name__ == "__main__":
    main()
