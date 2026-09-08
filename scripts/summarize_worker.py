#!/usr/bin/env python3
"""核查worker任务产物与回归日志，导出不含完整tour的精简证据。"""

import json
import re
from pathlib import Path

from gp_faco.worker import content_hash, file_hash

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts/gpu/worker-engine"


def checked_log(path: Path, pattern: str) -> dict:
    match = re.search(pattern, path.read_text())
    if not match:
        raise RuntimeError(f"缺少终态证据: {path}")
    return {"result": match.group(), "sha256": file_hash(path)}


def main() -> None:
    report = json.loads((ARTIFACTS / "smoke/summary.json").read_text())
    if report["status"] != "passed" or report["coordinator_loaded_native_extension"]:
        raise RuntimeError("协调进程/worker隔离不符")
    if len(report["records"]) != 10 or len({r["task_id"] for r in report["records"]}) != 10:
        raise RuntimeError("预定任务未完整保存")
    for name, expected in report["inputs"].items():
        if file_hash(PROJECT / name) != expected:
            raise RuntimeError(f"运行后输入改变: {name}")
    for name, runtime in report["runtimes"].items():
        protocol = report["protocols"][name]
        if (
            runtime["pid"] == report["coordinator_pid"]
            or runtime["start_method"] != "spawn"
            or runtime["protocol_sha256"] != content_hash(protocol)
            or runtime["binary_sha256"] != file_hash(PROJECT / "build/cuda/gp_faco_ext.so")
        ):
            raise RuntimeError("spawn或硬件/代码身份不符")
    expected_failures = {"state:invalid-problem"}
    failures = {r["occurrence_id"] for r in report["records"] if r["status"] == "failed"}
    if failures != expected_failures:
        raise RuntimeError("任务失败没有按预期保留")
    result_fingerprints = {}
    for row in report["records"]:
        path = ARTIFACTS / "smoke" / f"{row['task_id']}.json"
        stored = json.loads(path.read_text())
        if (
            content_hash(stored["task"]) != row["task_id"]
            or stored["result"]["status"] != row["status"]
        ):
            raise RuntimeError("单任务产物与归集身份不符")
        result_fingerprints[path.name] = file_hash(path)
    if {p["dimension"] for p in report["development_panels"]} != {500, 1000}:
        raise RuntimeError("两个开发规模未覆盖")
    if any(
        len(p["score"]["members"]) != 32 or p["score"]["error"] is not None
        for p in report["development_panels"]
    ):
        raise RuntimeError("64个预定开发成员未完整核验")
    report["logs"] = {
        "cpu_pytest": checked_log(
            PROJECT / "artifacts/environment/pytest-worker-cpu.log",
            r"42 passed, 29 skipped in [\d.]+s",
        ),
        "gpu_pytest": checked_log(ARTIFACTS / "pytest.log", r"73 passed in [\d.]+s"),
    }
    report["task_artifact_sha256"] = result_fingerprints
    report["inputs"].update(
        {
            path: file_hash(PROJECT / path)
            for path in ("scripts/run_worker_checks.sh", "scripts/summarize_worker.py")
        }
    )
    (PROJECT / "docs/reports/worker_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print("已核验10个worker任务与64个真实开发成员；DEAP演化及正式实验仍待执行")


if __name__ == "__main__":
    main()
