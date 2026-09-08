#!/usr/bin/env python3
"""原UUID空闲后重启原776配置基线完整CLI；只接续明确的未提交资源暂停。"""

import argparse
import csv
import datetime
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402

ADMISSION_ERROR = "目标GPU出现外来计算进程，当前任务尚未提交，保留待办"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unsubmitted_resource_pause(state, directory):
    """实际返回、running尝试、无证据错误均不能走自动接续。"""
    pending = state.get("pending")
    if not isinstance(pending, dict) or pending.get("attempts") != []:
        return False
    if (directory / "tasks" / (pending["key"] + ".json")).exists():
        return False
    admission = state.get("admission", [])
    return bool(
        admission
        and admission[-1].get("task_key") == pending["key"]
        and type(admission[-1].get("foreign_processes")) is int
        and admission[-1]["foreign_processes"] > 0
        and state["phase"] in ("search", "validation")
    )


def is_idle(observation):
    return (
        type(observation.get("processes")) is int
        and observation["processes"] == 0
        and type(observation.get("memory_used_mib")) is int
        and observation["memory_used_mib"] <= 1024
        and type(observation.get("utilization_percent")) is int
        and observation["utilization_percent"] <= 5
    )


def observe(uuid):
    try:
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=name,driver_version,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        model, driver, memory, utilization = [
            v.strip() for v in next(csv.reader(rows.splitlines()))
        ]
        return {
            "processes": sum(bool(row) for row in csv.reader(apps.splitlines())),
            "gpu_model": model,
            "driver_version": driver,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
        }
    except Exception as error:
        return {"processes": None, "error": f"{type(error).__name__}: {error}"}


def run(request_path):
    request = json.loads(request_path.read_text())
    request_sha = file_hash(request_path)
    require(
        request["request_spec_id"] == 1
        and request["no_wall_clock_limit"]
        and request["host"] == socket.gethostname()
        and os.environ.get("CUDA_VISIBLE_DEVICES") == request["gpu_uuid"],
        "等待身份不符",
    )
    directory = (PROJECT / request["input"]).resolve()
    output = (PROJECT / request["output"]).resolve()
    require(
        directory.is_relative_to(PROJECT) and output.is_relative_to(PROJECT), "产物必须在GPLSACO内"
    )
    receipt_path = output / "waiting-receipt.json"
    require(not receipt_path.exists(), "等待回执已存在，不能重复编排")
    checkpoint = directory / "checkpoint.json"
    require(file_hash(checkpoint) == request["paused_checkpoint_sha256"], "已审计暂停点改变")
    require(
        file_hash(PROJECT / request["prefix_audit"]) == request["prefix_audit_sha256"],
        "审计证据改变",
    )
    state = load_checkpoint(checkpoint)
    require(unsubmitted_resource_pause(state, directory), "不是可接续的未提交资源暂停")
    manifest = load_checkpoint(directory / "manifest.json")
    p = manifest["worker_protocol"]
    require(
        state["run_id"] == request["run_id"]
        and p["gpu_uuid"] == request["gpu_uuid"]
        and p["execution_host"] == request["host"]
        and p["gpu_model"] == request["gpu_model"]
        and p["driver_version"] == request["driver_version"],
        "原运行硬件/身份改变",
    )
    require(
        manifest["data"]["identity"]["config_path"] == request["config"], "完整基线配置路径改变"
    )
    command = [
        sys.executable,
        str(PROJECT / "scripts/search_baselines.py"),
        "--config",
        str(PROJECT / request["config"]),
        "--gpu-uuid",
        request["gpu_uuid"],
        "--output",
        str(directory),
        "--resume",
    ]
    receipt = {
        "request_sha256": request_sha,
        "run_id": state["run_id"],
        "host": request["host"],
        "gpu_uuid": request["gpu_uuid"],
        "waiting_samples": 0,
        "cli_attempts": [],
        "native_submissions_while_waiting": 0,
        "status": "waiting_for_original_gpu",
    }
    checkpoint_sha = file_hash(checkpoint)
    try:
        while True:
            require(file_hash(request_path) == request_sha, "等待请求改变")
            for name, digest in request["sources"].items():
                require(file_hash(PROJECT / name) == digest, "冻结入口或编排实现改变")
            require(file_hash(checkpoint) == checkpoint_sha, "等待期间其他操作者推进了原运行")
            observation = observe(request["gpu_uuid"])
            if "gpu_model" in observation:
                require(
                    observation["gpu_model"] == request["gpu_model"]
                    and observation["driver_version"] == request["driver_version"],
                    "等待期间硬件改变",
                )
            receipt.update(
                status="waiting_for_original_gpu",
                observation=observation,
                observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            atomic_json(receipt_path, receipt)
            if not is_idle(observation):
                receipt["waiting_samples"] += 1
                time.sleep(5)
                continue
            previous = load_checkpoint(checkpoint)
            require(unsubmitted_resource_pause(previous, directory), "待提交状态改变")
            number = len(receipt["cli_attempts"]) + 1
            log, resources = (
                output / f"resume-{number}.log",
                output / f"resume-{number}-resources.json",
            )
            timing = [
                "/usr/bin/time",
                "-q",
                "-f",
                '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"exit_code":%x}',
                "-o",
                str(resources),
                *command,
            ]
            with log.open("x") as stream:
                child = subprocess.Popen(
                    timing, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT
                )
                attempt = {
                    "number": number,
                    "child_pid": child.pid,
                    "log": str(log.relative_to(PROJECT)),
                    "resources": str(resources.relative_to(PROJECT)),
                    "exit_code": None,
                }
                receipt["cli_attempts"].append(attempt)
                receipt.update(status="original_full_cli_resuming")
                atomic_json(receipt_path, receipt)
                code = child.wait()
            attempt["exit_code"] = code
            state = load_checkpoint(checkpoint)
            require(
                state["run_id"] == request["run_id"]
                and all(
                    state["completed"].get(key) == digest
                    for key, digest in previous["completed"].items()
                ),
                "接续覆盖了已有正式任务",
            )
            if (
                code == 1
                and log.read_text().rstrip().endswith(ADMISSION_ERROR)
                and unsubmitted_resource_pause(state, directory)
            ):
                checkpoint_sha = file_hash(checkpoint)
                continue
            receipt.update(status="resume_cli_exited; inspect terminal audit", exit_code=code)
            require(
                code != 0
                or (
                    state["phase"] == "complete"
                    and state["pending"] is None
                    and state["active_worker"] is None
                ),
                "零退出码没有完整终态",
            )
            atomic_json(receipt_path, receipt)
            return code
    except Exception as error:
        receipt.update(
            status="requires_original_run_inspection", error=f"{type(error).__name__}: {error}"
        )
        atomic_json(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    request_path = parser.parse_args().request.resolve()
    require(request_path.is_relative_to(PROJECT), "请求必须在GPLSACO内")
    request = json.loads(request_path.read_text())
    # 只锁编排者；PersistentGpuWorker仍独立持有共享GPU锁，避免父子自锁。
    with (PROJECT / ".tmp" / f"baseline-waiter-{request['gpu_uuid']}.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run(request_path))


if __name__ == "__main__":
    main()
