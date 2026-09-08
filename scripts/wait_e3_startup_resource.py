#!/usr/bin/env python3
"""接续有独立证据的E3 worker初始化资源竞争；原待办未提交，等待不启动CUDA。"""

import argparse
import csv
import datetime
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
from gp_faco.e3_protocol import require  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402


def verify_startup_pause(state, directory, request):
    incident_path = (PROJECT / request["startup_incident"]).resolve()
    require(
        incident_path.is_relative_to(PROJECT)
        and file_hash(incident_path) == request["startup_incident_sha256"],
        "启动竞争证据改变",
    )
    incident = json.loads(incident_path.read_text())
    prefix = load_checkpoint(incident_path.parent / "prefix-snapshot/checkpoint.json")
    pending = state["pending"]
    require(
        incident["run_id"] == state["run_id"] == request["run_id"]
        and incident["new_native_submissions"] == 0
        and incident["checkpoint_sha256"] == request["paused_checkpoint_sha256"]
        and state["completed"] == prefix["completed"]
        and state["costs"] == prefix["costs"]
        and state["worker_history"] == prefix["worker_history"]
        and state["active_worker"] is None
        and isinstance(pending, dict)
        and pending["attempts"] == []
        and pending["key"] == incident["pending_key"]
        and not (directory / "tasks" / (pending["key"] + ".json")).exists(),
        "不是完整保留旧结果的未提交启动竞争",
    )
    log_path, resources_path = PROJECT / incident["log"], PROJECT / incident["resources"]
    require(
        file_hash(log_path) == incident["log_sha256"]
        and file_hash(resources_path) == incident["resources_sha256"]
        and json.loads(resources_path.read_text())["exit_code"] == 1
        and "RuntimeError: 指定GPU已有计算进程，worker不启动" in log_path.read_text(),
        "缺少明确GPU初始化占用错误",
    )
    audit_path = PROJECT / request["startup_audit"]
    require(file_hash(audit_path) == request["startup_audit_sha256"], "启动竞争审计改变")
    audit = json.loads(audit_path.read_text())
    require(
        audit["status"] == "snapshot_passed" and audit["run_id"] == state["run_id"],
        "未通过独立暂停点审计",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request_path = args.request.resolve()
    require(request_path.is_relative_to(PROJECT), "等待请求必须位于工作树")
    request = json.loads(request_path.read_text())
    request_sha, script_sha = file_hash(request_path), file_hash(Path(__file__))
    paths = {}
    for name in ("input", "receipt", "log", "resources"):
        paths[name] = (PROJECT / request[name]).resolve()
        require(paths[name].is_relative_to(PROJECT), "等待产物必须位于工作树")
    require(
        request["no_wall_clock_limit"] and socket.gethostname() == request["host"], "原host不符"
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == request["gpu_uuid"], "原UUID不符")
    require(not any(paths[k].exists() for k in ("receipt", "log", "resources")), "等待产物已存在")
    for name, digest in request["sources"].items():
        require(file_hash(PROJECT / name) == digest, "恢复入口或政策改变")
    verify_startup_pause(
        load_checkpoint(paths["input"] / "checkpoint.json"), paths["input"], request
    )
    receipt = {
        "request_sha256": request_sha,
        "wrapper_sha256": script_sha,
        "run_id": request["run_id"],
        "host": request["host"],
        "gpu_uuid": request["gpu_uuid"],
        "waiting_samples": 0,
        "status": "waiting_for_idle_gpu",
        "native_submissions_while_waiting": 0,
    }
    try:
        while True:
            require(file_hash(request_path) == request_sha, "等待期间请求改变")
            require(file_hash(Path(__file__)) == script_sha, "等待期间编排实现改变")
            observation = {}
            try:
                apps = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                    text=True,
                )
                observation["foreign_processes"] = sum(
                    bool(row and row[0].strip() == request["gpu_uuid"])
                    for row in csv.reader(apps.splitlines())
                )
                row = next(
                    csv.reader(
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--id=" + request["gpu_uuid"],
                                "--query-gpu=name,driver_version,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                        ).splitlines()
                    )
                )
                model, driver, memory, utilization = [v.strip() for v in row]
                observation.update(
                    gpu_model=model,
                    driver_version=driver,
                    memory_used_mib=int(memory),
                    utilization_percent=int(utilization),
                )
            except Exception as error:
                observation.update(error=f"{type(error).__name__}: {error}", foreign_processes=None)
            if "gpu_model" in observation:
                require(
                    observation["gpu_model"] == request["gpu_model"]
                    and observation["driver_version"] == request["driver_version"],
                    "等待期间硬件身份改变",
                )
            receipt.update(
                observation=observation,
                observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            free = (
                observation.get("foreign_processes") == 0
                and observation.get("memory_used_mib", 1 << 30) <= 1024
                and observation.get("utilization_percent", 100) <= 5
            )
            atomic_json(paths["receipt"], receipt)
            print(json.dumps(receipt, ensure_ascii=False), flush=True)
            if free:
                break
            receipt["waiting_samples"] += 1
            time.sleep(5)
        # 其他操作者若已推进原运行，停止该等待器，不覆盖或再次启动。
        checkpoint = paths["input"] / "checkpoint.json"
        require(
            file_hash(checkpoint) == request["paused_checkpoint_sha256"], "原暂停点已由别处推进"
        )
        state = load_checkpoint(checkpoint)
        verify_startup_pause(state, paths["input"], request)
        for name, digest in request["sources"].items():
            require(file_hash(PROJECT / name) == digest, "启动前恢复入口或政策改变")
        command = [
            "/usr/bin/time",
            "-q",
            "-f",
            '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"exit_code":%x}',
            "-o",
            str(paths["resources"]),
            *request["command"],
        ]
        with paths["log"].open("x") as stream:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
            receipt.update(status="original_run_resuming", child_pid=child.pid)
            atomic_json(paths["receipt"], receipt)
            code = child.wait()
        receipt.update(status="resume_cli_exited; inspect terminal audit", exit_code=code)
        atomic_json(paths["receipt"], receipt)
        raise SystemExit(code)
    except Exception as error:
        receipt.update(
            status="requires_original_run_inspection", error=f"{type(error).__name__}: {error}"
        )
        atomic_json(paths["receipt"], receipt)
        raise


if __name__ == "__main__":
    main()
