#!/usr/bin/env python3
"""按用户资源变更在已落盘任务边界停用既有进程；不修改冻结求解代码。"""

import argparse
import ctypes
import json
import os
import select
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    stat = directory.joinpath("stat").read_text().split(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(stat[19]),
        "uid": directory.stat().st_uid,
        "state": stat[0],
        "cwd": str(directory.joinpath("cwd").resolve()),
        "args": [s.decode() for s in directory.joinpath("cmdline").read_bytes().split(b"\0") if s],
    }


def open_verified(expected):
    # pidfd与启动tick共同防止陈旧PID误中另一个进程。
    descriptor = os.pidfd_open(expected["pid"])
    actual = process_identity(expected["pid"])
    if actual["uid"] != os.getuid() or any(
        actual[key] != expected[key] for key in ("start_ticks", "cwd", "args")
    ):
        os.close(descriptor)
        raise ValueError("目标进程身份已改变，不能发送信号")
    return descriptor


def stopped(expected):
    while True:
        value = process_identity(expected["pid"])
        if value["start_ticks"] != expected["start_ticks"]:
            raise ValueError("进程PID已被复用")
        if value["state"] in ("T", "t"):
            return
        if value["state"] == "Z":
            raise ValueError("目标已退出，需核查终态")
        time.sleep(0.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text())
    root, directory = Path(request["project_root"]), Path(request["run_directory"])
    if not request_path.is_relative_to(root) or not directory.is_relative_to(root):
        raise ValueError("文件必须位于本项目")
    if request["host"] != socket.getfqdn():
        raise ValueError("只停用当前主机上明确登记的任务")
    target = request.get("coordinator")
    receipt_path = request_path.with_name("pause-receipt.json")
    if receipt_path.exists():
        raise ValueError("已有停用记录，不能重复执行")
    receipt = {
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": request["reason"],
        "name": request["name"],
        "gpu_uuid": request["gpu_uuid"],
        "host": request["host"],
        "status": "disabling_automatic_restarts",
        "boundary_observations": [],
    }
    write(receipt_path, receipt)
    wrappers = []
    for expected in request["wrappers"]:
        fd = open_verified(expected)
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        stopped(expected)
        wrappers.append((expected, fd))
    # 等待器本身不持有GPU；先阻止它自动启动下一个训练进程。
    receipt["wrappers_stopped"] = [p["pid"] for p, _ in wrappers]
    write(receipt_path, receipt)
    if target is not None:
        fd = open_verified(target)
        checkpoint = directory / "checkpoint.json"
        libc = ctypes.CDLL(None, use_errno=True)
        watch = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if watch < 0 or libc.inotify_add_watch(watch, os.fsencode(directory), 0x80) < 0:
            raise OSError(ctypes.get_errno(), "无法观察本机原子checkpoint替换")
        while True:
            signal.pidfd_send_signal(fd, signal.SIGSTOP)
            stopped(target)
            raw = checkpoint.read_bytes()
            envelope = json.loads(raw)
            state, pending = envelope["state"], envelope["state"]["pending"]
            boundary = pending is None or not pending["attempts"]
            receipt["boundary_observations"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_calls": state["costs"]["solve_jobs"],
                    "pending_key": None if pending is None else pending["key"],
                    "pending_attempts": None if pending is None else pending["attempts"],
                    "safe_boundary": boundary,
                }
            )
            write(receipt_path, receipt)
            if boundary:
                request_path.with_name("boundary-checkpoint.json").write_bytes(raw)
                receipt.update(status="boundary_verified_closing_worker", costs=state["costs"])
                write(receipt_path, receipt)
                # SIGINT走既有finally->worker.close；只在没有活动提交的落盘边界触发。
                signal.pidfd_send_signal(fd, signal.SIGINT)
                signal.pidfd_send_signal(fd, signal.SIGCONT)
                while not select.select([fd], [], [], 1)[0]:
                    pass
                receipt["coordinator_exit_confirmed_by_pidfd"] = True
                break
            # 该任务已经提交：仅让同一任务返回并落盘，不重启或丢弃原始结果。
            try:
                while os.read(watch, 65536):
                    pass
            except BlockingIOError:
                pass
            signal.pidfd_send_signal(fd, signal.SIGCONT)
            while not select.select([watch, fd], [], [], 1)[0]:
                pass
            if select.select([fd], [], [], 0)[0]:
                raise ValueError("观察边界前协调进程自行退出，需检查既有产物")
        os.close(watch)
        os.close(fd)
    for _expected, fd in wrappers:
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        signal.pidfd_send_signal(fd, signal.SIGCONT)
        while not select.select([fd], [], [], 1)[0]:
            pass
        os.close(fd)
    receipt.update(
        status="registered_processes_exited", exited_at_utc=datetime.now(timezone.utc).isoformat()
    )
    write(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
