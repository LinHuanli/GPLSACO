#!/usr/bin/env python3
"""每张A5000一个常驻会话，从协调器邮箱顺序接收独立任务。"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read
from gp_faco.campaign_jobs import solve_job
from gp_faco.checkpoint import atomic_json
from gp_faco.distributed_population import solve_population_job
from gp_faco.gpu_session import GpuSession
from gp_faco.remote import environment, process_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    directory = args.campaign.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("worker产物必须在项目内")
    worker = directory / "workers" / args.gpu
    worker.mkdir(parents=True, exist_ok=True)
    lease = os.fdopen(args.lock_fd) if args.lock_fd is not None else (worker / "running.lock").open("a")
    if args.lock_fd is None:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "running", "runtime": read(worker / "runtime.json")})); return
    if args.launch:
        with (worker / "worker.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--campaign", str(directory),
                "--gpu", args.gpu, "--lock-fd", str(lease.fileno())], cwd=PROJECT, env=environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lease.fileno(),))
        print(json.dumps({"status": "launched", "pid": process.pid})); lease.close(); return
    receipt = {**process_receipt(), "gpu_uuid": args.gpu}
    atomic_json(worker / "runtime.json", {**receipt, "status": "starting", "updated_unix": time.time()})
    session = None
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    current = None
    try:
        config = read(directory / "campaign.json")
        session = GpuSession(args.gpu, "exact", extension_directory=config["build"])
        atomic_json(worker / "runtime.json", {**receipt, "status": "idle", "device": session.device, "updated_unix": time.time()})
        last = None
        idle_since = time.monotonic()
        while not stop and not (directory / "stop_workers.json").exists():
            request = read(worker / "request.json")
            if request and (request.get("worker_pid"), request.get("worker_start_ticks")) != (receipt["pid"], receipt["start_ticks"]):
                request = None  # 上一worker遗留的邮箱不能触发无授权的再次执行。
            token = (request["job"], request["attempt"]) if request else None
            if request is None or token == last:
                if time.monotonic() - idle_since > 60:
                    break  # 没有可执行任务时释放GPU；不是求解或进化时间截止。
                time.sleep(0.25)
                continue
            path = directory / "jobs" / request["job"]
            last = token
            if (path / "result.json").exists():
                continue
            current = path
            job = read(path / "job.json")
            started = time.time()
            runtime = {**receipt, "status": "running", "attempt": request["attempt"],
                       "job": job["id"], "started_unix": started}
            atomic_json(path / "runtime.json", runtime)
            atomic_json(worker / "runtime.json", {**runtime, "updated_unix": started})
            if job.get("pilot"):
                from gp_faco.representation_pilot import solve_gpu_job
                result = solve_gpu_job(directory, job, session)
            else:
                result = solve_population_job(directory, job, session) if job["kind"] == "population" else \
                    solve_job(directory, job, args.gpu, gpu_session=session)
            result.update(job=job["id"], wall_seconds=time.time() - started, execution=runtime)
            atomic_json(path / "result.json", result)
            atomic_json(path / "runtime.json", {**runtime, "status": "completed", "finished_unix": time.time()})
            current = None
            idle_since = time.monotonic()
            atomic_json(worker / "runtime.json", {**receipt, "status": "idle", "last_job": job["id"], "updated_unix": time.time()})
    except BaseException as error:
        busy = isinstance(error, BlockingIOError) or "新会话需要实时空闲" in str(error)
        failure = {**receipt, "status": "resource_wait" if busy else "failed", "error": str(error),
                   "retryable": not isinstance(error, (ValueError, TypeError, AssertionError)), "finished_unix": time.time()}
        if current:
            atomic_json(current / "runtime.json", {**read(current / "runtime.json", {}), **failure})
        atomic_json(worker / "runtime.json", failure)
        traceback.print_exc()
        raise
    else:
        atomic_json(worker / "runtime.json", {**receipt, "status": "stopped", "finished_unix": time.time()})
    finally:
        if session:
            session.close()
        lease.close()


if __name__ == "__main__":
    main()
