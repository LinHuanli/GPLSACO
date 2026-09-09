#!/usr/bin/env python3
"""调度器的远程任务入口：租约跨启动继承，结果先落盘，进程状态另记。"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.campaign import read  # noqa: E402
from gp_faco.campaign_calibration import solve_calibration  # noqa: E402
from gp_faco.campaign_jobs import run_training, solve_job  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.remote import environment, process_receipt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    directory = args.campaign.resolve()
    job_dir = directory / "jobs" / args.job
    if not job_dir.resolve().is_relative_to(PROJECT):
        raise ValueError("任务目录必须在项目内")
    job = read(job_dir / "job.json")
    runtime = job_dir / "runtime.json"
    if args.launch:
        if (job_dir / "result.json").exists():
            print(json.dumps({"status": "completed", "job": args.job}))
            return
        lease = (job_dir / "running.lock").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "running", "runtime": read(runtime)}))
            return
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--campaign",
            str(directory),
            "--job",
            args.job,
            "--attempt",
            str(args.attempt),
            "--lock-fd",
            str(lease.fileno()),
        ]
        if args.gpu:
            command += ["--gpu", args.gpu]
        with (job_dir / f"attempt-{args.attempt:02d}.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lease.fileno(),),
            )
        print(json.dumps({"status": "launched", "pid": process.pid}), flush=True)
        # 不执行 LOCK_UN；子进程继承同一 open-file-description 的租约。
        lease.close()
        return
    lease = (
        os.fdopen(args.lock_fd)
        if args.lock_fd is not None
        else (job_dir / "running.lock").open("a")
    )
    if args.lock_fd is None:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    receipt = {
        **process_receipt(),
        "status": "running",
        "attempt": args.attempt,
        "job": args.job,
        "gpu_uuid": args.gpu,
        "started_unix": started,
    }
    atomic_json(runtime, receipt)
    try:
        if job["kind"] == "calibration":
            result = solve_calibration(directory, job, args.gpu)
        elif job["kind"] == "training":
            configuration = read(directory / "campaign.json")
            if configuration.get("distributed_population") or configuration.get("representation_pilot"):
                if configuration.get("representation_pilot"):
                    from gp_faco.representation_pilot import run_training as run_training_pool
                else:
                    from gp_faco.campaign_tsp500 import run_training_pool
                result = run_training_pool(directory, job)
                result.update(job=args.job, wall_seconds=time.time() - started, execution=receipt)
                atomic_json(job_dir / "result.json", result)
                atomic_json(runtime, {**receipt, "status": "completed", "finished_unix": time.time()})
                return
            gpu = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu}",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                .strip()
                .split(", ")
            )
            if gpu[0] != "NVIDIA RTX A5000":
                raise ValueError("训练只允许 A5000")
            result = run_training(directory, job, args.gpu, {"driver": gpu[1]})
        else:
            result = solve_job(directory, job, args.gpu)
        result.update(job=args.job, wall_seconds=time.time() - started, execution=receipt)
        atomic_json(job_dir / "result.json", result)
        atomic_json(runtime, {**receipt, "status": "completed", "finished_unix": time.time()})
    except BaseException as error:
        if type(error).__name__ == "PilotPaused":
            atomic_json(runtime, {**receipt, "status": "paused", "finished_unix": time.time()})
            return
        status = (
            "resource_wait"
            if "新会话需要实时空闲" in str(error) or isinstance(error, BlockingIOError)
            else "failed"
        )
        retryable = not isinstance(error, (ValueError, TypeError, AssertionError))
        atomic_json(
            job_dir / f"failure-{args.attempt:02d}.json",
            {
                **receipt,
                "status": status,
                "error": str(error),
                "retryable": retryable,
                "finished_unix": time.time(),
            },
        )
        atomic_json(
            runtime, {**receipt, "status": status, "error": str(error), "retryable": retryable}
        )
        traceback.print_exc()
        raise
    finally:
        lease.close()


if __name__ == "__main__":
    main()
