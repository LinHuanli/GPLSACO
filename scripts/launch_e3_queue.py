#!/usr/bin/env python3
"""按冻结设备分组顺序运行正式E3任务；仅等待真实CLI退出，不设置算法时间上限。"""

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.e3_protocol import require  # noqa: E402
from gp_faco.worker import content_hash, file_hash  # noqa: E402
from research_e3 import load_execution  # noqa: E402


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    require(
        plan["sha256"] == content_hash({k: v for k, v in plan.items() if k != "sha256"}),
        "队列计划摘要改变",
    )
    require(plan["launcher_sha256"] == file_hash(Path(__file__)), "启动器改变")
    groups = [g for g in plan["groups"] if g["gpu_uuid"] == args.gpu_uuid]
    require(len(groups) == 1, "GPU没有唯一队列")
    group = groups[0]
    require(group["host"] == socket.gethostname(), "不能静默迁移运行节点")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "可见GPU与队列不符")
    _, execution, _ = load_execution(PROJECT / plan["execution_directory"], Path(plan["database"]))
    require(execution["sha256"] == plan["execution_sha256"], "执行冻结改变")
    state_path = PROJECT / group["queue_receipt"]
    require(not state_path.exists(), "队列已有提交记录；必须先核查原进程，禁止盲目重启")
    state = {
        "plan_sha256": plan["sha256"],
        "gpu_uuid": args.gpu_uuid,
        "host": socket.gethostname(),
        "launcher_pid": os.getpid(),
        "started_at_utc": now(),
        "status": "running",
        "jobs": [],
    }
    atomic_json(state_path, state)
    for job in group["jobs"]:
        output, log, resources = (PROJECT / job[k] for k in ("output", "log", "resources"))
        require(
            all(p.is_relative_to(PROJECT) for p in (output, log, resources)), "全部产物须在工作树"
        )
        require(not any(p.exists() for p in (output, log, resources)), "原运行产物不可覆盖")
        command = [
            sys.executable,
            str(PROJECT / "scripts/research_e3.py"),
            job["stage"],
            "--execution",
            str(PROJECT / plan["execution_directory"]),
            "--database",
            plan["database"],
            "--dataset-root",
            plan["dataset_root"],
            "--output",
            str(output),
            "--gpu-uuid",
            args.gpu_uuid,
            "--condition",
            job["condition"],
        ]
        if job["stage"] == "train":
            command += ["--evolution-seed", str(job["evolution_seed"])]
        receipt = {"job": job, "command": command, "status": "launching", "at_utc": now()}
        state["jobs"].append(receipt)
        atomic_json(state_path, state)
        log.parent.mkdir(parents=True, exist_ok=True)
        timing = (
            '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,'
            '"max_rss_kib":%M,"exit_code":%x}'
        )
        with log.open("xb") as stream:
            process = subprocess.Popen(
                ["/usr/bin/time", "-q", "-f", timing, "-o", str(resources), *command],
                cwd=PROJECT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            receipt.update(status="running", time_wrapper_pid=process.pid)
            atomic_json(state_path, state)
            # 不取消Future、不按耗时终止；此等待只观察独立CLI的真实终态。
            code = process.wait()
        receipt.update(status="exited", exit_code=code, exited_at_utc=now())
        atomic_json(state_path, state)
        try:
            costs = json.loads(resources.read_text())
            summary = load_checkpoint(output / "summary.json")
            require(
                code in (0, 1)
                and costs["exit_code"] == code
                and summary["status"] == ("complete" if code == 0 else "failed"),
                "CLI与完整研究终态不一致",
            )
        except Exception as error:
            state.update(
                status="requires_original_run_inspection", error=f"{type(error).__name__}: {error}"
            )
            atomic_json(state_path, state)
            raise
        receipt.update(
            summary_sha256=file_hash(output / "summary.json"),
            resources_sha256=file_hash(resources),
            run_id=summary["run_id"],
        )
        atomic_json(state_path, state)
        print(
            json.dumps(
                {
                    "event": "actual_cli_exit",
                    "job": job["name"],
                    "exit_code": code,
                    "run_id": summary["run_id"],
                }
            ),
            flush=True,
        )
        # 后续GP不读取Static的得分或选中策略；顺序仅复用已释放的固定设备。
    state.update(status="all_planned_clis_exited", finished_at_utc=now())
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
