#!/usr/bin/env python3
"""执行已冻结全部E3非测试图准备；CPU并行，不设原生候选超时，不提前发布目录。"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.e3_preparation import (  # noqa: E402
    checked_json,
    exclusive_lock,
    host_identity,
    initialize_worker,
    load_freeze,
    run_job,
    verify_completed,
)
from gp_faco.e3_protocol import require  # noqa: E402
from gp_faco.worker import content_hash, file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    args.freeze, args.output = args.freeze.resolve(), args.output.resolve()
    require(
        args.freeze.is_relative_to(PROJECT) and args.output.is_relative_to(PROJECT),
        "冻结输入与准备产物必须位于工作树",
    )
    require(
        type(args.processes) is int and 1 <= args.processes <= len(os.sched_getaffinity(0)),
        "CPU进程数越界",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "准备协调器也须显式禁用GPU可见性")
    plan, jobs = load_freeze(args.freeze, args.database)
    with exclusive_lock(args.output / ".run.lock"):
        identity = {
            "plan_sha256": plan["sha256"],
            "jobs_sha256": content_hash(jobs),
            "database_sha256": plan["database_sha256"],
            "wall_clock_limit": None,
            "scope": "complete frozen non-test paired graph preparation; no labels",
        }
        identity_path = args.output / "identity.json"
        if identity_path.exists():
            require(checked_json(identity_path) == identity, "恢复准备身份不同")
        else:
            atomic_json(identity_path, identity)
        sessions = args.output / "sessions"
        sessions.mkdir(exist_ok=True)
        session_path = sessions / f"{len(list(sessions.glob('*.json'))):04d}.json"
        session = {
            **host_identity(),
            "processes": args.processes,
            "identity_sha256": content_hash(identity),
            "status": "running",
        }
        atomic_json(session_path, session)
        started, completed, failures = time.perf_counter(), {}, []
        # 有界队列不预投递全量；出现失败只收回已在途结果，保留其完整收据。
        iterator = iter(jobs)
        with ProcessPoolExecutor(
            max_workers=args.processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(plan, str(args.database), str(args.dataset_root), str(args.output)),
        ) as pool:
            pending = {}

            def submit_one():
                job = next(iterator, None)
                if job is not None:
                    pending[pool.submit(run_job, job)] = job

            for _ in range(args.processes):
                submit_one()
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                # 30秒仅用于进度观察；同一Future一直等待，不是算法时间预算。
                for future in done:
                    job = pending.pop(future)
                    try:
                        row = future.result()
                        completed[row["index"]] = row
                    except Exception as exc:
                        failures.append(
                            {"job": job, "type": type(exc).__name__, "message": str(exc)}
                        )
                if not failures:
                    for _ in done:
                        submit_one()
                observation = {
                    "status": "running" if not failures else "draining_after_failure",
                    "completed": len(completed),
                    "expected": len(jobs),
                    "active": len(pending),
                    "failures": failures,
                    "wall_seconds": time.perf_counter() - started,
                }
                atomic_json(args.output / "observation.json", observation)
                if len(completed) % 16 < len(done) or not done or failures or not pending:
                    print(json.dumps(observation, ensure_ascii=False), flush=True)
        session.update(
            status="failed" if failures else "complete",
            completed=len(completed),
            failures=failures,
            wall_seconds=time.perf_counter() - started,
        )
        atomic_json(session_path, session)
        if failures:
            raise RuntimeError("图准备保留失败及已完成任务；详见observation.json，不自动重跑")
        require(len(completed) == len(jobs), "图准备未覆盖全部冻结成员")
        # 再读逐实例收据后才写终态manifest；独立审计随后才能发布可供训练的catalog。
        rows = []
        for job in jobs:
            directory = args.output / "instances" / f"{job['index']:05d}"
            receipt = verify_completed(directory, job, plan)
            rows.append(
                {
                    "job": job,
                    "mode": receipt["mode"],
                    "receipt_path": str((directory / "complete.json").relative_to(PROJECT)),
                    "receipt_sha256": file_hash(directory / "complete.json"),
                }
            )
        manifest = {
            "status": "complete_pending_independent_audit",
            "plan_sha256": plan["sha256"],
            "identity_sha256": file_hash(identity_path),
            "instances": len(rows),
            "rows": rows,
            "sessions": {
                str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(sessions.glob("*.json"))
            },
            "formal_test_released": False,
            "catalog_released": False,
        }
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            previous = checked_json(manifest_path)
            require(previous["rows"] == rows, "已封存准备终态改变")
        else:
            atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "instances": len(rows),
                    "manifest_sha256": file_hash(manifest_path),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
