#!/usr/bin/env python3
"""四个真实非测试实例的CPU成对准备、缓存复用、进程重建恢复及独立图核验。"""

import argparse
import importlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from audit_e3_graphs import audit_instance  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_preparation import initialize_worker, run_job  # noqa: E402
from gp_faco.e3_protocol import require, validate_config  # noqa: E402
from gp_faco.worker import content_hash, file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "dataset-root", "candidate-binary", "reuse-catalog", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(
        args.output.is_relative_to(PROJECT) and not args.output.exists(), "需要工作树内全新工程输出"
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU检查须禁用GPU")
    config = json.loads((PROJECT / "configs/e3_protocol_v1.json").read_text())
    validate_config(config)
    native_path = (PROJECT / "build/cuda/gp_faco_ext.so").resolve()
    plan = {
        "scope": "engineering_preparation_check; not a formal protocol freeze",
        "config": config,
        "database_sha256": file_hash(args.database),
        "native_binary_path": str(native_path.relative_to(PROJECT)),
        "native_binary_sha256": file_hash(native_path),
        "candidate_binary_path": str(args.candidate_binary.resolve().relative_to(PROJECT)),
        "candidate_binary_sha256": file_hash(args.candidate_binary),
        "candidate_build_manifest_sha256": file_hash(
            args.candidate_binary.parent / "build-manifest.json"
        ),
        "reuse_catalog_path": str(args.reuse_catalog.resolve().relative_to(PROJECT)),
        "reuse_catalog_sha256": file_hash(args.reuse_catalog),
        "preparation_sources": {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in [
                Path(__file__),
                PROJECT / "scripts/audit_e3_graphs.py",
                *[
                    PROJECT / "python/gp_faco" / name
                    for name in (
                        "candidate_prior.py",
                        "e3_preparation.py",
                        "e3_protocol.py",
                        "graph_matching.py",
                        "graph_catalog.py",
                    )
                ],
            ]
        },
    }
    plan["sha256"] = content_hash(plan)
    jobs = []
    with IndexedDataset(args.database, args.dataset_root) as source:
        for n in (500, 1000):
            for split in ("train", "development"):
                jobs.append(
                    {
                        "index": len(jobs),
                        "dimension": n,
                        "instance_id": sorted(source.record_ids(split, n))[0],
                        "roles": ["engineering_" + split],
                    }
                )
    args.output.mkdir(parents=True)
    atomic_json(args.output / "plan.json", plan)
    atomic_json(args.output / "jobs.json", jobs)
    started, phases = time.perf_counter(), []
    for phase in range(2):
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(plan, str(args.database), str(args.dataset_root), str(args.output)),
        ) as pool:
            rows = list(pool.map(run_job, jobs))
        require(
            all(row["resumed"] == (phase == 1) for row in rows), "首次准备/进程重建恢复身份不符"
        )
        phases.append(rows)
        atomic_json(args.output / f"phase-{phase}.json", rows)
        print(json.dumps({"phase": phase, "rows": rows}), flush=True)
    require(
        [r["receipt_sha256"] for r in phases[0]] == [r["receipt_sha256"] for r in phases[1]],
        "进程重建改变了成功收据",
    )
    sys.path.insert(0, str(native_path.parent))
    native = importlib.import_module("gp_faco_ext")
    native_settings = native.FixedFacoSettings()
    for name, value in config["solver"].items():
        setattr(native_settings, name, value)
    audits = []
    with IndexedDataset(args.database, args.dataset_root) as source:
        for job in jobs:
            payload = json.loads(
                (args.output / "instances" / f"{job['index']:05d}" / "graphs.json").read_text()
            )
            value = audit_instance(
                source.load_instance(job["instance_id"]), payload, plan, native, native_settings
            )
            audits.append({"job": job, **value})
    report = {
        "status": "passed",
        "scope": plan["scope"],
        "plan_sha256": plan["sha256"],
        "instances": len(jobs),
        "graphs": 2 * len(jobs),
        "new_native_prior_calls": 4,
        "reused_native_prior_receipts": 4,
        "resumed_receipts_unchanged": 4,
        "wall_seconds": time.perf_counter() - started,
        "independent_audits": audits,
        "max_common_cost_error": max(v["common_cost_absolute_error"] for v in audits),
    }
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "instances", "graphs", "wall_seconds", "max_common_cost_error")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
