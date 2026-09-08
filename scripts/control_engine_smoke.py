#!/usr/bin/env python3
"""开发池固定32-colony的真实GP控制计时；未训练规则不作为研究性能结论。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(PROJECT / "build/cuda"))

import gp_faco_ext as native  # noqa: E402
from gp_faco.data import point_set_hash, tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/gpu/control-engine/smoke"
    )
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT):
        parser.error("所有产物必须位于GPLSACO内")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        parser.error("CUDA_VISIBLE_DEVICES必须是指定UUID")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("已有完整运行，使用新目录")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if any(row and row[0].strip() == args.gpu_uuid for row in csv.reader(apps.splitlines())):
        raise RuntimeError("目标GPU已有计算进程")
    device = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu_uuid}",
            "--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    row = next(csv.reader(device.splitlines()))
    if int(row[2]) > 1024 or int(row[3]) > 5:
        raise RuntimeError("目标GPU未满足空闲阈值")
    (args.output / "device.csv").write_text(device)
    all_runs = []
    manifest = {
        "split": "development",
        "instances_per_scale": 16,
        "seeds": [17, 29],
        "colonies": 32,
        "ants_per_colony": 32,
        "programs": {
            "keep_mne16": Program((0,), (5,)).to_dict(),
            "prefer_restart": Program((0,), (4,)).to_dict(),
        },
        "split_sha256": hashlib.sha256(
            (PROJECT / "provenance/splits.v1.json").read_bytes()
        ).hexdigest(),
        "records": {},
        "registration_fees": {},
    }
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as data:
        for n, budgets in ((500, [0.002, 0.5, 1.0]), (1000, [0.002, 1.0, 2.0])):
            engine = native.FacoBatchEngine(n, 32, native.FixedFacoSettings())
            ids = data.record_ids("development", n)[:16]
            if len(ids) != 16:
                raise RuntimeError("开发池不足16实例")
            manifest["records"][str(n)] = ids
            instances, keys, seeds, fees = [], [], [], {}
            for record_id in ids:
                instance = data.load_instance(record_id)
                key = int(point_set_hash(instance)[:16], 16)
                fees[record_id] = engine.register_problem(
                    key, np.asarray(instance.coordinates, dtype=np.float64)
                )
                for seed in (17, 29):
                    instances.append(instance)
                    keys.append(key)
                    seeds.append(seed)
            manifest["registration_fees"][str(n)] = fees
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for mode in ("cached_charged", "end_to_end"):
                for program_name, program in manifest["programs"].items():
                    for budget in budgets:
                        result = engine.evaluate_program(
                            np.asarray(keys, dtype=np.uint64),
                            np.asarray(seeds, dtype=np.uint64),
                            budget,
                            program,
                            mode,
                        )
                        checked = []
                        for instance, seed, item in zip(
                            instances, seeds, result["items"], strict=True
                        ):
                            row = {
                                "record_id": instance.instance_id,
                                "seed": seed,
                                "has_incumbent": item["has_incumbent"],
                                "completed_seconds": item["completed_seconds"],
                            }
                            if item["has_incumbent"]:
                                independent = tour_cost(instance, item["tour"])
                                if (
                                    abs(independent - item["cost"]) > 1e-8
                                    or item["completed_seconds"] > budget
                                ):
                                    raise RuntimeError("成本不符或输出迟到")
                                label = data.load_label(instance.instance_id)
                                row.update(
                                    {
                                        "cost": item["cost"],
                                        "independent_cost": independent,
                                        "absolute_cost_error": abs(independent - item["cost"]),
                                        "reference_gap_percent": 100
                                        * (independent / label.cost - 1),
                                    }
                                )
                            checked.append(row)
                        name = f"{n}-{program_name}-{mode}-{budget:g}.json"
                        (args.output / name).write_text(json.dumps(result, indent=2) + "\n")
                        entry = {key: value for key, value in result.items() if key != "items"}
                        entry.update(
                            {"dimension": n, "program_name": program_name, "checked_items": checked}
                        )
                        all_runs.append(entry)
                        present = sum(item["has_incumbent"] for item in checked)
                        print(
                            f"n={n} {program_name} {mode} B={budget:g}: {present}/32 feasible, "
                            f"batches={result['completed_batches']}, "
                            f"discarded={result['discarded_batches']}",
                            flush=True,
                        )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "scope": "untrained GP control development timing; training and E1-E4 pending",
                "runs": all_runs,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
