#!/usr/bin/env python3
"""仅在预登记开发池运行固定迭代GPU FACO，外部独立重算返回路线。"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/gpu/fixed-faco/smoke")
    parser.add_argument("--instances-per-scale", type=int, default=8)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT):
        parser.error("结果必须保存在GPLSACO内")
    if args.instances_per_scale < 1 or args.batches < 1:
        parser.error("实例数与批次数必须为正")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("已有完整运行；使用新目录，不覆盖历史结果")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        parser.error("CUDA_VISIBLE_DEVICES必须设为本次指定UUID")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if any(row and row[0].strip() == args.gpu_uuid for row in csv.reader(apps.splitlines())):
        raise RuntimeError("目标GPU已有计算进程，本次不启动")
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
        raise RuntimeError("目标GPU显存/利用率不满足空闲阈值")
    (args.output / "device.csv").write_text(device)
    settings = native.FixedFacoSettings()
    seeds = [17, 29, 43]
    manifest = {
        "variant": "FACO-GPU-IterationPilot",
        "split": "development",
        "batches": args.batches,
        "mne_target": 8,
        "seeds": seeds,
        "settings": {
            name: getattr(settings, name)
            for name in (
                "ants",
                "primary_width",
                "backup_width",
                "ls_width",
                "beta",
                "retention",
                "p_best",
                "epoch_source_probability",
                "ls_evaluation_limit",
                "initial_ls_evaluation_limit",
            )
        },
        "device": native.cuda_device_info(),
        "split_manifest_sha256": hashlib.sha256(
            (PROJECT / "provenance/splits.v1.json").read_bytes()
        ).hexdigest(),
        "records": {},
    }
    runs = []
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as dataset:
        for dimension in (500, 1000):
            ids = dataset.record_ids("development", dimension)[: args.instances_per_scale]
            if len(ids) != args.instances_per_scale:
                raise ValueError("开发池数量不足")
            manifest["records"][str(dimension)] = ids
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            for record_id in ids:
                instance = dataset.load_instance(record_id)
                coordinates = np.asarray(instance.coordinates, dtype=np.float64)
                key = int(point_set_hash(instance)[:16], 16)
                # 注册和完整求解只接收坐标、配置及不含标签的key。
                engine = native.FixedFacoGpu(coordinates, key, settings)
                for seed in seeds:
                    result = engine.run_iterations(seed, args.batches, 8)
                    independent = tour_cost(instance, result["tour"])
                    error = abs(independent - result["cost"])
                    if error > 1e-8 or result["cost"] > result["initial_cost"] + 1e-8:
                        raise RuntimeError("返回成本与独立重算不符或GB劣化")
                    # 标签只在求解完成后由外部evaluator读取。
                    label = dataset.load_label(record_id)
                    record = {
                        "record_id": record_id,
                        "dimension": dimension,
                        "seed": seed,
                        **result,
                        "independent_cost": independent,
                        "absolute_cost_error": error,
                        "reference_cost": label.cost,
                        "reference_gap_percent": 100 * (independent / label.cost - 1),
                        "certificate_status": label.certificate_status,
                        "valid": True,
                    }
                    name = f"{dimension}-{record_id[:16]}-{seed}.json"
                    (args.output / name).write_text(json.dumps(record, indent=2) + "\n")
                    runs.append({k: v for k, v in record.items() if k != "tour"})
                    print(
                        f"{dimension} {record_id[:12]} seed={seed}: valid, batches={args.batches}",
                        flush=True,
                    )
    summary = {
        "status": "passed",
        "scope": (
            "development-only fixed-iteration feasibility pilot; no formal time-budget comparison"
        ),
        "runs": runs,
        "negative_reference_gaps": sum(run["reference_gap_percent"] < -1e-8 for run in runs),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
