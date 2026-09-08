#!/usr/bin/env python3
"""执行完整GP训练/验证状态机；当前入口仅开放开发池pilot，正式预算须先通过G4。"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import EvolutionSettings  # noqa: E402
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/training_pilot.json")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(PROJECT):
        parser.error("配置文件必须在GPLSACO内")
    config = json.loads(config_path.read_text())
    if config["scope"] != "engineering_development":
        parser.error("此入口尚未开放正式实验；先完成G4和正式配置冻结")
    settings = TrainingSettings(
        **{
            **config["training"],
            "evolution": EvolutionSettings(**config["evolution"]),
            "scope": config["scope"],
        }
    )
    row = [
        v.strip()
        for v in next(
            csv.reader(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu_uuid}",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).splitlines()
            )
        )
    ]
    protocol = WorkerProtocol(
        args.gpu_uuid,
        row[0],
        row[1],
        file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
        dimensions=tuple(n for n, _ in settings.budgets),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    split = PROJECT / "provenance/splits.v1.json"
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        training, validation = {}, {}
        for n, _ in settings.budgets:
            ids = source.record_ids("development", n)
            a, b = (
                config["development_training_per_scale"],
                config["development_validation_per_scale"],
            )
            if len(ids) < a + b:
                raise ValueError("开发池不足；不得借用正式测试数据")
            training[n], validation[n] = ids[:a], ids[a : a + b]
        data = TrainingData(
            source,
            training,
            validation,
            {
                "database_sha256": file_hash(database),
                "split_sha256": file_hash(split),
                "config_sha256": file_hash(config_path),
                "entrypoint_sha256": file_hash(Path(__file__)),
                "selection": "sorted development IDs; disjoint leading training then validation",
                "reference_status": "user_supplied_not_independently_certified",
            },
        )
        run = TrainingRun(
            args.output,
            settings,
            protocol,
            data,
            resume=args.resume,
            event=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True),
        )
        result = run.run(stop_after_tasks=args.stop_after_tasks)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
