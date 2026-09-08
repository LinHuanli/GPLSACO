#!/usr/bin/env python3
"""按用户补充使用任意空闲兼容GPU；原E1演化、FE、成员与面板冻结保持有效。"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.training import TrainingData  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402
from research_e1 import RegisteredTrainingRun, load_plan, require, training_settings  # noqa: E402


def load_amendment(path, plan):
    amendment = json.loads(path.read_text())
    require(
        amendment["sha256"] == content_hash({k: v for k, v in amendment.items() if k != "sha256"})
        and amendment["base_e1_plan_sha256"] == plan["sha256"]
        and amendment["native_binary_sha256"] == plan["native_binary_sha256"],
        "硬件补充协议摘要或原冻结身份不符",
    )
    require(amendment["entrypoint_sha256"] == file_hash(Path(__file__)), "补充入口源码改变")
    return amendment


def available_protocol(plan, amendment, uuid, device):
    row = [v.strip() for v in next(csv.reader(device.splitlines()))]
    require(len(row) == 2, "设备型号/驱动观察不完整")
    report_path = PROJECT / amendment["compatibility_report"]
    require(
        file_hash(report_path) == amendment["compatibility_report_sha256"],
        "硬件兼容验收记录改变",
    )
    report = json.loads(report_path.read_text())
    require(
        report["status"] == "passed"
        and report["gpu_model"] == row[0]
        and report["driver_version"] == row[1]
        and report["native_binary_sha256"] == plan["native_binary_sha256"],
        "当前型号/驱动/共同二进制尚未完成该兼容验收",
    )
    config = plan["config"]
    return WorkerProtocol(
        uuid,
        row[0],
        row[1],
        plan["native_binary_sha256"],
        dimensions=tuple(config["dimensions"]),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
    )


def train(args):
    plan, members, panels = load_plan(args.freeze)
    amendment = load_amendment(args.hardware_amendment, plan)
    settings = training_settings(plan["config"], args.condition, args.evolution_seed)
    device = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu_uuid}",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    )
    protocol = available_protocol(plan, amendment, args.gpu_uuid, device)
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        # 严格核对实际训练/验证成员，标签只由原有外部evaluator读取。
        for n in plan["config"]["dimensions"]:
            for role in ("train", "validation"):
                require(
                    sorted(source.record_ids(role, n)) == members[str(n)][role], "split成员改变"
                )
        data = TrainingData(
            source,
            {n: members[str(n)]["train"] for n in plan["config"]["dimensions"]},
            {n: members[str(n)]["validation"] for n in plan["config"]["dimensions"]},
            {
                "database_sha256": plan["database_sha256"],
                "split_sha256": plan["split_sha256"],
                "config_sha256": plan["config_sha256"],
                "entrypoint_sha256": file_hash(Path(__file__)),
                "frozen_training_entrypoint_sha256": plan["entrypoint_sha256"],
                "e1_plan_sha256": plan["sha256"],
                "condition": args.condition,
                "hardware_amendment_sha256": amendment["sha256"],
                "selection": "complete frozen train and validation splits; "
                "no development or test members",
                "reference_status": "user_supplied_not_independently_certified",
            },
        )
        run = RegisteredTrainingRun(
            args.output,
            settings,
            protocol,
            data,
            resume=args.resume,
            expected_panels=panels,
            event=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True),
        )
        result = run.run()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--hardware-amendment", type=Path, required=True)
    parser.add_argument("--condition", choices=("GP-Full", "GP-NoFeedback"), required=True)
    parser.add_argument("--evolution-seed", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("freeze", "hardware_amendment", "output"):
        value = getattr(args, name).resolve()
        require(value.is_relative_to(PROJECT), "补充入口的文件必须在GPLSACO内")
        setattr(args, name, value)
    train(args)


if __name__ == "__main__":
    main()
