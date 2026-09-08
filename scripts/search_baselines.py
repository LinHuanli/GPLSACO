#!/usr/bin/env python3
"""只用development池运行预登记基线配置族；按FE停止，完整保存任务及恢复身份。"""

import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.configuration_search import ConfigurationRun, SearchData, SearchSettings  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, file_hash  # noqa: E402


def policy_grid(spec):
    """由预登记笛卡尔积展开全部配置；不读取数据或根据结果追加参数。"""
    output = []
    static = spec["static"]
    for level, region in product(spec["mne_levels"], spec["regions"]):
        variants = [{"restart_mode": "none"}]
        variants.extend(
            {"restart_mode": "periodic", "restart_period": value} for value in static["periods"]
        )
        variants.extend(
            {"restart_mode": "bernoulli", "restart_probability": value}
            for value in static["probabilities"]
        )
        output.extend(
            BaselinePolicy(mne_level=level, max_mne_level=level, region=region, **v)
            for v in variants
        )
    rule = spec["rule"]
    restarts = [(0, 0), *product(rule["restart_thresholds"], rule["cooldowns"])]
    for base, maximum, region in product(spec["mne_levels"], spec["mne_levels"], spec["regions"]):
        if maximum < base:
            continue
        for step in [0] if maximum == base else rule["stagnation_steps"]:
            output.extend(
                BaselinePolicy(
                    kind="rule",
                    mne_level=base,
                    max_mne_level=maximum,
                    region=region,
                    stagnation_step=step,
                    restart_stagnation=threshold,
                    restart_cooldown=cooldown,
                )
                for threshold, cooldown in restarts
            )
    return tuple(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    if not args.config.resolve().is_relative_to(PROJECT):
        parser.error("配置必须位于GPLSACO内")
    config = json.loads(args.config.read_text())
    if config["scope"] not in (
        "development_calibration",
        "development_tuning",
        "engineering_development",
    ):
        parser.error("本入口只允许development数据，不能读取正式测试成绩")
    settings = SearchSettings(**config["search"])
    if ("policies" in config) == ("policy_grid" in config):
        parser.error("配置须在显式policies与policy_grid中二选一")
    policies = (
        tuple(BaselinePolicy.from_dict(value) for value in config["policies"])
        if "policies" in config
        else policy_grid(config["policy_grid"])
    )
    gpu = [
        value.strip()
        for value in next(
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
        gpu[0],
        gpu[1],
        file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
        dimensions=tuple(config["dimensions"]),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        pools = {}
        for role, selection in config["data_pools"].items():
            offset, count = selection["offset"], selection["count"]
            if type(offset) is not int or offset < 0 or type(count) is not int or count < 1:
                raise ValueError("development成员区间无效")
            pools[role] = {}
            for n in protocol.dimensions:
                ids = sorted(source.record_ids("development", n))
                if offset + count > len(ids):
                    raise ValueError("development池不足，不能借用测试实例")
                pools[role][n] = ids[offset : offset + count]
        data = SearchData(
            source,
            pools,
            {
                "database_sha256": file_hash(database),
                "split_sha256": file_hash(PROJECT / "provenance/splits.v1.json"),
                "config_sha256": file_hash(args.config.resolve()),
                "config_path": str(args.config.resolve().relative_to(PROJECT)),
                "entrypoint_sha256": file_hash(Path(__file__)),
                "selection": "explicit disjoint ranges of sorted development IDs",
                "reference_status": "user_supplied_not_independently_certified",
                "scope": config["scope"],
            },
        )
        run = ConfigurationRun(
            args.output,
            settings,
            protocol,
            policies,
            data,
            resume=args.resume,
            event=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True),
        )
        result = run.run(stop_after_tasks=args.stop_after_tasks)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "run_id", "purpose", "costs", "selected", "completed_records")
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
