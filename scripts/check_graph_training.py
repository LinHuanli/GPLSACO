#!/usr/bin/env python3
"""单个E3条件的真实8×3开发训练/完整验证/第5任务恢复验收；正式训练须另行冻结。"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.configuration_search import gpu_boundary  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import EvolutionSettings  # noqa: E402
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, file_hash  # noqa: E402


class ObservedGraphTrainingRun(TrainingRun):
    """只增加设备观察，沿用已验证的完整DEAP/准备/求解/恢复状态机。"""

    def _before_submit(self):
        observation = gpu_boundary(self.protocol, self.state["active_worker"]["pid"])
        self.state.setdefault("admission", []).append(
            {"task_key": self.state["pending"]["key"], **observation}
        )
        self._save()
        if observation["foreign_processes"]:
            raise RuntimeError("目标GPU有外来进程，当前任务尚未提交")

    def _operation(self, kind, description, submit, validate):
        def observed_submit(worker):
            original = submit(worker)

            class ObservedFuture:
                def result(inner, timeout=None):
                    outcome = original.result(timeout=timeout)
                    # 即使后续nvidia-smi失败，也已有独立原始返回；不再次提交原任务。
                    atomic_json(
                        self.directory / "raw_returns" / f"{self.state['pending']['key']}.json",
                        outcome,
                    )
                    try:
                        observation = gpu_boundary(
                            self.protocol, self.state["active_worker"]["pid"]
                        )
                    except Exception as error:
                        observation = {
                            "foreign_processes": None,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    outcome["gpu_boundary_after"] = observation
                    return outcome

            return ObservedFuture()

        def checked(outcome):
            if outcome.get("gpu_boundary_after", {}).get("foreign_processes") != 0:
                raise RuntimeError("任务结束时GPU占用未知或有外来进程，保留真实返回等待排查")
            return validate(outcome)

        return super()._operation(kind, description, observed_submit, checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--constraint-mode", choices=("hard", "escape"), required=True)
    parser.add_argument("--prior-kind", choices=("ALPHA", "POPMUSIC"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog, output = args.catalog.resolve(), args.output.resolve()
    if not all(p.is_relative_to(PROJECT) for p in (catalog, output)) or output.exists():
        parser.error("需要工作树内冻结目录及全新工程运行输出")
    base = json.loads((PROJECT / "configs/training_counts_pilot.json").read_text())
    # 32 colonies/32 ants不变；32个已核验实例中每规模8训练+8验证，4 solver seeds。
    # 此8×4面板仅验证工程通路，正式E3仍须使用另行冻结的完整16×2协议。
    settings = TrainingSettings(
        **{
            **base["training"],
            "evolution": EvolutionSettings(**base["evolution"]),
            "scope": "engineering_development",
            "instances_per_panel": 8,
            "solver_seeds_per_instance": 4,
            "validation_seeds": (17, 29, 41, 53),
        }
    )
    model, driver = [
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
        model,
        driver,
        file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
        settings=SolverSettings(**base["solver"]),
        maximum_registered_per_dimension=32,
        constraint_mode=args.constraint_mode,
        graph_prior_kind=args.prior_kind,
        graph_catalog_path=str(catalog.relative_to(PROJECT)),
        graph_catalog_sha256=file_hash(catalog),
    )
    entries = json.loads(catalog.read_text())["entries"]
    training, validation = {}, {}
    with IndexedDataset(args.database, args.dataset_root) as source:
        for n in protocol.dimensions:
            ids = sorted(row["instance_id"] for row in entries if row["dimension"] == n)
            if len(ids) != 16 or not set(ids) <= set(source.record_ids("development", n)):
                raise ValueError("工程面板必须为每规模16个既有开发实例")
            training[n], validation[n] = ids[:8], ids[8:]
        data = TrainingData(
            source,
            training,
            validation,
            {
                "database_sha256": file_hash(args.database),
                "split_sha256": file_hash(PROJECT / "provenance/splits.v1.json"),
                "entrypoint_sha256": file_hash(Path(__file__)),
                "catalog_sha256": file_hash(catalog),
                "base_config_sha256": file_hash(PROJECT / "configs/training_counts_pilot.json"),
                "selection": "sorted 16 development IDs/scale; "
                "first8 training,last8 validation; 4 solver seeds",
                "scope": "engineering only; four separately instantiated training conditions",
            },
        )

        def event(record):
            print(json.dumps(record, ensure_ascii=False), flush=True)

        paused = ObservedGraphTrainingRun(output, settings, protocol, data, event=event).run(
            stop_after_tasks=5
        )
        if paused["status"] != "paused" or paused["solve_jobs"] != 5:
            raise RuntimeError("第5个实际求解任务暂停失败")
        shutil.copyfile(output / "checkpoint.json", output / "paused-checkpoint.json")
        result = ObservedGraphTrainingRun(
            output, settings, protocol, data, event=event, resume=True
        ).run()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["status"] != "complete":
            raise RuntimeError("完整开发训练/验证未完成")


if __name__ == "__main__":
    main()
