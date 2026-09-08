#!/usr/bin/env python3
"""在正常elapsed/未插桩生产入口核验预算候选；保留全部超限、丢弃和准备成本。"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402
from profile_solver import MeasurementRun, conditions, require  # noqa: E402


def cases(config, n, colonies, names):
    """事前固定覆盖：批量全部32动作＋两树；单实例均衡覆盖32动作，中/长保留两棵树。"""
    by_name = {v["name"]: v for v in conditions()}
    budgets = config["budgets"][f"{n}/{colonies}"]
    if colonies == 32:
        panels = [names]
    else:
        panels = [[name] for name in names]
    for panel_index, panel in enumerate(panels):
        for budget_level in ("short", "medium", "long"):
            if budget_level == "short":
                if colonies == 32:
                    selected = list(by_name.values())
                else:
                    selected = [
                        by_name[
                            f"{'restart' if (panel_index // 4) % 2 else 'keep'}"
                            f"-r{(panel_index + level) % 4}-m{2 << level}"
                        ]
                        for level in range(4)
                    ]
            elif colonies == 32 or panel_index == 0:
                selected = [by_name[name] for name in config["longer_budget_conditions"]]
            else:
                continue
            replicas = [
                [name, seed]
                for name in panel
                for seed in (
                    config["solver_seeds"][:1] if colonies == 1 else config["solver_seeds"]
                )
            ]
            for condition in selected:
                for mode in config["preparation_modes"]:
                    yield {
                        "dimension": n,
                        "colonies": colonies,
                        "replicas": replicas,
                        "condition": condition,
                        "preparation_mode": mode,
                        "budget_level": budget_level,
                        "budget_seconds": budgets[f"{budget_level}_seconds"],
                    }


class CalibrationRun(MeasurementRun):
    def __init__(self, config, output, gpu_uuid, resume, fees):
        super().__init__(config, output, gpu_uuid, resume)
        require(not self.state["fees"] or self.state["fees"] == fees, "校准恢复的冻结费用改变")
        self.state["fees"] = fees
        # 母类源manifest另通过配置锁定本入口；运行副本保证后续修改仍可审计原代码。
        source_copy = output / "runtime/calibrate_budget.py"
        if source_copy.exists():
            require(file_hash(source_copy) == config["entrypoint_sha256"], "校准入口快照不符")
        else:
            source_copy.write_bytes(Path(__file__).read_bytes())
        self.save()

    def measure_shape(self, n, colonies, names, problems):
        self.admit()
        with self.timer("engine_creation"):
            settings = self.native.FixedFacoSettings()
            settings.ants = self.config["ants"]
            engine = self.native.FacoBatchEngine(n, colonies, settings)
        keys = {name: int(coordinate_hash(problem)[:16], 16) for name, problem in problems.items()}
        require(len(set(keys.values())) == len(keys), "64位实例键碰撞")
        registration = {}
        with self.timer("registration"):
            for name in names:
                measured = engine.register_problem(
                    keys[name], np.asarray(problems[name].coordinates, dtype=np.float64)
                )
                registration[name] = {
                    "measured": measured,
                    "phases": engine.preparation_profile(keys[name]),
                }
                frozen = self.state["fees"][name]
                require(
                    frozen["coordinate_sha256"] == coordinate_hash(problems[name]), "费用坐标改变"
                )
                engine.set_preparation_charges(
                    keys[name], frozen["cheap_seconds"], frozen["preparation_seconds"]
                )
        for case in cases(self.config, n, colonies, names):
            group_ids = sorted({v[0] for v in case["replicas"]})
            description = {
                "run_id": self.identity,
                **case,
                "problems": [[name, coordinate_hash(problems[name])] for name in group_ids],
            }
            key = content_hash(description)
            path = self.output / "conditions" / f"{key}.json"
            self.current, self.partial_samples = key, []
            if path.exists():
                record = load_checkpoint(path)
                require(
                    record["key"] == key and record["description"] == description, "任务身份不符"
                )
                if key in self.state["completed"]:
                    require(content_hash(record) == self.state["completed"][key], "已完成任务改变")
            else:
                require(key not in self.state["completed"], "已完成任务原始记录缺失")
                self.admit()
                with self.timer("journal"):
                    save_checkpoint(
                        self.output / "pending.json", {"key": key, "description": description}
                    )
                native_keys = np.asarray([keys[v[0]] for v in case["replicas"]], dtype=np.uint64)
                seeds = np.asarray([v[1] for v in case["replicas"]], dtype=np.uint64)
                with self.timer("native_calls"):
                    # 生产入口保持动态elapsed、默认诊断关闭；每个调用从实例初态开始。
                    result = engine.evaluate_program(
                        native_keys,
                        seeds,
                        budget_seconds=case["budget_seconds"],
                        program=case["condition"]["program"],
                        preparation_mode=case["preparation_mode"],
                        experiment_mask=case["condition"]["mask"],
                    )
                self.partial_samples.append({"result": result})
                with self.timer("verification"):
                    require(
                        "profile" not in result and "scope" not in result, "生产入口意外开启诊断"
                    )
                    require(len(result["items"]) == colonies, "返回成员缺失")
                    errors = []
                    for (name, _), item in zip(case["replicas"], result["items"], strict=True):
                        require(item["has_incumbent"], "预算未提供完整面板的合法初解")
                        require(
                            item["completed_seconds"] <= case["budget_seconds"], "迟到解进入输出"
                        )
                        errors.append(abs(tour_cost(problems[name], item["tour"]) - item["cost"]))
                    require(max(errors) < 1e-8, "原目标独立重算失败")
                self.admit()
                record = {
                    "key": key,
                    "description": description,
                    "pid": os.getpid(),
                    "status": "completed",
                    "result": result,
                    "registration": {name: registration[name] for name in group_ids},
                    "maximum_cost_absolute_error": max(errors),
                }
                with self.timer("journal"):
                    save_checkpoint(path, record)
            self.state["completed"][key] = content_hash(record)
            self.save()
            if len(self.state["completed"]) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "completed_cases": len(self.state["completed"]),
                            "dimension": n,
                            "colonies": colonies,
                        }
                    ),
                    flush=True,
                )
        del engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/budget_calibration_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/gpu/profiling/deadline-v1"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(
        args.config.resolve().is_relative_to(PROJECT)
        and args.output.resolve().is_relative_to(PROJECT),
        "配置和输出必须位于项目内",
    )
    config = json.loads(args.config.read_text())
    config["entrypoint_sha256"] = file_hash(Path(__file__))
    report_path = PROJECT / "docs/reports/profiling_results.json"
    require(file_hash(report_path) == config["profiling_report_sha256"], "成本审计报告改变")
    report = json.loads(report_path.read_text())
    require(config["budgets"] == report["budget_candidates"], "预算与已审计候选不符")
    require(
        config["normal_elapsed"] is True and config["diagnostics"] is False,
        "校准必须使用生产计时入口",
    )
    origin = PROJECT / config["profiling_directory"]
    source = load_checkpoint(origin / "checkpoint.json")
    require(
        source["status"] == "complete" and source["identity"] == config["profiling_identity"],
        "成本来源未完成或身份不符",
    )
    require(
        file_hash(origin / "checkpoint.json") == config["profiling_checkpoint_sha256"],
        "成本来源记录改变",
    )
    run = CalibrationRun(config, args.output.resolve(), args.gpu_uuid, args.resume, source["fees"])
    run.execute()
    print(
        json.dumps(
            {
                "status": run.state["status"],
                "cases": len(run.state["completed"]),
                "identity": run.identity,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
