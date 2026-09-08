#!/usr/bin/env python3
"""冻结E1训练依赖与测试成员身份，并执行完整128×50训练；不开放测试求解入口。"""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.configuration_search import gpu_boundary  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import EvolutionSettings  # noqa: E402
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def training_settings(config, condition, seed):
    require(
        condition in config["conditions"] and seed in config["evolution_seeds"],
        "训练条件或演化seed未预登记",
    )
    return TrainingSettings(
        **config["training"],
        evolution=EvolutionSettings(
            **config["evolution"], no_feedback=condition == "GP-NoFeedback"
        ),
        evolution_seed=seed,
        scope="E1_formal_training",
    )


def data_members(source, config):
    """只枚举已冻结split的成员ID；freeze阶段不读取坐标或标签。"""
    result, seen = {}, set()
    for n in config["dimensions"]:
        result[str(n)] = {}
        for role in ("train", "validation", "test"):
            ids = sorted(source.record_ids(role, n))
            require(len(ids) == config["data"]["expected_counts"][str(n)][role], "split成员数改变")
            require(len(ids) == len(set(ids)) and not seen.intersection(ids), "split成员重复或重叠")
            seen.update(ids)
            result[str(n)][role] = ids
    return result


def training_panels(config, members):
    """与标准TrainingRun相同的独立面板流；在任何正式fitness前冻结全部50代。"""
    rng, result = random.Random(config["training"]["panel_seed"]), []
    for _ in range(config["evolution"]["generations"]):
        generation = []
        for n in config["dimensions"]:
            ids = sorted(
                rng.sample(members[str(n)]["train"], config["training"]["instances_per_panel"])
            )
            seeds = []
            while len(seeds) < config["training"]["solver_seeds_per_instance"]:
                value = rng.getrandbits(64)
                if value not in seeds:
                    seeds.append(value)
            generation.append({"dimension": n, "ids": ids, "seeds": seeds})
        result.append(generation)
    return result


class RegisteredTrainingRun(TrainingRun):
    """只增加预登记核对和设备边界观察，不改变已验证的GP评价/恢复事务。"""

    def __init__(self, *args, expected_panels, boundary=gpu_boundary, **kwargs):
        self.expected_panels, self._boundary = expected_panels, boundary
        super().__init__(*args, **kwargs)

    def _begin_generation(self):
        super()._begin_generation()
        require(
            self.state["panels"] == self.expected_panels[self.evolution.generation],
            "实际演化面板与预登记随机流不符，尚未提交评价",
        )

    def _before_submit(self):
        observation = self._boundary(self.protocol, self.state["active_worker"]["pid"])
        self.state.setdefault("admission", []).append(
            {"task_key": self.state["pending"]["key"], **observation}
        )
        self._save()
        require(
            observation["foreign_processes"] == 0, "目标GPU出现外来进程；当前任务未提交，保留待办"
        )

    def _operation(self, kind, description, submit, validate):
        def observed_submit(worker):
            original = submit(worker)

            class ObservedFuture:
                def result(inner, timeout=None):
                    outcome = original.result(timeout=timeout)
                    try:
                        observation = self._boundary(
                            self.protocol, self.state["active_worker"]["pid"]
                        )
                    except Exception as error:
                        # 资源查询失败也保留真实返回；不能因此另起求解。
                        observation = {
                            "foreign_processes": None,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    outcome["gpu_boundary_after"] = observation
                    return outcome

            return ObservedFuture()

        return super()._operation(kind, description, observed_submit, validate)


def freeze(args):
    config = json.loads(args.config.read_text())
    require(
        config["protocol_spec_id"] == 1 and config["wall_clock_limit"] is None,
        "未知规格或存在墙钟限额",
    )
    require(
        config["conditions"] == ["GP-Full", "GP-NoFeedback"]
        and config["evolution_seeds"] == [1103, 2207, 3313, 4409, 5519],
        "E1条件或预登记seed改变",
    )
    settings = training_settings(config, "GP-Full", 1103)
    require(
        config["dimensions"] == [500, 1000]
        and config["colonies"] == 32
        and settings.instances_per_panel == 16
        and settings.solver_seeds_per_instance == 2
        and config["maximum_registered_per_dimension"] >= 1056
        and config["gpu_model"] == "NVIDIA RTX A5000"
        and config["driver_version"] == "610.43.02",
        "正式训练形状、容量或硬件配置不符",
    )
    require(
        settings.evolution.population == 128
        and settings.evolution.generations == 50
        and settings.evolution.elites == 4
        and settings.budget_kind == "search_tour_evaluations"
        and settings.budgets == ((500, 4096), (1000, 4096)),
        "完整训练配置或主FE未匹配",
    )
    calibration_path = PROJECT / "docs/reports/baseline_fe_calibration_results.json"
    calibration = json.loads(calibration_path.read_text())
    require(
        calibration["status"] == "passed"
        and calibration["purpose"] == "calibration"
        and calibration["routes"] == 15360
        and calibration["expected_tour_evaluations"] == 66846720
        and calibration["phase_jobs"] == {"search": 480},
        "完整FE校准尚未通过独立审计",
    )
    decision_path = PROJECT / "provenance/fe_budget_v1.json"
    decision = json.loads(decision_path.read_text())
    require(
        decision["calibration_report_sha256"] == file_hash(calibration_path)
        and decision["primary_fe_per_colony"] == 4096
        and decision["additional_fe_per_colony"] == [1024, 16384],
        "次数档尚未冻结",
    )
    baseline = load_checkpoint(args.baseline_run / "manifest.json")
    require(
        baseline["settings"]["purpose"] == "tuning"
        and baseline["settings"]["evaluation_limits"] == [4096]
        and len(baseline["policies"]) == 776,
        "完整基线调优的配置族尚未固定",
    )
    binary = PROJECT / "build/cuda/gp_faco_ext.so"
    require(
        file_hash(binary)
        == baseline["worker_protocol"]["binary_sha256"]
        == calibration["identity"]["binary_sha256"],
        "共同底座二进制改变",
    )
    for name, digest in baseline["sources"].items():
        require(file_hash(PROJECT / name) == digest, "共同Python实现改变")
    require(config["solver"] == baseline["worker_protocol"]["settings"], "共同求解设置不同")
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        members = data_members(source, config)
    panels = training_panels(config, members)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "members.json", members)
    atomic_json(args.output / "training_panels.json", panels)
    plan = {
        "protocol_spec_id": 1,
        "status": "training_dependencies_frozen; baseline selection and test release pending",
        "config": config,
        "config_sha256": file_hash(args.config),
        "config_path": str(args.config.resolve().relative_to(PROJECT)),
        "members_sha256": file_hash(args.output / "members.json"),
        "training_panels_sha256": file_hash(args.output / "training_panels.json"),
        "database_sha256": file_hash(database),
        "split_sha256": file_hash(PROJECT / "provenance/splits.v1.json"),
        "calibration_report_sha256": file_hash(calibration_path),
        "fe_decision_sha256": file_hash(decision_path),
        "baseline_run_id": content_hash(baseline),
        "baseline_config_sha256": baseline["data"]["identity"]["config_sha256"],
        "baseline_config_path": baseline["data"]["identity"]["config_path"],
        "native_binary_sha256": file_hash(binary),
        "entrypoint_sha256": file_hash(Path(__file__)),
        "sources": {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in sorted((PROJECT / "python/gp_faco").glob("*.py"))
        },
        "training_runs": [
            {"condition": condition, "evolution_seed": seed}
            for condition in config["conditions"]
            for seed in config["evolution_seeds"]
        ],
        "training_fitness_evaluations_per_run": 6400,
        "training_native_calls_per_run": 12800,
        "training_search_tour_evaluations_per_run": 1677721600,
        "validation": "all unique generation winners plus final population; "
        "complete fixed validation pool",
        "formal_test_released": False,
    }
    atomic_json(args.output / "plan.json", {**plan, "sha256": content_hash(plan)})
    print(json.dumps({"status": plan["status"], "sha256": content_hash(plan), "training_runs": 10}))


def load_plan(directory):
    plan = json.loads((directory / "plan.json").read_text())
    require(
        plan["sha256"] == content_hash({k: v for k, v in plan.items() if k != "sha256"}),
        "E1冻结计划摘要错误",
    )
    require(plan["entrypoint_sha256"] == file_hash(Path(__file__)), "正式入口源码改变")
    for name, digest in plan["sources"].items():
        require(file_hash(PROJECT / name) == digest, "正式Python源码改变")
    for name in ("members", "training_panels"):
        require(
            file_hash(directory / f"{name}.json") == plan[f"{name}_sha256"], "冻结成员或面板改变"
        )
    require(file_hash(PROJECT / plan["config_path"]) == plan["config_sha256"], "正式配置改变")
    for relative, field in (
        ("provenance/fe_budget_v1.json", "fe_decision_sha256"),
        ("docs/reports/baseline_fe_calibration_results.json", "calibration_report_sha256"),
        (plan["baseline_config_path"], "baseline_config_sha256"),
    ):
        require(file_hash(PROJECT / relative) == plan[field], "训练依赖冻结身份改变")
    require(
        file_hash(PROJECT / "build/cuda/gp_faco_ext.so") == plan["native_binary_sha256"],
        "正式原生二进制改变",
    )
    require(
        file_hash(PROJECT / "artifacts/data/main-index-v1/instances.sqlite")
        == plan["database_sha256"]
        and file_hash(PROJECT / "provenance/splits.v1.json") == plan["split_sha256"],
        "原始split改变",
    )
    members = json.loads((directory / "members.json").read_text())
    panels = json.loads((directory / "training_panels.json").read_text())
    require(panels == training_panels(plan["config"], members), "面板与预登记随机流不符")
    return plan, members, panels


def train(args):
    plan, members, panels = load_plan(args.freeze)
    config = plan["config"]
    settings = training_settings(config, args.condition, args.evolution_seed)
    device = next(
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
    require(
        [v.strip() for v in device] == [config["gpu_model"], config["driver_version"]],
        "正式训练需要冻结同型号GPU/driver",
    )
    protocol = WorkerProtocol(
        args.gpu_uuid,
        config["gpu_model"],
        config["driver_version"],
        plan["native_binary_sha256"],
        dimensions=tuple(config["dimensions"]),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
    )
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        require(data_members(source, config) == members, "冻结split与当前索引成员不一致")
        data = TrainingData(
            source,
            {n: members[str(n)]["train"] for n in config["dimensions"]},
            {n: members[str(n)]["validation"] for n in config["dimensions"]},
            {
                "database_sha256": plan["database_sha256"],
                "split_sha256": plan["split_sha256"],
                "config_sha256": plan["config_sha256"],
                "entrypoint_sha256": plan["entrypoint_sha256"],
                "e1_plan_sha256": plan["sha256"],
                "condition": args.condition,
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
    subparsers = parser.add_subparsers(dest="stage", required=True)
    frozen = subparsers.add_parser("freeze")
    frozen.add_argument("--config", type=Path, default=PROJECT / "configs/e1_protocol_v1.json")
    frozen.add_argument("--baseline-run", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--freeze", type=Path, required=True)
    training.add_argument("--condition", choices=("GP-Full", "GP-NoFeedback"), required=True)
    training.add_argument("--evolution-seed", type=int, required=True)
    training.add_argument("--gpu-uuid", required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(args.output.is_relative_to(PROJECT), "所有产物必须在GPLSACO内")
    if args.stage == "freeze":
        require(args.config.resolve().is_relative_to(PROJECT), "正式配置必须在项目内")
        freeze(args)
    else:
        require(args.freeze.resolve().is_relative_to(PROJECT), "冻结计划必须在项目内")
        train(args)


if __name__ == "__main__":
    main()
