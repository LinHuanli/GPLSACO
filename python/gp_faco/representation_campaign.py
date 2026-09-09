"""修复后的两种Controller表示：六次完整进化、轻量选模、冻结和迁移测试。"""

import copy
import statistics
from dataclasses import asdict

from gp_faco.campaign import NATIVE_BASELINE, read, source
from gp_faco.checkpoint import atomic_json
from gp_faco.controller import Controller
from gp_faco.diverse_evolution import DiversitySettings
from gp_faco.experiment_v2 import write_once
from gp_faco.remote import PROJECT
from gp_faco.representation_pilot import (
    BASELINES,
    REPRESENTATIONS,
    SEEDS,
    PilotScheduler,
    pairs_for,
)
from gp_faco.representation_pilot import (
    DEFAULT_DIRECTORY as PILOT,
)

DEFAULT_DIRECTORY = PROJECT / "artifacts/v3/representation-50gen"
RUN_IDS = tuple(f"{r}-s{s}" for r in REPRESENTATIONS for s in SEEDS)


def initialize(directory, *, build, shard_size=32, engineering=False):
    directory = directory.resolve()
    if (
        not directory.is_relative_to(PROJECT)
        or directory == PILOT
        or shard_size not in (4, 16, 32, 64, 128)
    ):
        raise ValueError("需要独立项目目录与已测分片大小")
    if not engineering and shard_size == 4:
        raise ValueError("4个体分片仅用于工程流程")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "requests", "runs"):
        (directory / name).mkdir(exist_ok=True)
    config = {
        **read(PILOT / "campaign.json"),
        "representation_pilot": False,
        "representation_campaign": True,
        "scope": "engineering_campaign_only" if engineering else "representation_3seed_50gen",
        "engineering": engineering,
        "build": build,
        "generations": 2 if engineering else 50,
        "population": 8 if engineering else 128,
        "population_shard_size": shard_size,
        "instances_per_panel": 2 if engineering else 16,
        "iterations": 10 if engineering else 1000,
        "end_iterations": 20 if engineering else 5000,
        "monitor_every": 1 if engineering else 5,
        "validation_finalists": 2 if engineering else 4,
        "test_dimensions": [500, 1000],
        "test_iterations": 20 if engineering else 5000,
        "training_dimensions": [500],
        "validation_dimensions": [500],
        "bootstrap_replicates": 100 if engineering else 10000,
        "statistics_seed": 91001,
        "auto_extend_generations": False,
        "dynamics": asdict(
            DiversitySettings(
                population=8 if engineering else 128, generations=2 if engineering else 50
            )
        ),
        "fresh_initialization": True,
        "selection_uses_test": False,
    }
    old = read(PROJECT / "artifacts/v2/round-tsp500-3seed/panels.json")
    panels = copy.deepcopy(
        {
            k: old[k]
            for k in ("training", "monitor", "validation", "validation_final", "test", "explain")
        }
    )
    panels["explain"] = [{**p, "ids": p["ids"][:2], "seeds": [17]} for p in old["explain"]]
    if engineering:
        with source() as data:
            pools = {n: data.record_ids("development", n) for n in (500, 1000)}
        panels["training"] = [
            {
                "generation": i,
                "panels": [
                    {"dimension": 500, "ids": pools[500][:2], "seeds": g["panels"][0]["seeds"]}
                ],
            }
            for i, g in enumerate(old["training"][:2])
        ]
        for role, left, right, seeds in (
            ("monitor", 4, 6, [17]),
            ("validation", 6, 8, [17]),
            ("validation_final", 8, 12, [17, 29, 41]),
        ):
            panels[role] = [{"dimension": 500, "ids": pools[500][left:right], "seeds": seeds}]
        panels["test"] = [
            {"dimension": n, "ids": pools[n][14:16], "seeds": [17, 29]} for n in (500, 1000)
        ]
        panels["explain"] = [
            {"dimension": n, "ids": pools[n][4:6], "seeds": [17]} for n in (500, 1000)
        ]
        config["inherited_baseline_directories"] = []
    write_once(directory / "campaign.json", config)
    write_once(directory / "panels.json", panels)
    write_once(directory / "contexts_manifest.json", read(PILOT / "contexts_manifest.json"))
    if not (directory / "contexts.npz").exists():
        (directory / "contexts.npz").write_bytes((PILOT / "contexts.npz").read_bytes())
        atomic_json(directory / "contexts_origin.json", {"source": str(PILOT), "additional_fe": 0})
    if not (directory / "native_hosts.json").exists():
        atomic_json(directory / "native_hosts.json", read(PILOT / "native_hosts.json"))
    return config


def finish_training(output, evolution):
    result = {
        "status": "completed",
        "run_id": output.name,
        "representation": evolution.representation,
        "seed": evolution.seed,
        "generations": evolution.generation,
        "history": evolution.history,
        "final_population": [p.controller().to_dict() for p in evolution.population],
        "total_fe": sum(m["costs"]["total_fe"] for m in evolution.history),
        "test_performed": False,
    }
    atomic_json(output / "training_summary.json", result)
    atomic_json(
        output / "progress.json",
        {**result, "status": "awaiting_validation", "completed_generations": evolution.generation},
    )
    return result


def candidate_controllers(summary):
    candidates, keys = [], []
    # 直接比较完整Controller IR；仅选模复用同表达式，不跨代跳过训练FE。
    for c in [m["winner"] for m in summary["history"]] + summary["final_population"]:
        key = Controller.from_dict(c).key
        if key not in keys:
            keys.append(key)
            candidates.append(c)
    return candidates


def rank_results(results, controllers, panel, iterations):
    rows = {c["controller_id"]: [] for c in controllers}
    for result in results:
        for member in result["members"]:
            rows[member["controller_id"]].extend(member["rows"])
    expected = sorted(pairs_for(panel))
    scores = {}
    for c in controllers:
        identifier = c["controller_id"]
        values = rows[identifier]
        if sorted((r["instance_id"], r["seed"]) for r in values) != expected or any(
            r["iterations"] != iterations or r["dimension"] != panel["dimension"] for r in values
        ):
            raise ValueError("选模必须收齐规定面板，不能重复或遗漏分片")
        scores[identifier] = {
            "gap_percent": statistics.mean(r["gap_percent"] for r in values),
            "nodes": sum(len(t["opcode"]) for t in c["trees"]),
        }
    order = sorted(scores, key=lambda k: (scores[k]["gap_percent"], scores[k]["nodes"], k))
    return scores, order


def compact_rows(result):
    rows = result.get("rows", []) or [r for m in result.get("members", []) for r in m["rows"]]
    return [{k: v for k, v in r.items() if k != "tour"} for r in rows]


class RepresentationScheduler(PilotScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._baseline_jobs = None

    def inherited_baselines(self):
        if self._inherited is not None:
            return self._inherited
        values = super().inherited_baselines()
        if not self.config["engineering"]:
            # 已完成预实验的四类对照与本轮同一参数；按普通任务编号导入一次。
            for path in (PILOT / "jobs").glob("baseline_*/result.json"):
                job = read(path.parent / "job.json")
                if job.get("role") != "baseline":
                    continue
                for row in read(path).get("rows", []):
                    values[
                        (
                            row["method"],
                            row["dimension"],
                            row["instance_id"],
                            row["seed"],
                            row["iterations"],
                        )
                    ] = row
        return values

    def add_baselines(self, groups, *, test=False):
        inherited = self.inherited_baselines()
        jobs = []
        for group, panel, iterations in groups:
            pairs = pairs_for(panel)
            for method in BASELINES:
                # CPU原生分片小一些，空闲主机可分摊长预算测试。
                size = 8 if method == NATIVE_BASELINE else 32
                for start in range(0, len(pairs), size):
                    present, missing = [], []
                    for name, seed in pairs[start : start + size]:
                        row = inherited.get((method, panel["dimension"], name, seed, iterations))
                        (present if row else missing).append(row if row else (name, seed))
                    kind = "baseline_native" if method == NATIVE_BASELINE else "baseline_gpu"
                    jobs.append(
                        self.add(
                            kind,
                            f"{group}-{method}-p{start:04d}",
                            pilot=True,
                            role="baseline",
                            method=method,
                            group=group,
                            dimension=panel["dimension"],
                            pairs=missing,
                            inherited_rows=present,
                            iterations=iterations,
                            test=test,
                        )
                    )
        return jobs

    def controller_jobs(self, run_id, role, controllers, panels, iterations, generation=0):
        keys = []
        for panel in panels:
            pairs = pairs_for(panel)
            for start in range(0, len(pairs), 32):
                for left in range(0, len(controllers), self.config["population_shard_size"]):
                    keys.append(
                        self.add(
                            "population",
                            f"{run_id}-{role}-n{panel['dimension']}-p{start:04d}-c{left:03d}",
                            pilot=True,
                            role=role,
                            run_id=run_id,
                            dimension=panel["dimension"],
                            pairs=pairs[start : start + 32],
                            controllers=controllers[
                                left : left + self.config["population_shard_size"]
                            ],
                            iterations=iterations,
                            generation=generation,
                            shard_index=left,
                            decision_iterations=[1, 100, 500, 1000, 2500, iterations],
                        )
                    )
        return keys

    def save_result_rows(self, keys, filename):
        path = self.directory / filename
        if not path.exists():
            results = self.results(keys)
            timings, rows = [], []
            for key, result in zip(keys, results, strict=True):
                rows.extend(compact_rows(result))
                job = self.jobs[key]
                controllers = job.get("controllers", [])
                timings.append(
                    {
                        "job": key,
                        "method": job.get("method")
                        or (controllers[0]["controller_id"] if len(controllers) == 1 else None),
                        "dimension": job["dimension"],
                        "computed_pairs": len(job["pairs"]),
                        "inherited_pairs": len(job.get("inherited_rows", [])),
                        "solve_seconds": result["solve_seconds"],
                        "host": result.get("execution", {}).get("host"),
                        "device": result.get("device"),
                    }
                )
            atomic_json(
                path,
                {
                    "rows": rows,
                    "jobs": keys,
                    "timings": timings,
                },
            )

    def advance(self):
        if not (self.directory / "contexts.npz").exists():
            raise ValueError("完整实验需要既有64个训练情境")
        self.stage = "baselines"
        if self._baseline_jobs is None:
            groups = [
                (f"train-g{i + 1:02d}", g["panels"][0], self.config["iterations"])
                for i, g in enumerate(self.declaration["training"])
            ]
            groups += [
                (
                    role,
                    self.declaration[role][0],
                    self.config["iterations"]
                    if role != "validation_final"
                    else self.config["end_iterations"],
                )
                for role in ("monitor", "validation", "validation_final")
            ]
            self._baseline_jobs = self.add_baselines(groups)
        if not self.finished(self._baseline_jobs):
            return
        self.save_result_rows(self._baseline_jobs, "baseline_scores.json")
        self.stage = "training_and_validation"
        for representation in REPRESENTATIONS:
            for seed in SEEDS:
                run_id = f"{representation}-s{seed}"
                output = self.directory / "runs" / run_id
                training = self.add("training", run_id, representation=representation, seed=seed)
                if (
                    self.states[training] != "completed"
                    or (output / "selected_controller.json").exists()
                ):
                    continue
                if run_id not in self._candidates:
                    self._candidates[run_id] = candidate_controllers(
                        read(output / "training_summary.json")
                    )
                controllers = self._candidates[run_id]
                quick = self.controller_jobs(
                    run_id,
                    "validation_quick",
                    controllers,
                    self.declaration["validation"],
                    self.config["iterations"],
                    self.config["generations"] + 1,
                )
                if not self.finished(quick):
                    continue
                screen = read(output / "validation_screen.json")
                if screen is None:
                    scores, order = rank_results(
                        self.results(quick),
                        controllers,
                        self.declaration["validation"][0],
                        self.config["iterations"],
                    )
                    screen = {
                        "scores": scores,
                        "finalists": [
                            next(c for c in controllers if c["controller_id"] == key)
                            for key in order[: self.config["validation_finalists"]]
                        ],
                    }
                    atomic_json(output / "validation_screen.json", screen)
                finalists = screen["finalists"]
                final = self.controller_jobs(
                    run_id,
                    "validation_final",
                    finalists,
                    self.declaration["validation_final"],
                    self.config["end_iterations"],
                    self.config["generations"] + 2,
                )
                if not self.finished(final):
                    continue
                scores, order = rank_results(
                    self.results(final),
                    finalists,
                    self.declaration["validation_final"][0],
                    self.config["end_iterations"],
                )
                controller = next(c for c in finalists if c["controller_id"] == order[0])
                costs = {"total_fe": 0, "gpu_seconds": 0.0}
                for result in self.results(quick + final):
                    costs["total_fe"] += result["total_fe"]
                    costs["gpu_seconds"] += result["solve_seconds"]
                atomic_json(
                    output / "selected_controller.json",
                    {
                        "export_version": 1,
                        "run_id": run_id,
                        "controller": controller,
                        "scores": scores,
                        "selected": scores[order[0]],
                        "costs": costs,
                    },
                )
                self.save_result_rows(quick + final, f"runs/{run_id}/validation_scores.json")
        if not all(
            (self.directory / "runs" / run / "selected_controller.json").exists() for run in RUN_IDS
        ):
            return
        frozen_path = self.directory / "methods_frozen.json"
        if not frozen_path.exists():
            write_once(
                frozen_path,
                {
                    "methods": {
                        run: read(self.directory / "runs" / run / "selected_controller.json")
                        for run in RUN_IDS
                    },
                    "baselines": list(BASELINES),
                    "test_panels": self.declaration["test"],
                    "test_iterations": self.config["test_iterations"],
                    "selection_uses_test": False,
                },
            )
        self.stage = "test_and_explanation"
        frozen = read(frozen_path)
        groups = [
            (f"test-n{p['dimension']}", p, self.config["test_iterations"])
            for p in self.declaration["test"]
        ]
        tests = self.add_baselines(groups, test=True)
        explanations = []
        for run, selected in frozen["methods"].items():
            # 一个方法独立求解与计时，不把多控制器并行摊销当作算法速度。
            controller = {**selected["controller"], "controller_id": run}
            tests += self.controller_jobs(
                run,
                "test",
                [controller],
                self.declaration["test"],
                self.config["test_iterations"],
                100,
            )
            explanations += self.controller_jobs(
                run,
                "explain",
                [controller],
                self.declaration["explain"],
                self.config["test_iterations"],
                101,
            )
        if self.finished(tests):
            self.save_result_rows(tests, "test_scores.json")
            if not (self.directory / "statistics.json").exists():
                from gp_faco.representation_statistics import comparison_statistics

                scores = read(self.directory / "test_scores.json")
                stats = comparison_statistics(scores["rows"], self.config, self.declaration["test"])
                stats["timings"] = scores.get("timings", [])
                atomic_json(self.directory / "statistics.json", stats)
        if self.finished(tests + explanations):
            self.stage = "complete"
            atomic_json(
                self.directory / "stop_workers.json",
                {"reason": "six_50_generation_runs_and_tests_complete"},
            )
