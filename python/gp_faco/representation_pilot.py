"""两种表示共享的进化和GPU执行；10代预实验与50代完整流程由各自入口限定。"""

import json
import math
import time
from dataclasses import asdict

import numpy as np

from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.campaign import GPU_BASELINE, NATIVE_BASELINE, read, source
from gp_faco.campaign_pool import PopulationScheduler
from gp_faco.checkpoint import atomic_json
from gp_faco.controller import DecisionContexts
from gp_faco.diverse_evolution import DiverseEvolution, DiversitySettings
from gp_faco.experiment_v2 import write_once
from gp_faco.remote import PROJECT, process_receipt

BUILD = "build/v3-gp-representation-exact"
DEFAULT_DIRECTORY = PROJECT / "artifacts/v3/gp-representation-pilot"
REPRESENTATIONS = ("joint_single", "conditional_three")
SEEDS = (1103, 2207, 3313)
FIXED16 = "gpu_fixed_mne16_region0_no_restart"
UNIFORM16 = "gpu_faco_mne16_uniform_no_restart"
BASELINES = (NATIVE_BASELINE, GPU_BASELINE, FIXED16, UNIFORM16)


class PilotPaused(Exception):
    """仅停止CPU协调；已提交的GPU分片继续落盘，恢复按普通任务编号读取。"""


def respect_pause(directory):
    if (directory / "pause.json").exists():
        raise PilotPaused()


def initialize(directory, engineering=False):
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("预实验目录必须位于GPLSACO")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "requests", "runs"):
        (directory / name).mkdir(exist_ok=True)
    config = {
        "version": 3,
        "representation_pilot": True,
        "engineering": engineering,
        "scope": "engineering_campaign_only" if engineering else "representation_pilot_3seed_10gen",
        "representations": list(REPRESENTATIONS),
        "seeds": list(SEEDS),
        "generations": 2 if engineering else 10,
        "population": 8 if engineering else 128,
        "iterations": 10 if engineering else 1000,
        "end_iterations": 20 if engineering else 5000,
        "monitor_every": 1 if engineering else 5,
        "population_shard_size": 4 if engineering else 128,
        "instances_per_panel": 2 if engineering else 16,
        "build": BUILD,
        "numeric_backend": "exact",
        "gpu_replicas": 32,
        "gpu_discovery_seconds": 60,
        "infrastructure_retries": 1,
        "cpu_host": "cuda07",
        "native_threads": 8,
        "native_initial_routes": 24,
        "inherited_baseline_directories": [
            "artifacts/v2/round-tsp500-3seed",
            "artifacts/v2/round-3seed",
            "artifacts/v2/protocol",
        ],
        "wall_clock_limit": None,
        "auto_extend_generations": False,
        "test_dimensions": [],
        "behavior_numeric_backend": "same_CUDA_scorer_as_training",
        "dynamics": asdict(
            DiversitySettings(
                population=8 if engineering else 128, generations=2 if engineering else 10, elites=4
            )
        ),
        "baselines": list(BASELINES),
    }
    write_once(directory / "campaign.json", config)
    if not (directory / "native_hosts.json").exists():
        atomic_json(
            directory / "native_hosts.json",
            read(PROJECT / "artifacts/v2/round-tsp500-3seed/native_hosts.json", ["cuda07"]),
        )
    old = read(PROJECT / "artifacts/v2/round-tsp500-3seed/panels.json")
    with source() as data:
        dev = data.record_ids("development", 500)
    panels = {
        "training": old["training"][: config["generations"]],
        "monitor": old["monitor"],
        "end_development": [{"dimension": 500, "ids": dev[96:128], "seeds": [17, 29, 41]}],
    }
    if engineering:
        panels["training"] = [
            {
                "generation": i,
                "panels": [{"dimension": 500, "ids": dev[:2], "seeds": g["panels"][0]["seeds"]}],
            }
            for i, g in enumerate(panels["training"])
        ]
        panels["monitor"] = [{"dimension": 500, "ids": dev[4:6], "seeds": [17]}]
        panels["end_development"] = [{"dimension": 500, "ids": dev[6:8], "seeds": [17, 29, 41]}]
    write_once(directory / "panels.json", panels)
    # 行为情境始终用正式首训练面板，工程检查不另造一个更容易通过的情境集。
    first = old["training"][0]["panels"][0]
    write_once(
        directory / "contexts_manifest.json",
        {
            "version": 1,
            "ids": first["ids"][:4],
            "seed": first["seeds"][0],
            "mne": [2, 4, 8, 16],
            "region": 0,
            "restart": "none",
            "pre_decision_iterations": [0, 31, 255, 999],
            "source_iterations": 1000,
            "features": "all_12_features_for_all_32_actions",
            "count": 64,
            "total_fe": 2048000,
            "labels_in_inputs": False,
            "comparison": "direct_array_equality",
        },
    )
    if directory != DEFAULT_DIRECTORY and (DEFAULT_DIRECTORY / "contexts.npz").exists():
        if read(directory / "contexts_manifest.json") != read(
            DEFAULT_DIRECTORY / "contexts_manifest.json"
        ):
            raise ValueError("情境采集参数不同，不能复用")
        if not (directory / "contexts.npz").exists():
            (directory / "contexts.npz").write_bytes(
                (DEFAULT_DIRECTORY / "contexts.npz").read_bytes()
            )
            atomic_json(
                directory / "contexts_origin.json",
                {"source": str(DEFAULT_DIRECTORY), "reused": True, "additional_fe": 0},
            )
    return config


def load_contexts(directory):
    with np.load(directory / "contexts.npz") as values:
        return DecisionContexts(values["features"], values["masks"])


def baseline_controller(method):
    if method == GPU_BASELINE:
        return None
    if method == FIXED16:
        return BaselinePolicy(mne_level=3, max_mne_level=3).to_dict()
    if method == UNIFORM16:
        return {"uniform_mne": 16}
    raise ValueError("未知GPU baseline")


def solve_gpu_job(directory, job, session):
    if job.get("role") == "breeding":
        contexts = load_contexts(directory)
        contexts.scorer = lambda c, f, m: session.native.score_controller_cuda(c.to_dict(), f, m)
        config = read(directory / "campaign.json")
        start = time.perf_counter()
        if job.get("evolution") is None:
            evolution = DiverseEvolution(
                job["representation"],
                job["seed"],
                contexts,
                DiversitySettings(**config["dynamics"]),
            )
            evolution.initialize()
        else:
            evolution = DiverseEvolution.from_state_dict(job["evolution"], contexts)
            winner = evolution.ranking()[0]
            evolution.history[-1]["feedback_intervention_changed_actions_of_64"] = (
                contexts.interventions(winner.controller())
            )
            evolution.advance()
        return {
            "status": "completed",
            "kind": "population",
            "evolution": evolution.state_dict(),
            "total_fe": 0,
            "solve_seconds": 0,
            "breeding_wall_seconds": time.perf_counter() - start,
            "behavior_numeric_backend": "same_CUDA_scorer_as_training",
            "device": session.device,
        }
    if job.get("role") == "contexts":
        manifest = read(directory / "contexts_manifest.json")
        pairs = [(name, manifest["seed"]) for name in manifest["ids"]]
        session.set_colonies(4)
        rows, total_fe, seconds = [], 0, 0.0
        with source() as data:
            problems = {name: data.load_instance(name) for name, _ in pairs}
            for level in range(4):
                policy = BaselinePolicy(mne_level=level, max_mne_level=level).to_dict()
                native, timing = session.solve(
                    problems, pairs, 1000, policy, decision_iterations=(1, 32, 256, 1000)
                )
                rows.extend(
                    {
                        **row,
                        "mne": 2 << level,
                        "instance_id": pairs[row["colony"]][0],
                        "pre_decision_iteration": row["iteration"] - 1,
                    }
                    for row in native["decisions"]
                )
                total_fe += native["total_tour_evaluations"]
                seconds += timing["solve_seconds"]
        return {
            "status": "completed",
            "kind": job["kind"],
            "contexts": rows,
            "total_fe": total_fe,
            "solve_seconds": seconds,
            "device": session.device,
        }
    pairs, iterations = job["pairs"], job["iterations"]
    session.set_colonies(len(pairs))
    with source() as data:
        problems = {name: data.load_instance(name) for name, _ in pairs}
        if job["kind"] == "population":
            if job["role"] in ("end_development", "explain") and len(job["controllers"]) == 1:
                points = sorted({
                    p for p in job.get("decision_iterations", (1, 32, 256, 1000, iterations))
                    if p <= iterations
                })
                result, timing = session.solve(
                    problems, pairs, iterations, job["controllers"][0], decision_iterations=points
                )
                result["decisions"] = [r for r in result["decisions"] if r["colony"] < 2]
                native = [result]
            else:
                native, timing = session.solve_population(
                    problems, pairs, iterations, job["controllers"]
                )
            ids = [c["controller_id"] for c in job["controllers"]]
        else:
            result, timing = session.solve(
                problems, pairs, iterations, baseline_controller(job["method"])
            )
            native, ids = [result], [job["method"]]
        # 求解完成后才加载参考值；不把标签路径或最优值传入原生Engine。
        references = {name: data.load_label(name).cost for name in problems}
        members, all_rows = [], []
        for identifier, result in zip(ids, native, strict=True):
            expected = len(pairs) * 128 * iterations
            if result["total_tour_evaluations"] != expected:
                raise ValueError("实际FE不等于预登记面板预算")
            rows = []
            for (name, seed), item in zip(pairs, result["items"], strict=True):
                if not item["has_incumbent"] or not np.isfinite(item["cost"]):
                    raise ValueError("未返回完整终局解")
                # 与既有外部fitness相同的坐标距离与fsum；不以LS增量累计成本打破理论同分。
                problem = problems[name]
                tour = item["tour"]
                cost = math.fsum(problem.distance(tour[i - 1], tour[i]) for i in range(len(tour)))
                rows.append(
                    {
                        "method": identifier,
                        "dimension": problem.dimension,
                        "instance_id": name,
                        "seed": seed,
                        "iterations": iterations,
                        "cost": cost,
                        "gap_percent": 100 * (cost / references[name] - 1),
                        "tour": item["tour"],
                        "status": "completed",
                    }
                )
            members.append(
                {
                    "controller_id": identifier,
                    "rows": rows,
                    "total_fe": expected,
                    "native_result": result,
                }
            )
            all_rows.extend(rows)
        return {
            "status": "completed",
            "kind": job["kind"],
            "members": members,
            "rows": [*job.get("inherited_rows", []), *all_rows]
            if job["kind"] != "population"
            else [],
            "total_fe": sum(m["total_fe"] for m in members),
            "solve_seconds": timing["solve_seconds"],
            "registration_seconds": timing.get("registration_seconds", 0),
            "device": session.device,
        }


def pairs_for(panel):
    return [(name, seed) for name in panel["ids"] for seed in panel["seeds"]]


def submit_evaluation(directory, run_id, role, generation, controllers, pairs, iterations):
    respect_pause(directory)
    config = read(directory / "campaign.json")
    keys = []
    for pi in range(0, len(pairs), 32):
        for ci in range(0, len(controllers), config["population_shard_size"]):
            key = f"population-{run_id}-{role}-g{generation:02d}-p{pi:03d}-c{ci:03d}"
            job = {
                "id": key,
                "kind": "population",
                "resource": "gpu",
                "pilot": True,
                "run_id": run_id,
                "role": role,
                "generation": generation,
                "shard_index": ci,
                "controllers": controllers[ci : ci + config["population_shard_size"]],
                "dimension": 500,
                "pairs": pairs[pi : pi + 32],
                "iterations": iterations,
            }
            write_once(directory / "requests" / f"{key}.json", job)
            keys.append(key)
    return keys


def collect_evaluation(directory, keys, controllers):
    results = {}
    while len(results) < len(keys):
        respect_pause(directory)
        if (directory / "campaign_abort.json").exists():
            raise RuntimeError("GPU池依赖任务失败；保留已完成分片等待修复")
        for key in keys:
            if key not in results:
                value = read(directory / "jobs" / key / "result.json")
                if value:
                    if value["status"] != "completed":
                        raise ValueError("评价分片没有完成")
                    results[key] = value
        if len(results) < len(keys):
            time.sleep(0.25)
    rows = {c["controller_id"]: [] for c in controllers}
    for result in results.values():
        for member in result["members"]:
            rows[member["controller_id"]].extend(member["rows"])
    return rows, {
        "jobs": keys,
        "total_fe": sum(r["total_fe"] for r in results.values()),
        "gpu_seconds": sum(r["solve_seconds"] for r in results.values()),
    }


def breed_on_gpu(directory, run_id, representation, seed, contexts, evolution=None):
    respect_pause(directory)
    generation = 0 if evolution is None else evolution.generation
    key = f"population-{run_id}-breed-after-g{generation:02d}"
    request = directory / "requests" / f"{key}.json"
    if not request.exists():
        write_once(
            request,
            {
                "id": key,
                "kind": "population",
                "resource": "gpu",
                "pilot": True,
                "role": "breeding",
                "run_id": run_id,
                "generation": generation,
                "representation": representation,
                "seed": seed,
                "evolution": None if evolution is None else evolution.state_dict(),
            },
        )
    while True:
        respect_pause(directory)
        if (directory / "campaign_abort.json").exists():
            raise RuntimeError("GPU繁殖任务失败；不改变多样性约束")
        result = read(directory / "jobs" / key / "result.json")
        if result:
            return DiverseEvolution.from_state_dict(result["evolution"], contexts)
        time.sleep(0.25)


def run_training(directory, job):
    config, panels = read(directory / "campaign.json"), read(directory / "panels.json")
    run_id = f"{job['representation']}-s{job['seed']}"
    output = directory / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    contexts = load_contexts(directory)
    checkpoint = output / "checkpoint.json"
    evolution = (
        DiverseEvolution.from_state_dict(read(checkpoint), contexts)
        if checkpoint.exists()
        else DiverseEvolution(
            job["representation"], job["seed"], contexts, DiversitySettings(**config["dynamics"])
        )
    )
    if not evolution.population:
        evolution = breed_on_gpu(directory, run_id, job["representation"], job["seed"], contexts)
        atomic_json(checkpoint, evolution.state_dict())
    while True:
        generation = evolution.generation
        if evolution.population[0].fitness is None:
            controllers = [p.controller().to_dict() for p in evolution.population]
            panel = panels["training"][generation - 1]["panels"][0]
            started = time.time()
            keys = submit_evaluation(
                directory,
                run_id,
                "training",
                generation,
                controllers,
                pairs_for(panel),
                config["iterations"],
            )
            rows, costs = collect_evaluation(directory, keys, controllers)
            evolution.assign(
                [
                    float(np.mean([r["gap_percent"] for r in rows[c["controller_id"]]]))
                    for c in controllers
                ]
            )
            winner = evolution.ranking()[0]
            metric = {
                "generation": generation,
                "training_gap_percent": winner.fitness,
                "population_mean_gap_percent": float(
                    np.mean([p.fitness for p in evolution.population])
                ),
                "winner": winner.controller().to_dict(),
                "expressions": winner.expressions(),
                "population_fitness": [p.fitness for p in evolution.population],
                "feedback_intervention_changed_actions_of_64": None,
                "diagnostics": evolution.metrics(),
                "costs": costs,
                "evaluation_wait_wall_seconds": time.time() - started,
            }
            evolution.history.append(metric)
            atomic_json(output / f"generation-{generation:02d}.json", metric)
            atomic_json(checkpoint, evolution.state_dict())
        metric = evolution.history[-1]
        if generation % config["monitor_every"] == 0 and "monitor" not in metric:
            controllers = [metric["winner"]]
            keys = submit_evaluation(
                directory,
                run_id,
                "monitor",
                generation,
                controllers,
                pairs_for(panels["monitor"][0]),
                config["iterations"],
            )
            rows, costs = collect_evaluation(directory, keys, controllers)
            metric["monitor"] = {
                "gap_percent": float(
                    np.mean([r["gap_percent"] for r in rows[controllers[0]["controller_id"]]])
                ),
                "costs": costs,
            }
            atomic_json(output / f"generation-{generation:02d}.json", metric)
            atomic_json(checkpoint, evolution.state_dict())
        atomic_json(
            output / "progress.json",
            {
                "run_id": run_id,
                "status": "training",
                "completed_generations": len(evolution.history),
                "history": evolution.history,
            },
        )
        finished = evolution.generation >= evolution.settings.generations
        evolution = breed_on_gpu(
            directory, run_id, job["representation"], job["seed"], contexts, evolution
        )
        atomic_json(output / f"generation-{generation:02d}.json", evolution.history[-1])
        atomic_json(checkpoint, evolution.state_dict())
        if finished:
            break
        # 繁殖补齐反馈干预后再发布，报告不再暂时显示空值。
        atomic_json(output / "progress.json", {
            "run_id": run_id, "status": "training",
            "completed_generations": len(evolution.history), "history": evolution.history,
        })
    if config.get("representation_campaign"):
        from gp_faco.representation_campaign import finish_training

        return finish_training(output, evolution)
    controllers = [evolution.history[-1]["winner"]]
    keys = submit_evaluation(
        directory,
        run_id,
        "end_development",
        evolution.generation,
        controllers,
        pairs_for(panels["end_development"][0]),
        config["end_iterations"],
    )
    rows, costs = collect_evaluation(directory, keys, controllers)
    result = {
        "status": "completed",
        "run_id": run_id,
        "representation": job["representation"],
        "seed": job["seed"],
        "generations": evolution.generation,
        "history": evolution.history,
        "end_development": {
            "gap_percent": float(
                np.mean([r["gap_percent"] for r in rows[controllers[0]["controller_id"]]])
            ),
            "costs": costs,
            "rows": rows[controllers[0]["controller_id"]],
        },
        "total_fe": sum(m["costs"]["total_fe"] for m in evolution.history),
        "test_performed": False,
    }
    atomic_json(output / "summary.json", result)
    atomic_json(output / "progress.json", {**result, "completed_generations": evolution.generation})
    return result


class PilotScheduler(PopulationScheduler):
    def reconcile(self):
        for key in list(self.active):
            runtime = read(self.directory / "jobs" / key / "runtime.json", {})
            if runtime.get("status") == "paused":
                self.active.pop(key)
                self.states[key] = (
                    "paused" if (self.directory / "pause.json").exists() else "pending"
                )
        return super().reconcile()

    def advance(self):
        contexts = self.directory / "contexts.npz"
        if not contexts.exists():
            self.stage = "collecting_training_contexts"
            key = self.add("baseline_gpu", "decision-contexts", pilot=True, role="contexts")
            if self.states[key] != "completed":
                return
            result = read(self.directory / "jobs" / key / "result.json")
            if result["total_fe"] != 2048000 or len(result["contexts"]) != 64:
                raise ValueError("行为情境数量或一次性FE不符")
            rows = result["contexts"]
            features = np.asarray([r["features"] for r in rows], dtype=np.float32).transpose(
                1, 0, 2
            )
            masks = np.asarray([r["legal_mask"] for r in rows], dtype=np.uint32)
            DecisionContexts(features, masks)
            with contexts.with_suffix(".tmp").open("wb") as handle:
                np.savez(handle, features=features, masks=masks)
            contexts.with_suffix(".tmp").replace(contexts)
        self.stage = "baselines_and_training"
        inherited = self.inherited_baselines()
        groups = [
            (f"train-g{i + 1:02d}", g["panels"][0], self.config["iterations"])
            for i, g in enumerate(self.declaration["training"])
        ]
        groups += [
            ("monitor", self.declaration["monitor"][0], self.config["iterations"]),
            ("end-dev", self.declaration["end_development"][0], self.config["end_iterations"]),
        ]
        required = []
        for group, panel, iterations in groups:
            pairs = pairs_for(panel)
            for method in BASELINES:
                for start in range(0, len(pairs), 32):
                    present, missing = [], []
                    for name, seed in pairs[start : start + 32]:
                        row = inherited.get((method, 500, name, seed, iterations))
                        (present if row else missing).append(row if row else (name, seed))
                    kind = "baseline_native" if method == NATIVE_BASELINE else "baseline_gpu"
                    key = self.add(
                        kind,
                        f"{group}-{method}-p{start:03d}",
                        pilot=True,
                        role="baseline",
                        method=method,
                        group=group,
                        dimension=500,
                        pairs=missing,
                        iterations=iterations,
                        inherited_rows=present,
                    )
                    if group != "end-dev":
                        required.append(key)
        if not self.finished(required):
            return
        training = [
            self.add(
                "training", f"{representation}-s{seed}", representation=representation, seed=seed
            )
            for representation in REPRESENTATIONS
            for seed in SEEDS
        ]
        if self.finished(training) and self.finished(
            self.group(("baseline_gpu", "baseline_native"))
        ):
            self.stage = "complete"
            atomic_json(
                self.directory / "stop_workers.json", {"reason": "10_generation_pilot_complete"}
            )

    def run(self):
        try:
            while True:
                self.reconcile()
                self.advance()
                if any(v == "failed" for v in self.states.values()):
                    self.stage = "failed"
                if (
                    self.stage not in ("complete", "failed")
                    and not (self.directory / "pause.json").exists()
                ):
                    self.dispatch()
                status = self.status()
                atomic_json(self.directory / "status.json", status)
                if time.monotonic() - self._last_report >= 60 or self.stage in (
                    "complete",
                    "failed",
                ):
                    write_report(self.directory)
                    print(json.dumps(status), flush=True)
                    self._last_report = time.monotonic()
                if self.stage == "complete":
                    return status
                if self.stage == "failed" and not self.active:
                    raise RuntimeError("训练或评价任务失败，已保留证据并停止派单")
                if (self.directory / "pause.json").exists() and not self.active:
                    self.stage = "paused"
                    atomic_json(self.directory / "stop_workers.json", {"reason": "pilot_paused"})
                    atomic_json(self.directory / "status.json", self.status())
                    return self.status()
                time.sleep(self.poll_seconds)
        except Exception as error:
            self.stage = "failed"
            atomic_json(self.directory / "coordinator_error.json", {"error": str(error)})
            atomic_json(self.directory / "campaign_abort.json", {"error": str(error)})
            atomic_json(self.directory / "status.json", self.status())
            raise
        finally:
            atomic_json(
                self.directory / "coordinator.json",
                {**process_receipt(), "status": self.stage, "campaign_started": self.started},
            )
            self.lease.close()


def write_report(directory):
    if read(directory / "campaign.json").get("representation_campaign"):
        from gp_faco.representation_campaign_report import publish_report

        return publish_report(directory)
    summaries = [
        read(directory / "runs" / f"{representation}-s{seed}" / "progress.json", {})
        for representation in REPRESENTATIONS
        for seed in SEEDS
    ]
    atomic_json(
        directory / "progress_summary.json",
        {"runs": summaries, "updated_unix": time.time(), "test_performed": False},
    )
    if read(directory / "campaign.json")["engineering"]:
        return
    from gp_faco.representation_report import publish_report

    publish_report(directory, summaries)
