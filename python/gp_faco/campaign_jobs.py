"""独立执行训练、验证、测试和解释分片；完整 GPU 返回先保存，再外部评分。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gp_faco.campaign import (
    GPU_BASELINE,
    NATIVE_BASELINE,
    baseline_manifest,
    read,
    source,
    training_directory,
)
from gp_faco.checkpoint import atomic_json
from gp_faco.data import tour_cost
from gp_faco.evolution import EvolutionSettings
from gp_faco.experiment_v2 import BaselineCache
from gp_faco.gpu_session import GpuSession
from gp_faco.native_baseline import solve_native
from gp_faco.remote import PROJECT
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings
from gp_faco.worker import WorkerProtocol, faco_ants


def run_training(directory, job, gpu, device):
    config, declaration, frozen = (
        read(directory / name) for name in ("campaign.json", "panels.json", "frozen.json")
    )
    seed = job["seed"]
    output = training_directory(directory, seed)
    settings = TrainingSettings(
        evolution=EvolutionSettings(
            population=config["population"],
            generations=config["generations"],
            feature_spec_id=2,
            elites=min(4, config["population"] - 1),
        ),
        evolution_seed=seed,
        instances_per_panel=config["instances_per_panel"],
        budgets=tuple((n, faco_ants(n) * frozen["training_iterations"]) for n in (500, 1000)),
        validation_budgets=tuple(
            (n, faco_ants(n) * config["validation_iterations"]) for n in (500, 1000)
        ),
        monitoring_every=config["monitoring_every"],
        preparation_mode="cached",
        budget_kind="search_tour_evaluations",
        scope=config["scope"],
        population_batch_size=config["population"],
        portable_a5000=True,
    )
    protocol = WorkerProtocol(
        gpu,
        "NVIDIA RTX A5000",
        device["driver"],
        Path(config["build"]).name,
        colonies=config["instances_per_panel"] * 2,
        extension_directory=config["build"],
    )
    cache = BaselineCache(
        directory / "baselines.sqlite", baseline_manifest(), readonly=True, immutable=True
    )
    try:
        with source() as dataset:
            training = {n: dataset.record_ids("train", n) for n in (500, 1000)}
            if config["scope"] == "engineering_campaign_only":
                training = {
                    n: sorted(
                        {
                            name
                            for g in declaration["training"]
                            for p in g["panels"]
                            if p["dimension"] == n
                            for name in p["ids"]
                        }
                    )
                    for n in (500, 1000)
                }
            data = TrainingData(
                dataset,
                training,
                {p["dimension"]: p["ids"] for p in declaration["validation"]},
                {
                    "campaign": str(directory.relative_to(PROJECT)),
                    "config": config,
                    "frozen": frozen,
                },
                shared_panels=declaration["training"],
                monitoring_panels=[
                    {
                        **p,
                        "evaluation_limit": faco_ants(p["dimension"])
                        * config["validation_iterations"],
                    }
                    for p in declaration["monitor"]
                ],
                baseline_cache=cache,
            )
            run = TrainingRun(
                output,
                settings,
                protocol,
                data,
                resume=(output / "checkpoint.json").exists(),
                event=lambda r: print(json.dumps(r, ensure_ascii=False), flush=True),
            )
            result = run.run(through="training")
            if result["status"] not in ("validation", "complete"):
                raise RuntimeError("没有完成所有训练代")
            return {
                "kind": "training",
                "seed": seed,
                "status": "completed",
                "total_fe": result["costs"]["search_tour_evaluations"],
                "solve_seconds": result["costs"]["native_actual_seconds"],
                "generations": len(result["generation_metrics"]),
                "shortlist": run.state["shortlist"],
                "training_summary": str((output / "training_summary.json").relative_to(PROJECT)),
            }
    finally:
        cache.close()


def solve_job(directory, job, gpu=None, *, gpu_session=None):
    config = read(directory / "campaign.json")
    if job["kind"] in ("test_gpu", "test_native", "explain") and not read(
        directory / "methods_frozen.json"
    ):
        raise ValueError("三个程序冻结前禁止读取测试标签或执行测试")
    output = directory / "jobs" / job["id"]
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    n, pairs, iterations = job["dimension"], job["pairs"], job["iterations"]
    session = gpu_session
    if session is not None:
        session.set_colonies(len(pairs))
    with source() as dataset:
        problems = {name: dataset.load_instance(name) for name, _ in pairs}
        references = {name: dataset.load_label(name).cost for name in problems}
        rows, decisions = list(job.get("inherited_rows", [])), []
        total_fe, solve_seconds, preparation_seconds, scoring_seconds = 0, 0.0, 0.0, 0.0
        device = None
        if job["kind"] in ("baseline_native", "test_native"):
            if os.cpu_count() != config["native_initial_routes"]:
                raise ValueError("原始 FACO 必须保留实际 24 条初始路线")
            for name, seed in pairs:
                path = parts / f"instance-{problems[name].numeric_id}-seed-{seed}"
                result = solve_native(problems[name], seed, iterations, path)
                if result["initial_route_count"] != config["native_initial_routes"]:
                    raise ValueError("原始 FACO 初始化条数发生变化")
                score_path = path / "external_score.json"
                row = read(score_path)
                if row is None:
                    before = time.perf_counter()
                    cost = tour_cost(problems[name], result["tour"])
                    row = {
                        "method": NATIVE_BASELINE,
                        "dimension": n,
                        "instance_id": name,
                        "seed": seed,
                        "iterations": iterations,
                        "cost": cost,
                        "gap_percent": 100 * (cost / references[name] - 1),
                        "seconds": result["total_seconds"],
                        "tour": result["tour"],
                        "status": "completed",
                        "scoring_seconds": time.perf_counter() - before,
                    }
                    atomic_json(score_path, row)
                rows.append(row)
                total_fe += faco_ants(n) * iterations
                solve_seconds += result["total_seconds"]
                scoring_seconds += row["scoring_seconds"]
        else:
            if job["kind"] == "baseline_gpu":
                groups = [(GPU_BASELINE, [None], [GPU_BASELINE])]
            elif job["kind"].startswith("validation"):
                groups = [
                    ("population", job["programs"], [p["program_id"] for p in job["programs"]])
                ]
            else:
                frozen = read(directory / "methods_frozen.json")
                if frozen is None:
                    raise ValueError("三个最终程序冻结前不能运行测试或解释")
                controllers = [(k, [v["program"]], [k]) for k, v in frozen["methods"].items()]
                if job["kind"] == "test_gpu":
                    controllers.append((GPU_BASELINE, [None], [GPU_BASELINE]))
                offset = job.get("order_rotation", 0) % len(controllers)
                groups = controllers[offset:] + controllers[:offset]
            try:
                for group_id, programs, methods in groups:
                    raw_path, scored_path = (
                        parts / f"{group_id}.raw.json",
                        parts / f"{group_id}.json",
                    )
                    scored = read(scored_path)
                    if scored is None:
                        raw = read(raw_path)
                        if raw is None:
                            if session is None:
                                before = time.perf_counter()
                                session = GpuSession(
                                    gpu,
                                    "exact",
                                    colonies=len(pairs),
                                    extension_directory=config["build"],
                                )
                                preparation_seconds += time.perf_counter() - before
                                if getattr(session.native, "campaign_interface_version", 0) != 1:
                                    raise ValueError("需要通过检查的 campaign 构建")
                            device = session.device
                            if job["kind"].startswith("validation"):
                                native, timing = session.solve_population(
                                    problems, pairs, iterations, programs
                                )
                            else:
                                points = []
                                if job["kind"] == "explain":
                                    points = sorted(
                                        {
                                            p
                                            for p in (1, 100, 500, 1000, 2500, iterations)
                                            if p <= iterations
                                        }
                                    )
                                result, timing = session.solve(
                                    problems,
                                    pairs,
                                    iterations,
                                    programs[0],
                                    decision_iterations=points,
                                )
                                native = [result]
                            raw = {"native": native, "timing": timing, "device": device}
                            atomic_json(raw_path, raw)
                        before = time.perf_counter()
                        member_rows, traces = [], []
                        fe = 0
                        for method, native in zip(methods, raw["native"], strict=True):
                            expected_fe = len(pairs) * faco_ants(n) * iterations
                            if native["total_tour_evaluations"] != expected_fe:
                                raise ValueError("GPU 返回没有完成全部预定 FE")
                            fe += expected_fe
                            for (name, seed), item in zip(pairs, native["items"], strict=True):
                                cost = tour_cost(problems[name], item["tour"])
                                row = {
                                    "method": method,
                                    "dimension": n,
                                    "instance_id": name,
                                    "seed": seed,
                                    "iterations": iterations,
                                    "cost": cost,
                                    "gap_percent": 100 * (cost / references[name] - 1),
                                    "seconds": raw["timing"]["solve_seconds"]
                                    / len(pairs)
                                    / len(programs),
                                    "status": "completed",
                                }
                                if job["kind"] == "baseline_gpu":
                                    row["tour"] = item["tour"]
                                member_rows.append(row)
                            for trace in native.get("decisions", []):
                                name, seed = pairs[trace["colony"]]
                                traces.append(
                                    {
                                        **trace,
                                        "method": method,
                                        "dimension": n,
                                        "instance_id": name,
                                        "seed": seed,
                                    }
                                )
                        scored = {
                            "rows": member_rows,
                            "decisions": traces,
                            "total_fe": fe,
                            "timing": raw["timing"],
                            "device": raw["device"],
                            "scoring_seconds": time.perf_counter() - before,
                        }
                        atomic_json(scored_path, scored)
                    rows.extend(scored["rows"])
                    decisions.extend(scored["decisions"])
                    total_fe += scored["total_fe"]
                    solve_seconds += scored["timing"]["solve_seconds"]
                    preparation_seconds += scored["timing"]["registration_seconds"]
                    scoring_seconds += scored["scoring_seconds"]
                    device = scored["device"]
            finally:
                if session and gpu_session is None:
                    session.close()
    result = {
        "status": "completed",
        "kind": job["kind"],
        "rows": rows,
        "total_fe": total_fe,
        "solve_seconds": solve_seconds,
        "preparation_seconds": preparation_seconds,
        "scoring_seconds": scoring_seconds,
        "device": device,
    }
    if decisions:
        result["decisions"] = decisions
    return result
