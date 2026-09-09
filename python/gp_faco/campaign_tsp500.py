"""TSP500单求解seed进化、两级轻量验证及共享GPU调度的实验声明。"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from gp_faco.campaign import (SEEDS, GPU_BASELINE, NATIVE_BASELINE, baseline_manifest,
    read, source, training_directory, macro_gap, validate_rows)
from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.evolution import Evolution, EvolutionSettings, individual_from_program
from gp_faco.experiment_v2 import BaselineCache, choose_horizon, write_once
from gp_faco.program_ir import Program
from gp_faco.remote import PROJECT
from gp_faco.training import TrainingData, TrainingSettings
from gp_faco.worker import WorkerProtocol

BUILD = "build/v2-tsp500-exact"
DEFAULT_DIRECTORY = PROJECT / "artifacts/v2/round-tsp500-3seed"


def initialize_tsp500(directory, *, engineering=False, seeds=SEEDS, shard_size=32):
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("实验产物必须位于项目内")
    directory.mkdir(parents=True, exist_ok=True)
    if sorted(seeds) != sorted(SEEDS) or shard_size not in (16, 32, 64, 128):
        raise ValueError("需要三个固定GP seed和已测量的种群分片大小")
    config = {
        "version": 2, "scope": "engineering_campaign_only" if engineering else "tsp500_three_seed_v3",
        "seeds": list(seeds), "generations": 2 if engineering else 50,
        "population": 8 if engineering else 128, "instances_per_panel": 2 if engineering else 16,
        "training_dimensions": [500], "validation_dimensions": [500], "test_dimensions": [500, 1000],
        "training_solver_seeds": 1, "numeric_backend": "exact", "build": BUILD,
        "distributed_population": True, "population_shard_size": 4 if engineering else shard_size,
        "gpu_replicas": 2 if engineering else 32, "monitoring_every": 1 if engineering else 5,
        "monitor_iterations": 10 if engineering else 1000,
        "validation_screen_iterations": 10 if engineering else 1000,
        "validation_iterations": 20 if engineering else 5000,
        "validation_finalists": 4, "two_stage_validation": True,
        "test_iterations": 20 if engineering else 5000,
        "source_protocol": "artifacts/v2/protocol",
        "inherited_baseline_directories": [] if engineering else ["artifacts/v2/round-3seed", "artifacts/v2/protocol"],
        "cpu_host": "cuda07", "native_threads": 8, "native_initial_routes": 24,
        "comparison_family": [NATIVE_BASELINE, GPU_BASELINE], "primary_dimension": 500, "report_macro": False,
        "bootstrap_replicates": 100 if engineering else 10000, "statistics_seed": 91001,
        "wall_clock_limit": None, "gpu_resource_limit": None, "gpu_discovery_seconds": 60,
        "infrastructure_retries": 1,
    }
    write_once(directory / "campaign.json", config)
    if not (directory / "panels.json").exists():
        original = read(PROJECT / config["source_protocol"] / "panels.json")
        val = next(p for p in original["validation"] if p["dimension"] == 500)
        declaration = {
            "training": [{**g, "panels": [{**p, "seeds": p["seeds"][:1]}
                          for p in g["panels"] if p["dimension"] == 500]} for g in original["training"]],
            "monitor": [{**p, "seeds": [17]} for p in original["monitor"] if p["dimension"] == 500],
            "validation": [{"dimension": 500, "ids": val["ids"][:32], "seeds": [17]}],
            "validation_final": [{"dimension": 500, "ids": val["ids"][32:96], "seeds": [17, 29, 41]}],
            "test": original["test"], "explain": original["monitor"],
        }
        if engineering:
            with source() as dataset:
                pools = {n: dataset.record_ids("development", n) for n in (500, 1000)}
            rng = random.Random(73001)
            declaration["training"] = [{"generation": g, "panels": [{"dimension": 500,
                "ids": sorted(rng.sample(pools[500][:4], 2)), "seeds": [rng.getrandbits(64)]}]} for g in range(2)]
            for role, left, right, values in (("monitor", 4, 6, [17]), ("validation", 6, 10, [17]),
                                              ("validation_final", 10, 14, [17, 29, 41])):
                declaration[role] = [{"dimension": 500, "ids": pools[500][left:right], "seeds": values}]
            declaration["test"] = [{"dimension": n, "ids": pools[n][14:18], "seeds": [17, 29]} for n in (500, 1000)]
            declaration["explain"] = [{"dimension": n, "ids": pools[n][4:6], "seeds": [17]} for n in (500, 1000)]
        write_once(directory / "panels.json", declaration)
    if not (directory / "frozen.json").exists():
        if engineering:
            selection = {"iterations": 10, "reason": "engineering_only"}
        else:
            origin = PROJECT / "artifacts/v2/round-3seed"
            manifest = read(origin / "calibration/manifest.json")
            rows = [r for path in sorted((origin / "jobs").glob("calibration-n500-*/result.json")) for r in read(path)["rows"]]
            selection = choose_horizon(rows, controllers=list(manifest["controllers"]),
                expected_panels=[p for p in manifest["panels"] if p["dimension"] == 500])
            if selection["iterations"] != 1000:
                raise ValueError("已有TSP500校准与确认的1000迭代方案不符")
        write_once(directory / "frozen.json", {"experiment_version": 3, "numeric_backend": "exact",
            "training_iterations": selection["iterations"], "validation_iterations": config["validation_iterations"],
            "test_iterations": config["test_iterations"], "horizon_selection": selection,
            "training_dimensions": [500], "wall_clock_limit": None})
    return config


def run_training_pool(directory, job):
    from gp_faco.distributed_population import DistributedTrainingRun, PopulationClient

    config, declaration, frozen = [read(directory / name) for name in ("campaign.json", "panels.json", "frozen.json")]
    seed = job["seed"]
    settings = TrainingSettings(
        evolution=EvolutionSettings(population=config["population"], generations=config["generations"],
                                    elites=min(4, config["population"] - 1), feature_spec_id=2),
        evolution_seed=seed, instances_per_panel=config["instances_per_panel"], solver_seeds_per_instance=1,
        validation_seeds=(17,), budgets=((500, 128 * frozen["training_iterations"]),),
        validation_budgets=((500, 128 * config["validation_screen_iterations"]),),
        monitoring_every=config["monitoring_every"], preparation_mode="cached", budget_kind="search_tour_evaluations",
        scope=config["scope"], population_batch_size=config["population"], portable_a5000=True)
    protocol = WorkerProtocol("pool", "NVIDIA RTX A5000", "per_worker", Path(config["build"]).name,
        dimensions=(500,), colonies=config["instances_per_panel"], extension_directory=config["build"], distributed=True)
    output = training_directory(directory, seed)
    cache = BaselineCache(directory / "baselines.sqlite", baseline_manifest(), readonly=True, immutable=True)
    try:
        with source() as dataset:
            training = {500: sorted({name for g in declaration["training"] for p in g["panels"] for name in p["ids"]})}
            data = TrainingData(dataset, training, {500: declaration["validation"][0]["ids"]},
                {"campaign": str(directory.relative_to(PROJECT)), "config": config, "frozen": frozen},
                shared_panels=declaration["training"],
                monitoring_panels=[{**p, "evaluation_limit": 128 * config["monitor_iterations"]} for p in declaration["monitor"]],
                baseline_cache=cache)
            run = DistributedTrainingRun(output, settings, protocol, data,
                resume=(output / "checkpoint.json").exists(),
                worker_factory=lambda p: PopulationClient(directory, seed, p),
                event=lambda r: print(__import__("json").dumps(r, ensure_ascii=False), flush=True))
            report = run.run(through="training")
            if report["status"] not in ("validation", "complete"):
                raise RuntimeError("三次进化之一没有完成全部训练代")
            return {"kind": "training", "status": "completed", "seed": seed,
                "total_fe": report["costs"]["search_tour_evaluations"],
                "solve_seconds": report["costs"]["native_actual_seconds"],
                "generations": len(report["generation_metrics"]), "shortlist": run.state["shortlist"]}
    finally:
        cache.close()


def validation_ranking(directory, seed, results, programs, panel_role):
    rows = [row for result in results for row in result["rows"]]
    validate_rows(rows, methods=[p["program_id"] for p in programs], panels=read(directory / "panels.json")[panel_role])
    by_program = defaultdict(list)
    for row in rows:
        by_program[row["method"]].append(row)
    evolution = Evolution.from_state_dict(load_checkpoint(training_directory(directory, seed) / "checkpoint.json")["evolution"])
    evaluations = {}
    for value in programs:
        program = Program.from_dict(value)
        evaluations[program.identifier] = {"fitness": macro_gap(by_program[program.identifier], dimensions=(500,)),
            "nodes": len(program.opcode), "expression": str(individual_from_program(program, evolution.pset))}
    order = sorted(evaluations, key=lambda k: (evaluations[k]["fitness"], evaluations[k]["nodes"], k))
    return evaluations, order


def select_light_validation(directory, seed, quick_results, final_results, programs):
    run = training_directory(directory, seed)
    config = read(directory / "campaign.json")
    quick_results, final_results = list(quick_results), list(final_results)
    quick = read(run / "validation_screen.json")
    finals = quick["finalists"]
    evaluations, order = validation_ranking(directory, seed, final_results, finals, "validation_final")
    selected = order[0]
    program = next(p for p in finals if p["program_id"] == selected)
    state = load_checkpoint(run / "checkpoint.json")
    evolution = Evolution.from_state_dict(state["evolution"])
    report = load_checkpoint(run / "training_summary.json")
    by_content = {Program.from_dict(p).key: p["program_id"] for p in programs}
    validation_fe = sum(r["total_fe"] for r in quick_results + final_results)
    report.update(status="complete", selected={"program_id": selected, **evaluations[selected]},
        validation_results=evaluations, validation_screen_results=quick["evaluations"],
        champion_validation_curve=[{"generation": i + 1, "program_id": by_content[Program.from_dict(w["program"]).key],
            **quick["evaluations"][by_content[Program.from_dict(w["program"]).key]]} for i, w in enumerate(evolution.winners)],
        validation_costs={"search_tour_evaluations": validation_fe,
            "solve_seconds": sum(r["solve_seconds"] for r in quick_results + final_results),
            "training_fe_fraction": validation_fe / (config["population"] * config["generations"] * config["instances_per_panel"] * 128 * read(directory / "frozen.json")["training_iterations"]),
            "screen_instances": len(read(directory / "panels.json")["validation"][0]["ids"]),
            "finalists": len(finals)})
    save_checkpoint(run / "selected_program.json", {"export_version": 1, "run_id": state["run_id"],
        "scope": config["scope"], "program": program, "selection": report["selected"]})
    save_checkpoint(run / "summary.json", report)
    state.update(phase="complete", selected=report["selected"], validation_results=evaluations,
                 external_validation_costs=report["validation_costs"])
    save_checkpoint(run / "checkpoint.json", state)
    return report
