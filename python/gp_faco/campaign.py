"""三次独立进化的固定实验声明、分片结果和统计；不计算完整性哈希。"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict

import numpy as np

from gp_faco.checkpoint import atomic_json, load_checkpoint
from gp_faco.dataset_index import IndexedDataset
from gp_faco.evolution import Evolution, individual_from_program
from gp_faco.experiment_v2 import (
    GPU_BASELINE,
    NATIVE_BASELINE,
    BaselineCache,
    batches,
    replicas,
    write_once,
)
from gp_faco.program_ir import Program
from gp_faco.remote import PROJECT

SEEDS = (1103, 2207, 3313)
BUILD = "build/v2-campaign-exact"


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def source():
    return IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite",
        PROJECT.parent / "Datasets/TSP",
    )


def initialize(directory, *, engineering=False, seeds=SEEDS):
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("全部实验产物必须位于 GPLSACO")
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "scope": "engineering_campaign_only" if engineering else "full_three_seed_v2",
        "seeds": list(seeds),
        "generations": 2 if engineering else 50,
        "population": 8 if engineering else 128,
        "instances_per_panel": 2 if engineering else 16,
        "numeric_backend": "exact",
        "build": BUILD,
        "monitoring_every": 1 if engineering else 5,
        "validation_iterations": 20 if engineering else 5000,
        "test_iterations": 20 if engineering else 5000,
        "source_protocol": "artifacts/v2/protocol",
        "cpu_host": "cuda07",
        "native_threads": 8,
        "native_initial_routes": 24,
        "comparison_family": [NATIVE_BASELINE, GPU_BASELINE],
        "bootstrap_replicates": 10000,
        "statistics_seed": 91001,
        "wall_clock_limit": None,
        "gpu_resource_limit": None,
        "gpu_discovery_seconds": 60,
        "infrastructure_retries": 1,
    }
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(s not in SEEDS for s in seeds):
        raise ValueError("本轮预登记三个 seed 为 1103、2207、3313")
    write_once(directory / "campaign.json", config)
    existing = read(directory / "panels.json")
    if existing is None:
        declaration = read(PROJECT / config["source_protocol"] / "panels.json")
        if not declaration:
            raise ValueError("需要已登记的 v2 面板")
        if engineering:
            rng = random.Random(73001)
            declaration = {"training": [], "validation": [], "test": [], "monitor": []}
            with source() as dataset:
                pools = {n: dataset.record_ids("development", n) for n in (500, 1000)}
            for g in range(2):
                declaration["training"].append(
                    {
                        "generation": g,
                        "panels": [
                            {
                                "dimension": n,
                                "ids": sorted(rng.sample(pools[n][:4], 2)),
                                "seeds": [rng.getrandbits(64), rng.getrandbits(64)],
                            }
                            for n in (500, 1000)
                        ],
                    }
                )
            for role, left, right in (("monitor", 4, 6), ("validation", 6, 10), ("test", 10, 14)):
                declaration[role] = [
                    {"dimension": n, "ids": pools[n][left:right], "seeds": [17, 29]}
                    for n in (500, 1000)
                ]
        write_once(directory / "panels.json", declaration)
    if engineering:
        write_once(
            directory / "frozen.json",
            {
                "experiment_version": 2,
                "numeric_backend": "exact",
                "training_iterations": 10,
                "validation_iterations": 20,
                "test_iterations": 20,
                "scope": "engineering_campaign_only",
            },
        )
    return config


def baseline_manifest():
    return {
        "experiment_version": 2,
        "dataset": "main-index-v1",
        "distance": "continuous_euclidean_fp64",
        "numeric_backend": "exact",
        "ants": "64*ceil(sqrt(n)/16)",
        "beta": 1,
        "retention": 0.5,
        "primary_width": 16,
        "backup_width": 64,
        "ls_width": 20,
        "p_best": 0.1,
        "source_probability": 0.01,
        "mne": 8,
        "gpu_ls_evaluation_limit": 100000,
        "gpu_initialization": "node0_nn_checklist_2opt_100000_identity_fallback",
        "native_initialization": "author_nn_3opt_24_initial_routes_8_threads_edgeguard2",
        "gpu_local_search": "checklist_2opt_identity_zero_v2",
        "native_continuous_adaptation_revision": 2,
    }


def baseline_groups(config, declaration, frozen):
    panels = [
        (p, frozen["training_iterations"]) for g in declaration["training"] for p in g["panels"]
    ] + [
        (p, config.get("monitor_iterations" if role == "monitor" else "validation_screen_iterations", config["validation_iterations"]))
        for role in ("monitor", "validation")
        for p in declaration[role]
    ]
    panels += [(p, config["validation_iterations"]) for p in declaration.get("validation_final", [])]
    width = config.get("gpu_replicas", config["instances_per_panel"] * 2)
    seen = set()
    for panel, iterations in panels:
        for pairs in batches(panel, min(width, len(replicas(panel)))):
            # 普通成员元组只用于避免重新求解相同 baseline；不用于训练 fitness 缓存。
            identity = (panel["dimension"], iterations, tuple(pairs))
            if identity in seen:
                continue
            seen.add(identity)
            yield {"dimension": panel["dimension"], "pairs": pairs, "iterations": iterations}


def row_key(row):
    return row["method"], row["dimension"], row["instance_id"], row["seed"]


def macro_gap(rows, dimensions=(500, 1000)):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not math.isfinite(row["gap_percent"]):
            raise ValueError("失败成员不能从宏平均中删除")
        grouped[row["dimension"]][row["instance_id"]].append(row["gap_percent"])
    if set(grouped) != set(dimensions):
        raise ValueError("宏平均要求所有声明的完整规模")
    return statistics.mean(
        statistics.mean(statistics.mean(v) for v in instances.values())
        for instances in grouped.values()
    )


def expected_members(panels):
    return {(p["dimension"], name, seed) for p in panels for name, seed in replicas(p)}


def validate_rows(rows, *, methods, panels):
    expected = {(m, *v) for m in methods for v in expected_members(panels)}
    found = [row_key(r) for r in rows]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError("评价分片存在重复、缺失或额外成员")
    if any(
        r.get("status") != "completed"
        or not all(math.isfinite(r[k]) for k in ("cost", "gap_percent"))
        or r["cost"] <= 0
        for r in rows
    ):
        raise ValueError("完整评价中存在失败成员")


def training_directory(directory, seed):
    return directory / "training" / f"full-seed{seed}"


def candidate_programs(directory, seed):
    # 完成通知携带小型候选列表，不依赖 NFS 上另一个新目录立即可见。
    job = read(directory / "jobs" / f"training-{seed}" / "result.json", {})
    if "shortlist" in job:
        return job["shortlist"]
    state = load_checkpoint(training_directory(directory, seed) / "checkpoint.json")
    if state["phase"] not in ("validation", "complete"):
        raise ValueError("只有完成全部训练代才能导出候选")
    return state["shortlist"]


def publish_baselines(directory, results):
    config = read(directory / "campaign.json")
    frozen = read(directory / "frozen.json")
    path = directory / "baselines.sqlite"
    cache = BaselineCache(path, baseline_manifest())
    for result in results:
        for row in result["rows"]:
            if (
                cache.get(
                    row["method"],
                    row["dimension"],
                    row["instance_id"],
                    row["seed"],
                    row["iterations"],
                )
                is None
            ):
                cache.put(
                    row["method"],
                    row["dimension"],
                    row["instance_id"],
                    row["seed"],
                    row["iterations"],
                    cost=row["cost"],
                    gap_percent=row["gap_percent"],
                    seconds=row["seconds"],
                    tour=row["tour"],
                )
    cache.commit()
    declaration = read(directory / "panels.json")
    for group in baseline_groups(config, declaration, frozen):
        for method in (GPU_BASELINE, NATIVE_BASELINE):
            for name, seed in group["pairs"]:
                if cache.get(method, group["dimension"], name, seed, group["iterations"]) is None:
                    raise ValueError("配对 baseline 尚未齐备")
    cache.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    cache.connection.execute("PRAGMA journal_mode=DELETE")
    cache.close()
    atomic_json(
        directory / "baselines_ready.json",
        {
            "status": "complete",
            "manifest": baseline_manifest(),
            "training_iterations": frozen["training_iterations"],
            "immutable": True,
        },
    )


def select_validation(directory, seed, results):
    """全部分片齐备才选择；重复调用只重建紧凑报告，不重复任何 GPU 评价。"""
    from gp_faco.checkpoint import save_checkpoint

    run = training_directory(directory, seed)
    results = list(results)
    state = load_checkpoint(run / "checkpoint.json")
    evolution = Evolution.from_state_dict(state["evolution"])
    programs = [Program.from_dict(v) for v in state["shortlist"]]
    rows = [row for result in results for row in result["rows"]]
    validate_rows(
        rows,
        methods=[p.identifier for p in programs],
        panels=read(directory / "panels.json")["validation"],
    )
    by_program = defaultdict(list)
    for row in rows:
        by_program[row["method"]].append(row)
    evaluations = {}
    for program in programs:
        evaluations[program.identifier] = {
            "fitness": macro_gap(by_program[program.identifier]),
            "nodes": len(program.opcode),
            "expression": str(individual_from_program(program, evolution.pset)),
        }
    selected = min(
        evaluations,
        key=lambda k: (evaluations[k]["fitness"], evaluations[k]["nodes"], k),
    )
    program = next(p for p in programs if p.identifier == selected)
    # 原有选择顺序：验证宏 gap、节点数、普通编号；不看测试性能。
    report = load_checkpoint(run / "training_summary.json")
    report.update(
        status="complete",
        validation_results=evaluations,
        selected={"program_id": selected, **evaluations[selected]},
        validation_costs={
            "search_tour_evaluations": sum(r["total_fe"] for r in results),
            "solve_seconds": sum(r["solve_seconds"] for r in results),
            "members": len(rows),
        },
    )
    by_content = {p.key: p.identifier for p in programs}
    report["champion_validation_curve"] = [
        {
            "generation": i + 1,
            "program_id": by_content[Program.from_dict(w["program"]).key],
            **evaluations[by_content[Program.from_dict(w["program"]).key]],
        }
        for i, w in enumerate(evolution.winners)
    ]
    save_checkpoint(
        run / "selected_program.json",
        {
            "export_version": 1,
            "run_id": state["run_id"],
            "scope": report["scope"],
            "program": program.to_dict(),
            "selection": report["selected"],
            "manifest": state["manifest"],
            "costs": report["costs"],
            "validation_costs": report["validation_costs"],
        },
    )
    save_checkpoint(run / "summary.json", report)
    state.update(
        phase="complete",
        selected=report["selected"],
        validation_results=evaluations,
        external_validation_costs=report["validation_costs"],
    )
    save_checkpoint(run / "checkpoint.json", state)
    return report


def freeze_methods(directory):
    config = read(directory / "campaign.json")
    methods = {}
    for seed in config["seeds"]:
        selected = load_checkpoint(training_directory(directory, seed) / "selected_program.json")
        methods[f"gp-{seed}"] = {
            "seed": seed,
            "program": selected["program"],
            "selection": selected["selection"],
        }
    value = {
        "methods": methods,
        "baselines": config["comparison_family"],
        "test_panels": read(directory / "panels.json")["test"],
        "test_iterations": config["test_iterations"],
        "selection_uses_test": False,
    }
    write_once(directory / "methods_frozen.json", value)
    return value


def comparison_statistics(rows, config, panels):
    from scipy.stats import wilcoxon

    methods = [f"gp-{s}" for s in config["seeds"]]
    dimensions = tuple(p["dimension"] for p in panels)
    primary_dimension = config.get("primary_dimension", "macro")
    validate_rows(rows, methods=[*methods, GPU_BASELINE, NATIVE_BASELINE], panels=panels)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dimension"], row["instance_id"])].append(row)
    means = {
        key: {field: statistics.mean(r[field] for r in values) for field in ("cost", "gap_percent")}
        for key, values in grouped.items()
    }
    rng = np.random.default_rng(config["statistics_seed"])
    result = {
        "rows": [],
        "per_evolution_seed": [],
        "bootstrap_replicates": config["bootstrap_replicates"],
    }
    for method in methods:
        for n in dimensions:
            selected = [
                v["gap_percent"] for (m, dim, _), v in means.items() if m == method and dim == n
            ]
            result["per_evolution_seed"].append(
                {"method": method, "dimension": n, "mean_gap_percent": statistics.mean(selected)}
            )
    for baseline in (NATIVE_BASELINE, GPU_BASELINE):
        cubes, relative, baseline_gaps = {}, {}, {}
        for panel in panels:
            n, ids = panel["dimension"], panel["ids"]
            baseline_gaps[n] = np.array([means[baseline, n, i]["gap_percent"] for i in ids])
            cubes[n] = np.array(
                [
                    [
                        means[baseline, n, i]["gap_percent"] - means[m, n, i]["gap_percent"]
                        for i in ids
                    ]
                    for m in methods
                ]
            )
            relative[n] = np.array(
                [
                    [
                        100
                        * (means[baseline, n, i]["cost"] - means[m, n, i]["cost"])
                        / means[baseline, n, i]["cost"]
                        for i in ids
                    ]
                    for m in methods
                ]
            )
        draws = {n: [] for n in dimensions}
        overall = []
        # 先平均求解 seed；全局重采样进化重复，再在每规模内配对重采样实例。
        for _ in range(config["bootstrap_replicates"]):
            repeats = rng.integers(0, len(methods), len(methods))
            values = []
            for n, cube in cubes.items():
                instances = rng.integers(0, cube.shape[1], cube.shape[1])
                value = float(cube[np.ix_(repeats, instances)].mean())
                draws[n].append(value)
                values.append(value)
            overall.append(statistics.mean(values))
        for n in (*dimensions, "macro") if config.get("report_macro", True) else dimensions:
            delta = (
                np.concatenate([cubes[k].mean(axis=0) for k in dimensions])
                if n == "macro"
                else cubes[n].mean(axis=0)
            )
            improvement = float(delta.mean())
            base_gap = float(
                statistics.mean(v.mean() for v in baseline_gaps.values())
                if n == "macro"
                else baseline_gaps[n].mean()
            )
            ci = np.percentile(overall if n == "macro" else draws[n], [2.5, 97.5])
            row = {
                "baseline": baseline,
                "dimension": n,
                "baseline_gap_percent": base_gap,
                "gp_gap_percent": base_gap - improvement,
                "improvement_pp": improvement,
                "ci95_pp": ci.tolist(),
                "relative_cost_improvement_percent": float(
                    statistics.mean(v.mean() for v in relative.values())
                    if n == "macro"
                    else relative[n].mean()
                ),
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
            }
            if n == primary_dimension:
                row["p_value"] = (
                    float(wilcoxon(delta, alternative="two-sided").pvalue)
                    if np.any(delta != 0)
                    else 1.0
                )
            result["rows"].append(row)
    primary = sorted(
        (r for r in result["rows"] if r["dimension"] == primary_dimension),
        key=lambda r: r["p_value"],
    )
    previous = 0.0
    for index, row in enumerate(primary):
        previous = max(previous, min(1.0, row["p_value"] * (len(primary) - index)))
        row["holm_p_value"] = previous
    return result
