"""E3正式规模、共同面板与Static参数族；只处理身份，不读取任何标签。"""

from __future__ import annotations

import random
from itertools import product

from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.candidate_prior import PriorSettings
from gp_faco.evolution import EvolutionSettings
from gp_faco.graph_matching import GraphSettings
from gp_faco.training import TrainingSettings
from gp_faco.worker import SolverSettings

CONDITIONS = (
    ("alpha-Hard", "ALPHA", "hard"),
    ("alpha-Escape", "ALPHA", "escape"),
    ("POPMUSIC-Hard", "POPMUSIC", "hard"),
    ("POPMUSIC-Escape", "POPMUSIC", "escape"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    require(
        type(config["protocol_spec_id"]) is int and config["protocol_spec_id"] == 1,
        "未知E3正式协议版本",
    )
    require(config["wall_clock_limit"] is None, "E3不得附带算法墙钟截止")
    require(
        config["conditions"]
        == [
            {"name": name, "prior_kind": prior, "constraint_mode": mode}
            for name, prior, mode in CONDITIONS
        ],
        "E3必须覆盖两先验×Hard/Escape四条件",
    )
    require(config["evolution_seeds"] == [1103, 2207, 3313, 4409, 5519], "需要五个完整演化重复")
    require(
        config["dimensions"] == [500, 1000] and config["colonies"] == 32,
        "正式E3固定500/1K和32 colonies",
    )
    settings = training_settings(config, config["evolution_seeds"][0])
    require(
        settings.evolution.population == 128
        and settings.evolution.generations == 50
        and settings.evolution.elites == 4
        and not settings.evolution.no_feedback,
        "正式E3需要每条件独立GP-Full的128×50训练",
    )
    require(
        settings.instances_per_panel == 16
        and settings.solver_seeds_per_instance == 2
        and settings.budget_kind == "search_tour_evaluations"
        and settings.budgets == ((500, 4096), (1000, 4096))
        and settings.preparation_mode == "cached",
        "正式面板或FE预算改变",
    )
    solver = SolverSettings(**config["solver"])
    graph = GraphSettings(**config["graph"])
    require(
        solver.ants == 32
        and (graph.primary_width, graph.backup_width, graph.ls_width) == (16, 64, 20)
        and (solver.primary_width, solver.backup_width, solver.ls_width) == (16, 64, 20)
        and graph.uniform_backup_slots == 16
        and graph.preparation_seed == 80471,
        "图槽位或准备随机流改变",
    )
    for prior in ("ALPHA", "POPMUSIC"):
        require(
            PriorSettings(**config["priors"][prior]) == PriorSettings(kind=prior),
            "正式先验必须沿用已核验设置",
        )
    require(
        all(
            type(config[key]) is int and config[key] == expected
            for key, expected in (
                ("graph_spec_id", 1),
                ("matching_spec_id", 2),
                ("escape_spec_id", 1),
            )
        ),
        "图机制版本改变",
    )
    require(len(static_policies(config)) == 160, "E3需要完整160配置Static族")
    require(
        config["static_search"]["data_pools"]
        == {"search": {"offset": 0, "count": 64}, "validation": {"offset": 64, "count": 32}},
        "Static调参成员区间改变",
    )


def training_settings(config, seed):
    require(seed in config["evolution_seeds"], "演化seed未预登记")
    return TrainingSettings(
        **config["training"],
        evolution=EvolutionSettings(**config["evolution"]),
        evolution_seed=seed,
        scope="E3_formal_training",
    )


def training_panels(config, members):
    """与TrainingRun共享同一MT19937面板规则，先冻结全部50代，不看适应度。"""
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


def static_policies(config):
    """沿用E1已登记Static笛卡尔积；E3按四条件各自完整搜索和固定验证。"""
    grid, output = config["static_search"]["policy_grid"], []
    for level, region in product(grid["mne_levels"], grid["regions"]):
        variants = [{"restart_mode": "none"}]
        variants.extend(
            {"restart_mode": "periodic", "restart_period": value} for value in grid["periods"]
        )
        variants.extend(
            {"restart_mode": "bernoulli", "restart_probability": value}
            for value in grid["probabilities"]
        )
        output.extend(
            BaselinePolicy(mne_level=level, max_mne_level=level, region=region, **value)
            for value in variants
        )
    require(len({p.identifier for p in output}) == len(output), "Static配置身份重复")
    return tuple(output)


def preparation_jobs(config, members, panels, development):
    """只准备实际训练采样、完整GP验证和Static调参；TEST仅冻结ID，不准备坐标。"""
    jobs, all_ids = [], set()
    for n in config["dimensions"]:
        roles = {}
        for role in ("train", "validation", "test"):
            ids = members[str(n)][role]
            require(len(ids) == config["data"]["expected_counts"][str(n)][role], "split规模改变")
            require(len(ids) == len(set(ids)) and not set(ids) & all_ids, "split成员重复或交叉")
            all_ids.update(ids)
        sampled = {
            name
            for generation in panels
            for panel in generation
            if panel["dimension"] == n
            for name in panel["ids"]
        }
        require(sampled <= set(members[str(n)]["train"]), "采样面板越过训练池")
        for name in sorted(sampled):
            roles.setdefault(name, []).append("gp_training_sampled")
        for name in members[str(n)]["validation"]:
            roles.setdefault(name, []).append("gp_validation")
        require(not set(development[str(n)]) & all_ids, "development与主split重叠")
        require(len(development[str(n)]) == len(set(development[str(n)])), "development成员重复")
        all_ids.update(development[str(n)])
        for role, selection in config["static_search"]["data_pools"].items():
            selected = development[str(n)][
                selection["offset"] : selection["offset"] + selection["count"]
            ]
            require(len(selected) == selection["count"], "Static开发成员不足")
            for name in selected:
                roles.setdefault(name, []).append(f"static_{role}")
        require(not set(roles) & set(members[str(n)]["test"]), "图准备计划包含TEST成员")
        jobs.extend(
            {"dimension": n, "instance_id": name, "roles": role}
            for name, role in sorted(roles.items())
        )
    require(len({job["instance_id"] for job in jobs}) == len(jobs), "跨规模准备身份重复")
    return [{"index": i, **job} for i, job in enumerate(jobs)]
