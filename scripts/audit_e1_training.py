#!/usr/bin/env python3
"""流式核验完整E1训练及其一致快照；重算路线/fitness并从原seed重放演化，不启动GPU。"""

import argparse
import datetime
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import Evolution, EvolutionSettings, individual_from_program  # noqa: E402
from gp_faco.fitness import aggregate_panels, score_panel  # noqa: E402
from gp_faco.program_ir import Program, export_tree  # noqa: E402
from gp_faco.training import (  # noqa: E402
    TrainingData,
    finite_or_none,
    json_value,
    training_manifest,
)
from gp_faco.worker import (  # noqa: E402
    SolverSettings,
    SolveTask,
    WorkerProtocol,
    content_hash,
    coordinate_hash,
    file_hash,
)
from research_e1 import load_plan, training_settings  # noqa: E402
from research_e1_available_gpu import available_protocol, load_amendment  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_context(directory, freeze, amendment_path, snapshot):
    plan, members, panels = load_plan(freeze)
    state = load_checkpoint(directory / "checkpoint.json")
    if not snapshot:
        require(
            state["phase"] in ("complete", "failed")
            and state["pending"] is None
            and state["active_worker"] is None,
            "完整审计要求运行真实终态；活动作业只能使用--snapshot",
        )
    manifest = load_checkpoint(directory / "manifest.json")
    require(
        manifest == state["manifest"] and content_hash(manifest) == state["run_id"], "run身份不符"
    )
    selection = load_checkpoint(directory / "data_selection.json")
    identity = selection["identity"]
    condition, seed = identity["condition"], manifest["settings"]["evolution_seed"]
    settings = training_settings(plan["config"], condition, seed)
    require(json_value(asdict(settings)) == manifest["settings"], "演化/FE/完整配置改变")
    expected_identity = {
        "database_sha256": plan["database_sha256"],
        "split_sha256": plan["split_sha256"],
        "config_sha256": plan["config_sha256"],
        "entrypoint_sha256": plan["entrypoint_sha256"],
        "e1_plan_sha256": plan["sha256"],
        "condition": condition,
        "selection": "complete frozen train and validation splits; no development or test members",
        "reference_status": "user_supplied_not_independently_certified",
    }
    p = manifest["worker_protocol"]
    protocol = WorkerProtocol(
        **{
            f.name: SolverSettings(**p[f.name]) if f.name == "settings" else p[f.name]
            for f in fields(WorkerProtocol)
            if f.init
        }
    )
    require(json_value(protocol.manifest()) == p, "worker实现身份改变")
    config = plan["config"]
    expected_protocol = WorkerProtocol(
        p["gpu_uuid"],
        config["gpu_model"],
        config["driver_version"],
        plan["native_binary_sha256"],
        dimensions=tuple(config["dimensions"]),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
        execution_host=p["execution_host"],
    )
    expected_p = json_value(expected_protocol.manifest())
    if "hardware_amendment_sha256" in identity:
        require(amendment_path is not None, "该运行要求明确的硬件补充协议")
        amendment = load_amendment(amendment_path, plan)
        require(amendment["sha256"] == identity["hardware_amendment_sha256"], "硬件补充身份不同")
        checked = available_protocol(
            plan, amendment, p["gpu_uuid"], f"{p['gpu_model']},{p['driver_version']}"
        )
        expected_p.update(gpu_model=checked.gpu_model, driver_version=checked.driver_version)
        expected_identity.update(
            entrypoint_sha256=amendment["entrypoint_sha256"],
            frozen_training_entrypoint_sha256=plan["entrypoint_sha256"],
            hardware_amendment_sha256=amendment["sha256"],
        )
    require(p == expected_p, "硬件调整改变了共同底座、batch形状或注册容量")
    require(identity == expected_identity, "数据/入口/条件来源改变")
    for n in config["dimensions"]:
        for role, frozen_role in (("training", "train"), ("validation", "validation")):
            require(selection[role][str(n)] == members[str(n)][frozen_role], "未完整使用冻结split")
    for worker in state["worker_history"]:
        require(
            worker["start_method"] == "spawn"
            and worker["host"] == p["execution_host"]
            and worker["protocol_sha256"] == protocol.sha256
            and worker["binary_sha256"] == p["binary_sha256"]
            and worker["device"]["name"] == p["gpu_model"],
            "实际worker与冻结身份不符",
        )
    return plan, members, panels, state, manifest, selection, settings, protocol


def verify_panels(state, selection, settings, frozen_panels, complete):
    width = settings.instances_per_panel
    expected_validation = [
        {
            "dimension": n,
            "ids": selection["validation"][str(n)][i : i + width],
            "seeds": list(settings.validation_seeds),
        }
        for n, _ in settings.budgets
        for i in range(0, len(selection["validation"][str(n)]), width)
    ]
    require(state["validation_panels"] == expected_validation, "验证成员/seed/面板有缩减或改变")
    history = state["training_panels"]
    require(len(history) <= settings.evolution.generations, "训练面板超过冻结代数")
    if complete:
        require(len(history) == settings.evolution.generations, "未完成全部代际面板")
    rng = random.Random(settings.panel_seed)
    for g, entry in enumerate(history):
        drawn = []
        for n, _ in settings.budgets:
            ids = sorted(rng.sample(selection["training"][str(n)], width))
            seeds = []
            while len(seeds) < settings.solver_seeds_per_instance:
                value = rng.getrandbits(64)
                if value not in seeds:
                    seeds.append(value)
            drawn.append({"dimension": n, "ids": ids, "seeds": seeds})
        require(
            entry["generation"] == g and entry["panels"] == drawn == frozen_panels[g],
            "面板RNG重放不符",
        )
        require(
            entry["panel_id"]
            == content_hash(
                {
                    "run_id": state["run_id"],
                    "generation": g,
                    "panels": drawn,
                }
            ),
            "代面板摘要不符",
        )
    require(json_value(rng.getstate()) == state["panel_rng"], "快照面板RNG不符")


WORK_FIELDS = {
    "search_tour_evaluations": "total_tour_evaluations",
    "completed_batches": "completed_batches",
    "completed_construction_steps": "completed_construction_steps",
    "completed_ls_evaluations": "completed_ls_evaluations",
    "native_actual_seconds": "actual_seconds",
    "charged_seconds": "charged_seconds",
    "overrun_seconds": "overrun_seconds",
}


def verify_attempts(record, worker_pids, retries):
    attempts = record["attempts"]
    require(1 <= len(attempts) <= retries + 1, "实际提交次数超出基础设施重试协议")
    returned = [i for i, a in enumerate(attempts) if a["status"] == "returned"]
    require(returned in ([], [len(attempts) - 1]), "普通返回后又重新评价或多个返回挑选")
    failures, unobserved = 0, 0
    for attempt in attempts:
        require(attempt["worker_pid"] in worker_pids, "提交到未知worker")
        require(
            attempt["status"]
            in (
                "returned",
                "broken_process_pool",
                "worker_terminated_unobserved",
            ),
            "已完成收据仍有活动或未知提交状态",
        )
        failures += attempt["status"] != "returned"
        unobserved += attempt["status"] == "worker_terminated_unobserved"
    if returned:
        require(record["outcome"]["worker_pid"] == attempts[-1]["worker_pid"], "返回PID与提交不符")
    else:
        require(
            len(attempts) == retries + 1 and record["outcome"]["status"] == "failed",
            "尚可等待的任务被伪造为完成",
        )
    return failures, unobserved, bool(returned)


def verify_costs(totals, recorded):
    """计数即使达到万亿量级也逐整数相等；只有累加计时允许浮点求和误差。"""
    timing = {
        "worker_seconds",
        "evaluator_seconds",
        "registration_seconds",
        "native_actual_seconds",
        "charged_seconds",
        "overrun_seconds",
    }
    counted = {
        "solve_jobs",
        "preparation_jobs",
        "failed_solves",
        "valid_members",
        "infrastructure_failures",
        "unobserved_terminated_attempts",
        *WORK_FIELDS,
    } - timing
    for field in timing | counted:
        actual, expected = totals.get(field, 0), recorded.get(field, 0)
        if field in counted:
            require(
                type(actual) is int and type(expected) is int and actual == expected,
                f"整数工作量账目不平: {field}",
            )
        else:
            require(
                type(expected) in (int, float)
                and math.isfinite(expected)
                and expected >= 0
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8),
                f"计时资源账目不平: {field}",
            )


def audit_receipts(directory, state, protocol, settings, source, allowed_ids, complete):
    """逐条读取原始路线后释放，避免完整128×50运行把全部tour常驻内存。"""
    completed = state["completed"]
    files = {p.stem for p in (directory / "tasks").glob("*.json")}
    require(set(completed) <= files, "已完成收据丢失，不能重算代替")
    if complete:
        require(files == set(completed), "终态仍有未归集或额外任务收据")
    totals, by_phase = Counter(), {"training": Counter(), "validation": Counter()}
    observations = Counter()
    worker_pids = {v["pid"] for v in state["worker_history"]}
    admissions = Counter()
    for a in state.get("admission", []):
        observations["before_foreign_or_unknown"] += a.get("foreign_processes") != 0
        if a.get("foreign_processes") == 0:
            admissions[a["task_key"]] += 1
    problems, labels, prepared, evaluated = {}, {}, {}, {}
    max_cost_error = 0.0
    for key, digest in sorted(completed.items()):
        record = load_checkpoint(directory / "tasks" / f"{key}.json")
        require(content_hash(record) == digest and record["key"] == key, "任务内容摘要改变")
        d, outcome = record["description"], record["outcome"]
        require(
            key
            == content_hash({"run_id": state["run_id"], "kind": record["kind"], "description": d}),
            "任务身份改变",
        )
        require(d["protocol_sha256"] == protocol.sha256, "任务协议身份不符")
        failures, unobserved, returned = verify_attempts(
            record, worker_pids, settings.infrastructure_retries
        )
        totals["infrastructure_failures"] += failures
        totals["unobserved_terminated_attempts"] += unobserved
        require(admissions[key] >= len(record["attempts"]), "缺失提交前空闲GPU观察")
        if returned:
            require("gpu_boundary_after" in outcome, "实际返回缺少GPU边界记录")
            observations["after_foreign"] += bool(
                outcome["gpu_boundary_after"].get("foreign_processes")
            )
            observations["after_unknown"] += (
                outcome["gpu_boundary_after"].get("foreign_processes") is None
            )
        require(outcome.get("engine_generation", 0) == 0, "阶段中更换了Engine容量")
        for name, digest in d["problems"]:
            require(name in allowed_ids, "读取任务越过训练/验证成员")
            if name not in problems:
                problems[name] = source.load_instance(name)
                labels[name] = source.load_label(name)
            require(
                problems[name].dimension == d["dimension"]
                and coordinate_hash(problems[name]) == digest,
                "任务坐标身份改变",
            )
        for field, origin in (
            ("worker_seconds", "worker_seconds"),
            ("registration_seconds", "registration_seconds"),
        ):
            value = outcome.get(origin, 0.0)
            require(
                type(value) in (int, float) and math.isfinite(value) and value >= 0, "无效资源账目"
            )
            totals[field] += value
        evaluation_seconds = record["evaluation_seconds"]
        require(
            type(evaluation_seconds) in (int, float)
            and math.isfinite(evaluation_seconds)
            and evaluation_seconds >= 0,
            "无效外部评价资源",
        )
        totals["evaluator_seconds"] += evaluation_seconds
        if record["kind"] == "preparation":
            require(
                outcome["status"] == "completed" and outcome["preparation_id"] == content_hash(d),
                "准备收据身份不符",
            )
            require(all(outcome[k] == v for k, v in d.items()), "准备返回与任务不同")
            names = {name for name, _ in d["problems"]}
            require(
                set(record["checked"]) == set(outcome["registration_fees"]) == names,
                "准备费用成员不完整",
            )
            for name in names:
                for field in ("cheap_seconds", "preparation_seconds"):
                    value = outcome["registration_fees"][name][field]
                    require(
                        type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= 0
                        and record["checked"][name][field] == value,
                        "准备费用核验不符",
                    )
            prepared[key] = record
            totals["preparation_jobs"] += 1
            continue
        require(record["kind"] == "solve", "未知任务类别")
        program_record = load_checkpoint(directory / "programs" / f"{d['program_sha256']}.json")
        program = Program.from_dict(program_record["program"])
        require(
            program.sha256 == d["program_sha256"] == program_record["program_sha256"],
            "程序IR身份不符",
        )
        task = SolveTask(
            d["occurrence_id"],
            program,
            tuple(problems[name] for name, _ in d["problems"]),
            tuple(tuple(v) for v in d["replicas"]),
            d["budget_seconds"],
            d["preparation_mode"],
            d["experiment_mask"],
            d["preparation_charges"],
            d["evaluation_limit_per_colony"],
        )
        require(json_value(task.manifest(protocol)) == d, "重建实际任务manifest不符")
        require(
            task.budget_seconds is None
            and task.preparation_charges is None
            and task.evaluation_limit_per_colony == dict(settings.budgets)[task.dimension]
            and task.preparation_mode == settings.preparation_mode
            and task.experiment_mask == settings.experiment_mask,
            "FE/模式/mask或无墙钟协议改变",
        )
        score = score_panel(
            task, protocol, outcome, {p.instance_id: labels[p.instance_id] for p in task.problems}
        )
        require(json_value(asdict(score)) == record["checked"], "从实际路线重算的外部fitness不同")
        require(task.occurrence_id not in evaluated, "重复评价同一个预定位置")
        evaluated[task.occurrence_id] = task, score
        totals["solve_jobs"] += 1
        totals["failed_solves"] += score.failed
        totals["valid_members"] += len(score.members)
        phase = "training" if ":generation" in task.occurrence_id else "validation"
        by_phase[phase]["solve_jobs"] += 1
        by_phase[phase]["failed_solves"] += score.failed
        native = outcome.get("native_result", {})
        for field, origin in WORK_FIELDS.items():
            value = native.get(origin, 0)
            # 非法原生返回照原事务保留失败；只归集实际记录的合法资源数值。
            integer = field not in ("native_actual_seconds", "charged_seconds", "overrun_seconds")
            if (
                (type(value) is int if integer else type(value) in (int, float))
                and math.isfinite(value)
                and value >= 0
            ):
                totals[field] += value
                by_phase[phase][field] += value
        if not score.failed:
            for (name, _), item in zip(task.replicas, native["items"], strict=True):
                max_cost_error = max(
                    max_cost_error, abs(tour_cost(problems[name], item["tour"]) - item["cost"])
                )
    for name, fee in state["fees"].items():
        require(
            fee["preparation_record"] in prepared and fee["protocol_sha256"] == protocol.sha256,
            "费用来源记录缺失",
        )
        record = prepared[fee["preparation_record"]]
        require(
            fee["coordinate_sha256"] == coordinate_hash(problems[name])
            and all(
                fee[f] == record["checked"][name][f]
                for f in ("cheap_seconds", "preparation_seconds")
            ),
            "费用历史被改写",
        )
    for task, _ in evaluated.values():
        require(
            all(p.instance_id in state["fees"] for p in task.problems), "求解发生在准备资源归档之前"
        )
    pending = state["pending"]
    if pending:
        for a in pending["attempts"]:
            totals["infrastructure_failures"] += a["status"] in (
                "broken_process_pool",
                "worker_terminated_unobserved",
            )
            totals["unobserved_terminated_attempts"] += (
                a["status"] == "worker_terminated_unobserved"
            )
    verify_costs(totals, state["costs"])
    return (
        evaluated,
        dict(totals),
        {k: dict(v) for k, v in by_phase.items()},
        dict(observations),
        max_cost_error,
    )


class Rescore:
    """把已核验任务严格对应到原演化位置；快照只允许预定顺序的已提交前缀。"""

    def __init__(self, state, evaluated, protocol):
        self.state, self.evaluated, self.protocol = state, evaluated, protocol
        self.used, self.gap = set(), False

    def __call__(self, program, occurrence, panels, required):
        tasks, scores = [], []
        for index, panel in enumerate(panels):
            name = f"{self.state['run_id']}:{occurrence}:panel{index}"
            if name not in self.evaluated:
                self.gap = True
                require(not required, "已评个体/候选缺失完整面板收据")
                continue
            require(not self.gap, "跳过未完成的预定评价位置后继续求解")
            task, score = self.evaluated[name]
            require(
                task.program == program and task.dimension == panel["dimension"],
                "程序/规模与原始演化重放不符",
            )
            require(
                task.replicas
                == tuple((name, seed) for name in panel["ids"] for seed in panel["seeds"]),
                "未完整共用代面板/固定验证面板",
            )
            require([p.instance_id for p in task.problems] == panel["ids"], "实际问题顺序改变")
            self.used.add(name)
            tasks.append(task)
            scores.append(score)
        value = (
            aggregate_panels(tuple(tasks), self.protocol, tuple(scores), self.protocol.dimensions)
            if len(tasks) == len(panels)
            else None
        )
        return value, tasks


def replay_evolution(directory, state, evaluated, protocol, settings, complete):
    replay = Evolution(EvolutionSettings(**asdict(settings.evolution)), settings.evolution_seed)
    replay.initialize()
    rescore = Rescore(state, evaluated, protocol)
    current = state["evolution"]
    generations = []
    for g in range(current["generation"] + 1):
        if g == len(state["training_panels"]):
            require(
                g == current["generation"] and current["panel_id"] is None, "缺少已使用的代面板"
            )
            break
        entry = state["training_panels"][g]
        replay.begin_panel(entry["panel_id"])
        for index, individual in enumerate(replay.population):
            assigned = g < current["generation"] or current["population"][index]["valid"]
            program = export_tree(individual)
            value, _ = rescore(
                program, f"generation{g}:individual{index}", entry["panels"], assigned
            )
            if assigned:
                require(value is not None, "已评个体没有完整实际fitness")
                replay.assign(index, value, entry["panel_id"], program.sha256)
        finished = len(current["winners"]) > g
        if finished:
            replay.finish_generation()
            archive = directory / "generations" / f"{g:04d}.json"
            if complete or g < current["generation"] or archive.exists():
                saved = load_checkpoint(archive)
                require(
                    saved
                    == {
                        "run_id": state["run_id"],
                        "panels": entry["panels"],
                        "evolution": json_value(replay.state_dict()),
                    },
                    "完整代归档与原seed/实际fitness重放不符",
                )
            generations.append(
                {
                    "generation": g,
                    "winner_sha256": Program.from_dict(replay.winners[-1]["program"]).sha256,
                    "panel_id": entry["panel_id"],
                }
            )
        if g < current["generation"]:
            require(finished and replay.advance(), "尚未评完整代就进入下一代")
    require(json_value(replay.state_dict()) == current, "种群、精英、fitness或DEAP RNG重放不符")
    require(
        state["panels"]
        == (
            state["training_panels"][current["generation"]]["panels"]
            if current["panel_id"] is not None
            else []
        ),
        "快照活动面板不符",
    )
    shortlist = []
    if state["phase"] != "training":
        require(
            len(replay.winners) == settings.evolution.generations and not replay.advance(),
            "最终一代未完成或又生成后代",
        )
        shortlist = replay.shortlist()
        require(
            [p.to_dict() for p in shortlist] == state["shortlist"],
            "shortlist不是全部代冠军加最终种群",
        )
        for program in shortlist:
            required = complete or program.sha256 in state["validation_results"]
            value, tasks = rescore(
                program, f"validation:{program.sha256}", state["validation_panels"], required
            )
            if program.sha256 in state["validation_results"]:
                result = state["validation_results"][program.sha256]
                require(
                    value is not None
                    and result["fitness"] == finite_or_none(value)
                    and result["nodes"] == len(program.opcode)
                    and result["expression"] == str(individual_from_program(program, replay.pset))
                    and result["task_ids"] == [t.task_id(protocol) for t in tasks],
                    "固定验证的宏平均、复杂度或任务归集改变",
                )
        require(
            set(state["validation_results"]) <= {p.sha256 for p in shortlist}, "验证包含额外候选"
        )
        if complete:
            require(
                set(state["validation_results"]) == {p.sha256 for p in shortlist},
                "未验证完整短名单",
            )
    else:
        require(
            not state["validation_results"]
            and not state["shortlist"]
            and state["selected"] is None,
            "完整训练结束前已经选择验证候选",
        )
    require(rescore.used == set(evaluated), "存在未归集或不属于原演化的实际求解")
    if complete:
        verify_selection(directory, state, shortlist)
    return {
        "completed_generations": len(generations),
        "generations": generations,
        "current_generation": current["generation"],
        "assigned_current_individuals": sum(v["valid"] for v in current["population"]),
        "validation_candidates": len(shortlist),
        "completed_validation_candidates": len(state["validation_results"]),
        "variation_counts": replay.variation_counts,
    }


def verify_selection(directory, state, shortlist):
    results = state["validation_results"]
    finite = [sha for sha, result in results.items() if result["fitness"] is not None]
    if not finite:
        require(
            state["phase"] == "failed"
            and state["selected"] is None
            and not (directory / "selected_program.json").exists(),
            "全部失败却导出成功候选",
        )
        return
    selected = min(finite, key=lambda sha: (results[sha]["fitness"], results[sha]["nodes"], sha))
    expected = {"program_sha256": selected, **results[selected]}
    exported = load_checkpoint(directory / "selected_program.json")
    require(
        state["phase"] == "complete" and state["selected"] == expected, "最终选择违反统一验证排序"
    )
    require(
        exported["run_id"] == state["run_id"]
        and exported["scope"] == "E1_formal_training"
        and exported["selection"] == expected
        and exported["manifest"] == state["manifest"]
        and exported["costs"] == state["costs"]
        and exported["validation_results_sha256"] == content_hash(results)
        and exported["validation_panels_sha256"] == content_hash(state["validation_panels"])
        and Program.from_dict(exported["program"])
        == next(p for p in shortlist if p.sha256 == selected),
        "导出程序缺失冻结来源或与验证选择不同",
    )


def audit(directory, freeze, amendment_path=None, snapshot=False):
    plan, members, panels, state, manifest, selection, settings, protocol = verify_context(
        directory, freeze, amendment_path, snapshot
    )
    complete = not snapshot
    verify_panels(state, selection, settings, panels, complete)
    database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    allowed = {
        name
        for role in ("training", "validation")
        for ids in selection[role].values()
        for name in ids
    }
    with IndexedDataset(database, PROJECT.parent / "Datasets/TSP") as source:
        for n in protocol.dimensions:
            for role in ("train", "validation"):
                require(
                    sorted(source.record_ids(role, n)) == members[str(n)][role], "当前split ID改变"
                )
            require(
                not allowed.intersection(source.record_ids("development", n))
                and not allowed.intersection(members[str(n)]["test"]),
                "训练/验证借用了development或test成员",
            )
        data = TrainingData(
            source,
            {int(n): ids for n, ids in selection["training"].items()},
            {int(n): ids for n, ids in selection["validation"].items()},
            selection["identity"],
        )
        require(
            training_manifest(settings, protocol, data) == manifest,
            "软件、数据、选择规则或核心源码身份改变",
        )
        evaluated, totals, phases, observations, max_error = audit_receipts(
            directory, state, protocol, settings, source, allowed, complete
        )
    replay = replay_evolution(directory, state, evaluated, protocol, settings, complete)
    if complete:
        require(
            phases["training"]["solve_jobs"] == plan["training_native_calls_per_run"],
            "正式训练未覆盖全部6400个体位置",
        )
        require(
            phases["validation"]["solve_jobs"]
            == replay["validation_candidates"] * len(state["validation_panels"]),
            "验证任务未完整覆盖",
        )
        if not phases["training"]["failed_solves"]:
            require(
                phases["training"]["search_tour_evaluations"]
                == plan["training_search_tour_evaluations_per_run"],
                "完整训练FE总量不符",
            )
        summary = load_checkpoint(directory / "summary.json")
        require(
            summary["status"] == state["phase"]
            and summary["run_id"] == state["run_id"]
            and summary["costs"] == state["costs"]
            and summary["selected"] == state["selected"],
            "终态摘要与原始收据不符",
        )
    return {
        "status": "passed" if complete else "snapshot_passed",
        "scope": "formal E1 training integrity; no test-performance conclusion",
        "complete_training_and_validation_audit": complete,
        "training_outcome": state["phase"],
        "formal_test_released": False,
        "snapshot_observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_state_sha256": content_hash(state),
        "run_id": state["run_id"],
        "artifact_directory": str(directory.relative_to(PROJECT)),
        "e1_plan_sha256": plan["sha256"],
        "protocol": manifest["worker_protocol"],
        "condition": selection["identity"]["condition"],
        "evolution_seed": settings.evolution_seed,
        "settings": manifest["settings"],
        "work_by_phase": phases,
        "recomputed_costs": totals,
        "recorded_costs": state["costs"],
        "resource_observations": observations,
        "clean_gpu_observations": not any(observations.values()),
        "max_independent_cost_error": max_error,
        "replay": replay,
        "selected": state["selected"],
        "required_training_fitness_evaluations": plan["training_fitness_evaluations_per_run"],
        "required_training_native_calls": plan["training_native_calls_per_run"],
        "required_training_search_tour_evaluations": plan[
            "training_search_tour_evaluations_per_run"
        ],
        "audit_source_sha256": file_hash(Path(__file__)),
        "limits": [
            "A live snapshot audits only committed receipts in one atomic checkpoint; "
            "it is not a terminal run.",
            "Failed task results are retained and replayed as non-finite fitness; "
            "they are never removed from coverage.",
            "Observer timeouts are recorded observations of the same Future, "
            "not algorithm stopping limits.",
            "Reference labels are user-supplied and not independently certified; "
            "only training/validation labels are read.",
            "Hardware timings must be stratified by model/driver; "
            "finite compatibility replay is not a universal equivalence proof.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--hardware-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot", action="store_true", help="仅审计当前原子快照，不宣称完整运行通过"
    )
    args = parser.parse_args()
    for name in ("input", "freeze", "hardware_amendment", "output"):
        value = getattr(args, name)
        if value is not None:
            value = value.resolve()
            require(value.is_relative_to(PROJECT), "所有审计文件必须在GPLSACO内")
            setattr(args, name, value)
    report = audit(args.input, args.freeze, args.hardware_amendment, args.snapshot)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": report["run_id"],
                "solve_jobs": report["recomputed_costs"].get("solve_jobs", 0),
                "failed_solves": report["recomputed_costs"].get("failed_solves", 0),
                "replayed_generations": report["replay"]["completed_generations"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
