#!/usr/bin/env python3
"""从真实训练产物重算全部fitness、重放DEAP/面板RNG并核查恢复；不启动CUDA。"""

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import Evolution, EvolutionSettings  # noqa: E402
from gp_faco.fitness import aggregate_panels, score_panel  # noqa: E402
from gp_faco.program_ir import Program, export_tree  # noqa: E402
from gp_faco.training import json_value  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    SolverSettings,
    SolveTask,
    WorkerProtocol,
    content_hash,
    coordinate_hash,
    file_hash,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summarize(directory, *, database=None, dataset_root=None, entrypoint=None, checks=None):
    database = database or PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
    dataset_root = dataset_root or PROJECT.parent / "Datasets/TSP"
    entrypoint = entrypoint or PROJECT / "scripts/train_gp.py"
    state = load_checkpoint(directory / "checkpoint.json")
    require(state["phase"] == "complete" and state["pending"] is None, "运行尚未完整结束")
    require(state["active_worker"] is None, "checkpoint仍标记活动worker")
    manifest = load_checkpoint(directory / "manifest.json")
    require(
        manifest == state["manifest"] and content_hash(manifest) == state["run_id"], "run身份不符"
    )
    for name, fingerprint in manifest["sources"].items():
        require(
            file_hash(PROJECT / "python/gp_faco" / name) == fingerprint, f"训练源码改变: {name}"
        )
    p = manifest["worker_protocol"]
    protocol = WorkerProtocol(
        **{
            v.name: SolverSettings(**p[v.name]) if v.name == "settings" else p[v.name]
            for v in fields(WorkerProtocol)
            if v.init
        }
    )
    require(json_value(protocol.manifest()) == p, "worker代码身份改变")
    require(
        file_hash(PROJECT / protocol.extension_directory / "gp_faco_ext.so") == p["binary_sha256"],
        "GPU二进制身份改变",
    )
    require(
        all(
            v["start_method"] == "spawn"
            and v["host"] == p["execution_host"]
            and v["protocol_sha256"] == protocol.sha256
            for v in state["worker_history"]
        ),
        "实际worker与冻结协议不符",
    )
    selection = load_checkpoint(directory / "data_selection.json")
    require(selection["identity"] == manifest["data"]["identity"], "数据身份不符")
    require(file_hash(database) == selection["identity"]["database_sha256"], "索引身份改变")
    require(
        file_hash(PROJECT / "provenance/splits.v1.json") == selection["identity"]["split_sha256"],
        "split身份改变",
    )
    require(
        file_hash(entrypoint) == selection["identity"]["entrypoint_sha256"],
        "训练入口改变",
    )
    for role in ("training", "validation"):
        for n, ids in selection[role].items():
            require(
                manifest["data"][role][n] == {"count": len(ids), "ids_sha256": content_hash(ids)},
                "数据成员表摘要不符",
            )
    settings = manifest["settings"]
    counted = settings.get("budget_kind") == "search_tour_evaluations"
    width = settings["instances_per_panel"]
    fixed_validation = [
        {
            "dimension": n,
            "ids": selection["validation"][str(n)][start : start + width],
            "seeds": settings["validation_seeds"],
        }
        for n, _ in settings["budgets"]
        for start in range(0, len(selection["validation"][str(n)]), width)
    ]
    require(fixed_validation == state["validation_panels"], "验证不是预先固定的完整面板")
    rng = random.Random(settings["panel_seed"])
    for entry in state["training_panels"]:
        drawn = []
        for n, _ in settings["budgets"]:
            ids = sorted(rng.sample(selection["training"][str(n)], settings["instances_per_panel"]))
            seeds = []
            while len(seeds) < settings["solver_seeds_per_instance"]:
                seed = rng.getrandbits(64)
                if seed not in seeds:
                    seeds.append(seed)
            drawn.append({"dimension": n, "ids": ids, "seeds": seeds})
        require(drawn == entry["panels"], "面板/求解seed不能从已保存RNG重放")
        require(
            content_hash(
                {"run_id": state["run_id"], "generation": entry["generation"], "panels": drawn}
            )
            == entry["panel_id"],
            "代面板摘要不符",
        )
    require(json_value(rng.getstate()) == state["panel_rng"], "最终面板RNG不符")
    files = {path.stem: path for path in (directory / "tasks").glob("*.json")}
    require(set(files) == set(state["completed"]), "任务日志与完整完成表有遗漏或额外成员")
    if protocol.graph_catalog is not None:
        require(
            {p.stem for p in (directory / "raw_returns").glob("*.json")} == set(files),
            "图训练独立原始返回集合与任务不符",
        )
        admissions = state.get("admission", [])
        require(
            len(admissions) == len(files)
            and {row["task_key"] for row in admissions} == set(files)
            and all(
                type(row.get("foreign_processes")) is int and row["foreign_processes"] == 0
                for row in admissions
            ),
            "图训练提交前的GPU观察缺失、重复或有污染",
        )
    records, evaluated, actual_measurements = [], {}, {}
    max_cost_error = 0.0
    members, total_restarts = 0, 0
    problems, labels = {}, {}
    with IndexedDataset(database, dataset_root) as source:
        for n in protocol.dimensions:
            development = set(source.record_ids("development", n))
            train = set(selection["training"][str(n)])
            val = set(selection["validation"][str(n)])
            require(
                not train & val and train | val <= development, "pilot越过开发池或训练/验证重叠"
            )
        for key, path in sorted(files.items()):
            record = load_checkpoint(path)
            require(
                record["outcome"].get("engine_generation", 0) == 0,
                "阶段内发生了未预期的Engine容量替换",
            )
            require(content_hash(record) == state["completed"][key], "任务产物摘要改变")
            require(
                key
                == content_hash(
                    {
                        "run_id": state["run_id"],
                        "kind": record["kind"],
                        "description": record["description"],
                    }
                ),
                "任务记录身份改变",
            )
            records.append(record)
            if protocol.graph_catalog is not None:
                raw = json.loads((directory / "raw_returns" / f"{key}.json").read_text())
                require(
                    raw
                    == {k: v for k, v in record["outcome"].items() if k != "gpu_boundary_after"},
                    "资源查询前的原始返回被改变",
                )
                require(
                    record["outcome"]["gpu_boundary_after"]["foreign_processes"] == 0,
                    "图训练返回后GPU占用未知或被污染",
                )
                graph_problems = tuple(
                    source.load_instance(name) for name, _ in record["description"]["problems"]
                )
                require(
                    json_value(protocol.graph_identity(graph_problems)["graph_inputs"])
                    == record["description"]["graph_inputs"]
                    == record["outcome"]["graph_inputs"],
                    "图训练准备/求解来源身份不符",
                )
            if record["kind"] == "preparation":
                continue
            d = record["description"]
            program = load_checkpoint(directory / "programs" / f"{d['program_sha256']}.json")
            ir = Program.from_dict(program["program"])
            require(ir.sha256 == d["program_sha256"], "实际程序IR与任务不符")
            for name, fingerprint in d["problems"]:
                if name not in problems:
                    problems[name] = source.load_instance(name)
                    labels[name] = source.load_label(name)
                require(coordinate_hash(problems[name]) == fingerprint, "任务坐标身份改变")
                fee = state["fees"][name]
                require(fee["coordinate_sha256"] == fingerprint, "冻结费用坐标身份不符")
                preparation = load_checkpoint(files[fee["preparation_record"]])
                require(
                    all(
                        fee[field] == preparation["checked"][name][field]
                        for field in ("cheap_seconds", "preparation_seconds")
                    ),
                    "冻结费用源记录不符",
                )
                actual = record["outcome"]["registration_fees"][name]
                actual_measurements.setdefault(name, set()).add(
                    (actual["cheap_seconds"], actual["preparation_seconds"])
                )
            task = SolveTask(
                d["occurrence_id"],
                ir,
                tuple(problems[name] for name, _ in d["problems"]),
                tuple(tuple(v) for v in d["replicas"]),
                d["budget_seconds"],
                d["preparation_mode"],
                d["experiment_mask"],
                None
                if d["preparation_charges"] is None
                else tuple(tuple(v) for v in d["preparation_charges"]),
                d.get("evaluation_limit_per_colony"),
            )
            require(json_value(task.manifest(protocol)) == d, "重建任务manifest不符")
            require(
                (counted and task.preparation_charges is None)
                or task.preparation_charges
                == tuple(
                    sorted(
                        (
                            p.instance_id,
                            state["fees"][p.instance_id]["cheap_seconds"],
                            state["fees"][p.instance_id]["preparation_seconds"],
                        )
                        for p in task.problems
                    )
                ),
                "个体之间或恢复后改变了固定费用",
            )
            score = score_panel(
                task,
                protocol,
                record["outcome"],
                {p.instance_id: labels[p.instance_id] for p in task.problems},
            )
            require(
                json_value(asdict(score)) == record["checked"] and not score.failed,
                "独立外部fitness核验失败",
            )
            require(task.occurrence_id not in evaluated, "重复求解了同一个实际评价位置")
            evaluated[task.occurrence_id] = task, score
            native = record["outcome"]["native_result"]
            members += len(score.members)
            total_restarts += sum(v["restarts"] for v in native["control_states"])
            for (name, _), item in zip(task.replicas, native["items"], strict=True):
                max_cost_error = max(
                    max_cost_error, abs(tour_cost(problems[name], item["tour"]) - item["cost"])
                )

    replay = Evolution(EvolutionSettings(**settings["evolution"]), settings["evolution_seed"])
    replay.initialize()
    used = set()
    generations = []

    def rescore(program, occurrence, panels):
        tasks, scores = [], []
        for index, panel in enumerate(panels):
            name = f"{state['run_id']}:{occurrence}:panel{index}"
            task, score = evaluated[name]
            require(
                task.program == program and task.dimension == panel["dimension"],
                "个体/规模任务不符",
            )
            require(
                task.replicas
                == tuple((name, seed) for name in panel["ids"] for seed in panel["seeds"]),
                "同代或统一验证面板不符",
            )
            actual_limit = task.evaluation_limit_per_colony if counted else task.budget_seconds
            require(actual_limit == dict(settings["budgets"])[task.dimension], "预算或次数限额改变")
            require(task.experiment_mask == settings["experiment_mask"], "mask改变")
            tasks.append(task)
            scores.append(score)
            used.add(name)
        return aggregate_panels(tuple(tasks), protocol, tuple(scores), protocol.dimensions), tasks

    for g in range(settings["evolution"]["generations"]):
        snapshot = load_checkpoint(directory / "generations" / f"{g:04d}.json")
        entry = state["training_panels"][g]
        require(snapshot["panels"] == entry["panels"], "代际归档面板不符")
        replay.begin_panel(entry["panel_id"])
        for index, individual in enumerate(replay.population):
            program = export_tree(individual)
            value, _ = rescore(program, f"generation{g}:individual{index}", entry["panels"])
            replay.assign(index, value, entry["panel_id"], program.sha256)
        replay.finish_generation()
        require(
            json_value(replay.state_dict()) == snapshot["evolution"],
            "DEAP代际/精英/fitness/RNG重放不符",
        )
        generations.append(
            {
                "generation": g,
                "unique_programs": len({export_tree(v).sha256 for v in replay.population}),
                "winner_sha256": Program.from_dict(replay.winners[-1]["program"]).sha256,
                "panel_id": entry["panel_id"],
            }
        )
        require(
            replay.advance() == (g + 1 < settings["evolution"]["generations"]),
            "最终代错误地产生后代",
        )
    require(json_value(replay.state_dict()) == state["evolution"], "最终种群/冠军/RNG重放不符")
    shortlist = replay.shortlist()
    require(
        [p.to_dict() for p in shortlist] == state["shortlist"],
        "验证shortlist未精确覆盖冠军+最终种群",
    )
    for program in shortlist:
        value, tasks = rescore(program, f"validation:{program.sha256}", state["validation_panels"])
        record = state["validation_results"][program.sha256]
        require(
            record["fitness"] == value
            and record["task_ids"] == [t.task_id(protocol) for t in tasks],
            "验证宏平均或任务覆盖不符",
        )
    require(used == set(evaluated), "存在未归集的实际求解任务")
    results = state["validation_results"]
    selected = min(results, key=lambda sha: (results[sha]["fitness"], results[sha]["nodes"], sha))
    exported = load_checkpoint(directory / "selected_program.json")
    require(
        state["selected"]["program_sha256"]
        == selected
        == Program.from_dict(exported["program"]).sha256,
        "导出选择不符合统一验证排序",
    )
    require(
        exported["manifest"] == manifest
        and exported["validation_results_sha256"] == content_hash(results),
        "导出未包含完整冻结来源",
    )
    paused = load_checkpoint(directory / "paused-checkpoint.json")
    require(
        paused["run_id"] == state["run_id"] and paused["costs"]["solve_jobs"] == 5, "暂停证据不符"
    )
    require(
        all(state["completed"].get(k) == v for k, v in paused["completed"].items()),
        "恢复重做/改变已完成结果",
    )
    require(all(state["fees"].get(k) == v for k, v in paused["fees"].items()), "恢复改变原有费用")
    pids = [r["pid"] for r in state["worker_history"]]
    require(len(set(pids)) == len(pids) >= 3, "未实际更换训练/恢复/验证worker")
    require(
        any(len(values) > 1 for values in actual_measurements.values()),
        "没有观察到恢复后的重新实测费用",
    )
    solves = [r for r in records if r["kind"] == "solve"]
    require(
        state["costs"]["solve_jobs"] == len(solves) and state["costs"]["valid_members"] == members,
        "累计任务/成员账目不符",
    )
    search_evaluations = sum(
        r["outcome"]["native_result"].get("total_tour_evaluations", 0) for r in solves
    )
    if counted:
        require(
            search_evaluations == state["costs"]["search_tour_evaluations"], "实际FE总数账目不符"
        )
    # 蚂蚁tour次数与内部工作量分别汇总；准备资源不混入搜索FE，也不伪称扣费。
    work_fields = (
        "total_tour_evaluations",
        "completed_construction_steps",
        "completed_ls_evaluations",
        "completed_batches",
    )
    work_by_phase = {}
    for phase, marker in (("training", ":generation"), ("validation", ":validation:")):
        phase_solves = [r for r in solves if marker in r["description"]["occurrence_id"]]
        work_by_phase[phase] = {
            "solve_jobs": len(phase_solves),
            **{
                field: sum(r["outcome"]["native_result"].get(field, 0) for r in phase_solves)
                for field in work_fields
            },
        }
    require(
        sum(v["solve_jobs"] for v in work_by_phase.values()) == len(solves),
        "训练/验证工作量分组有遗漏",
    )
    if checks is None:
        previous_checks = PROJECT / (
            "artifacts/gpu/evaluation-count" if counted else "artifacts/gpu/gp-training"
        )
        checks = (previous_checks / "pytest.log", previous_checks / "ctest.log")
    pytest_log, ctest_log = checks
    pytest_count = int(re.search(r"(\d+) passed", pytest_log.read_text()).group(1))
    match = re.search(r"100% tests passed, 0 tests failed out of (\d+)", ctest_log.read_text())
    require(
        match is not None and int(match.group(1)) >= (10 if counted else 8), "原生检查未完整通过"
    )
    return {
        "status": "passed",
        "scope": "engineering development only; no formal E1/E3 conclusion",
        "run_id": state["run_id"],
        "artifact_directory": str(directory.relative_to(PROJECT)),
        "protocol": p,
        "settings": settings,
        "generations": generations,
        "training_individuals": settings["evolution"]["population"]
        * settings["evolution"]["generations"],
        "solve_jobs": len(solves),
        "validation_candidates": len(shortlist),
        "valid_members": members,
        "search_tour_evaluations": search_evaluations,
        "work_by_phase": work_by_phase,
        "max_independent_cost_error": max_cost_error,
        "completed_restarts": total_restarts,
        "completed_batches": sum(
            r["outcome"]["native_result"]["completed_batches"] for r in solves
        ),
        "discarded_batches": sum(
            r["outcome"]["native_result"]["discarded_batches"] for r in solves
        ),
        "overrun_seconds_range": [
            min(r["outcome"]["native_result"]["overrun_seconds"] for r in solves),
            max(r["outcome"]["native_result"]["overrun_seconds"] for r in solves),
        ],
        "worker_pids": pids,
        "preserved_completed_records_after_pause": len(paused["completed"]),
        "instances_with_changed_actual_preparation_measurements": sum(
            len(v) > 1 for v in actual_measurements.values()
        ),
        "preparation_accounting": (
            "original resource history preserved; no charges or search-FE deductions"
            if counted
            else "original fixed charges preserved despite remeasured actual costs"
        ),
        "costs": state["costs"],
        "selected": state["selected"],
        "variation_counts": replay.variation_counts,
        "verified": [
            f"all {settings['evolution']['population'] * settings['evolution']['generations']} "
            "individual occurrences actually reevaluated",
            "common full panels and budgets",
            "DEAP evolution and panel RNG replay from original seeds",
            "final evaluated population",
            "exact champions plus final population shortlist",
            "fixed validation and external macro fitness",
            (
                "original completed results and preparation resource history preserved "
                "after real worker restart; no wall-clock limit or fee deduction"
                if counted
                else "original completed results and frozen fees preserved "
                "after real worker restart"
            ),
        ],
        "limits": [
            (
                "small pilot collapsed to one distinct validation candidate"
                if len(shortlist) == 1
                else "small pilot is insufficient for an efficacy conclusion"
            ),
            "FE excludes internal LS move checks, which are reported separately"
            if counted
            else "wall-clock stopping is not bitwise reproducible",
            "worker/evaluator costs recorded; previous profiling retains its own identity",
            "no formal budgets, five-seed E1/E3 or E4 efficacy evidence",
        ],
        "checks": {
            "gpu_python_passed": pytest_count,
            "gpu_ctest_passed": int(match.group(1)),
            "pytest_log_sha256": file_hash(pytest_log),
            "ctest_log_sha256": file_hash(ctest_log),
        },
        "source_sha256": {
            **manifest["sources"],
            "summarize_training.py": file_hash(Path(__file__)),
            str(entrypoint.relative_to(PROJECT)): file_hash(entrypoint),
        },
        "artifact_sha256": {
            name: file_hash(directory / name)
            for name in (
                "checkpoint.json",
                "manifest.json",
                "summary.json",
                "selected_program.json",
                "paused-checkpoint.json",
                "data_selection.json",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT / "artifacts/gpu/gp-training/pilot-v2"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "docs/reports/training_results.json"
    )
    parser.add_argument("--database", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--pytest-log", type=Path)
    parser.add_argument("--ctest-log", type=Path)
    args = parser.parse_args()
    if bool(args.pytest_log) != bool(args.ctest_log):
        parser.error("需要同时指定pytest和CTest日志")
    report = summarize(
        args.input.resolve(),
        database=args.database,
        dataset_root=args.dataset_root,
        entrypoint=args.entrypoint.resolve() if args.entrypoint else None,
        checks=(args.pytest_log, args.ctest_log) if args.pytest_log else None,
    )
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "solve_jobs",
                    "valid_members",
                    "validation_candidates",
                    "max_independent_cost_error",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
