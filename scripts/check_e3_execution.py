#!/usr/bin/env python3
"""真实E3统一执行通路验收；只用开发集，完成后由独立CPU入口重放审计。"""

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from audit_e3_static import verify_raw_returns  # noqa: E402
from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.configuration_search import SearchData, SearchSettings  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_execution import (  # noqa: E402
    RegisteredGraphStaticRun,
    RegisteredGraphTrainingRun,
    worker_protocol,
)
from gp_faco.e3_preparation import load_freeze  # noqa: E402
from gp_faco.e3_protocol import (  # noqa: E402
    require,
    static_policies,
    training_panels,
    training_settings,
)
from gp_faco.evaluation_run import json_value  # noqa: E402
from gp_faco.training import TrainingData, training_manifest  # noqa: E402
from gp_faco.training_audit import audit_receipts, replay_evolution, verify_panels  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402
from research_e3 import execution_sources  # noqa: E402
from summarize_configuration_search import summarize  # noqa: E402

CONDITION = "POPMUSIC-Escape"


def settings_and_grid(plan):
    # 此规模仅用于工程验收；正式入口仍只接受预登记128×50、4096 FE和完整160族。
    base = training_settings(plan["config"], 1103)
    training = replace(
        base,
        evolution=replace(base.evolution, population=8, generations=3),
        budgets=((500, 256), (1000, 256)),
        scope="E3_execution_development_engineering",
    )
    search = SearchSettings(
        purpose="static_tuning",
        evaluation_limits=(256,),
        instances_per_panel=16,
        solver_seeds=(17, 29),
        validation_shortlist_per_kind=2,
    )
    grid = static_policies(plan["config"])
    return training, search, tuple(grid[i] for i in (0, 39, 80, 159))


def data_for(source, plan, directory):
    train, validation, pools = {}, {}, {"search": {}, "validation": {}}
    for n in plan["config"]["dimensions"]:
        ids = source.record_ids("development", n)
        require(len(ids) >= 48, "开发池不足")
        train[n], validation[n] = ids[:16], ids[16:32]
        pools["search"][n], pools["validation"][n] = ids[:32], ids[32:48]
    identity = {
        "database_sha256": plan["database_sha256"],
        "split_sha256": plan["split_sha256"],
        "entrypoint_sha256": file_hash(Path(__file__)),
        "e3_plan_sha256": plan["sha256"],
        "scope": "E3_execution_development_engineering",
    }
    path = directory / "static-config.json"
    static_identity = {
        **identity,
        "config_path": str(path.relative_to(PROJECT)),
        "config_sha256": file_hash(path),
    }
    return TrainingData(source, train, validation, identity), SearchData(
        source, pools, static_identity
    )


def panel_list(settings, data):
    return training_panels(
        {
            "dimensions": [n for n, _ in settings.budgets],
            "evolution": asdict(settings.evolution),
            "training": asdict(settings),
        },
        {str(n): {"train": ids} for n, ids in data.training.items()},
    )


def static_configuration(plan, search, policies):
    return {
        "scope": "E3_execution_development_engineering",
        "search": asdict(search),
        "data_pools": {
            "search": {"offset": 0, "count": 32},
            "validation": {"offset": 32, "count": 16},
        },
        "solver": plan["config"]["solver"],
        "policies": [p.to_dict() for p in policies],
    }


def event(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def preserve_prefix(directory):
    state = load_checkpoint(directory / "checkpoint.json")
    require(state["pending"] is None and state["active_worker"] is None, "暂停点未关闭")
    require(state["costs"]["solve_jobs"] == 5, "没有停在第5个原生求解后")
    shutil.copyfile(directory / "checkpoint.json", directory / "paused-checkpoint.json")
    proof = {
        "completed": state["completed"],
        "raw_returns": {p.name: file_hash(p) for p in (directory / "raw_returns").glob("*.json")},
        "task_files": {p.name: file_hash(p) for p in (directory / "tasks").glob("*.json")},
    }
    atomic_json(directory / "prefix-proof.json", proof)


def check_prefix(directory, state):
    before = load_checkpoint(directory / "paused-checkpoint.json")
    proof = json.loads((directory / "prefix-proof.json").read_text())
    require(before["run_id"] == state["run_id"], "恢复运行身份改变")
    require(before["completed"] == proof["completed"], "暂停证明改变")
    require(
        all(state["completed"].get(k) == v for k, v in proof["completed"].items()),
        "已完成记录被重算或改写",
    )
    for folder, key in (("raw_returns", "raw_returns"), ("tasks", "task_files")):
        require(
            all(file_hash(directory / folder / name) == sha for name, sha in proof[key].items()),
            "恢复改写了暂停前不可变返回",
        )
    old = before["worker_history"]
    require(
        old == state["worker_history"][: len(old)] and len(state["worker_history"]) > len(old),
        "恢复没有新建worker",
    )
    return {
        "preserved_records": len(proof["completed"]),
        "preserved_raw_returns": len(proof["raw_returns"]),
        "paused_workers": len(old),
        "total_workers": len(state["worker_history"]),
    }


def run(args):
    directory, catalog = args.output.resolve(), args.catalog.resolve()
    require(directory.is_relative_to(PROJECT) and not directory.exists(), "需全新工作树输出")
    require(catalog.is_relative_to(PROJECT), "图必须在工作树")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "工程GPU必须固定UUID")
    plan, jobs = load_freeze(args.protocol.resolve(), args.database)
    require(len(jobs) == 2296, "需要完整冻结图计划")
    model, driver = [
        v.strip()
        for v in next(
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
    ]
    execution = {
        "catalog_path": str(catalog.relative_to(PROJECT)),
        "catalog_sha256": file_hash(catalog),
    }
    protocol = replace(
        worker_protocol(
            plan, execution, CONDITION, args.gpu_uuid, model, driver, socket.gethostname()
        ),
        maximum_registered_per_dimension=64,
    )
    settings, search, policies = settings_and_grid(plan)
    directory.mkdir(parents=True)
    atomic_json(directory / "static-config.json", static_configuration(plan, search, policies))
    info = {
        "scope": "engineering only; no formal training or TEST inference",
        "condition": CONDITION,
        "protocol_directory": str(args.protocol.resolve().relative_to(PROJECT)),
        "plan_sha256": plan["sha256"],
        "worker_protocol": protocol.manifest(),
        "execution_sources": execution_sources(),
        "engineering_entrypoint_sha256": file_hash(Path(__file__)),
        "training_settings": asdict(settings),
        "static_settings": asdict(search),
        "static_config_sha256": file_hash(directory / "static-config.json"),
        "wall_clock_limit": None,
        "formal_test_released": False,
    }
    atomic_json(directory / "run-info.json", info)
    with IndexedDataset(args.database, args.dataset_root) as source:
        training, static = data_for(source, plan, directory)
        expected = panel_list(settings, training)
        atomic_json(directory / "expected-panels.json", expected)
        for name, run_type, positional, keywords in (
            (
                "training",
                RegisteredGraphTrainingRun,
                (settings, protocol, training),
                {"expected_panels": expected},
            ),
            ("static", RegisteredGraphStaticRun, (search, protocol, policies, static), {}),
        ):
            target = directory / name
            result = run_type(target, *positional, **keywords, event=event).run(stop_after_tasks=5)
            require(result["status"] == "paused", "精确任务边界没有暂停")
            preserve_prefix(target)
            result = run_type(target, *positional, **keywords, event=event, resume=True).run()
            require(
                result["status"] == "complete" and result["costs"]["failed_solves"] == 0,
                "真实工程训练或验证失败；保留原运行等待排查",
            )
            event({"stage": name, "result": result})
    require(info["execution_sources"] == execution_sources(), "运行中执行源码改变")
    event({"status": "complete_pending_independent_audit", "directory": str(directory)})


def audit(args):
    directory = args.input.resolve()
    info = json.loads((directory / "run-info.json").read_text())
    resources = json.loads(args.resources.read_text())
    require(
        type(resources["exit_code"]) is int and resources["exit_code"] == 0, "CLI未实际成功终止"
    )
    require(
        info["execution_sources"] == execution_sources()
        and info["engineering_entrypoint_sha256"] == file_hash(Path(__file__)),
        "工程执行来源改变",
    )
    plan, _ = load_freeze(PROJECT / info["protocol_directory"], args.database)
    require(plan["sha256"] == info["plan_sha256"], "冻结图计划改变")
    settings, search, policies = settings_and_grid(plan)
    require(
        json_value(asdict(settings)) == info["training_settings"]
        and json_value(asdict(search)) == info["static_settings"],
        "工程规模改变",
    )
    config = directory / "static-config.json"
    require(
        file_hash(config) == info["static_config_sha256"]
        and json.loads(config.read_text())
        == json_value(static_configuration(plan, search, policies)),
        "工程Static配置改变",
    )
    p = info["worker_protocol"]
    protocol = WorkerProtocol(
        **{
            f.name: SolverSettings(**p[f.name]) if f.name == "settings" else p[f.name]
            for f in fields(WorkerProtocol)
            if f.init
        }
    )
    expected_protocol = replace(
        worker_protocol(
            plan,
            {"catalog_path": p["graph_catalog_path"], "catalog_sha256": p["graph_catalog_sha256"]},
            CONDITION,
            p["gpu_uuid"],
            p["gpu_model"],
            p["driver_version"],
            p["execution_host"],
        ),
        maximum_registered_per_dimension=64,
    )
    require(
        p == json_value(protocol.manifest()) == json_value(expected_protocol.manifest()),
        "实际批形状、图或设备身份改变",
    )
    proofs = {}
    for name in ("training", "static"):
        state = load_checkpoint(directory / name / "checkpoint.json")
        require(
            state["phase"] == "complete"
            and state["pending"] is None
            and state["active_worker"] is None
            and state["costs"]["failed_solves"] == 0,
            "尚未完整结束工程训练及验证",
        )
        require(
            state["worker_history"]
            and all(
                v["start_method"] == "spawn"
                and v["host"] == p["execution_host"]
                and v["protocol_sha256"] == protocol.sha256
                and v["binary_sha256"] == p["binary_sha256"]
                and v["device"]["name"] == p["gpu_model"]
                and v["constraint_mode"] == p["constraint_mode"]
                and v["graph_prior_kind"] == p["graph_prior_kind"]
                and v["graph_catalog_sha256"] == p["graph_catalog_sha256"]
                for v in state["worker_history"]
            ),
            "真实worker记录不符",
        )
        proofs[name] = check_prefix(directory / name, state)
    target = directory / "training"
    state = load_checkpoint(target / "checkpoint.json")
    with IndexedDataset(args.database, args.dataset_root) as source:
        training, _ = data_for(source, plan, directory)
        manifest = training_manifest(settings, protocol, training)
        require(
            manifest == state["manifest"] == load_checkpoint(target / "manifest.json")
            and content_hash(manifest) == state["run_id"],
            "GP来源或运行身份改变",
        )
        selection = load_checkpoint(target / "data_selection.json")
        require(
            selection
            == json_value(
                {
                    "training": training.training,
                    "validation": training.validation,
                    "identity": training.identity,
                }
            ),
            "工程数据选择改变",
        )
        panels = panel_list(settings, training)
        require(
            panels == json.loads((directory / "expected-panels.json").read_text()), "固定面板改变"
        )
        verify_panels(state, selection, settings, panels, True)
        allowed = {
            name
            for pool in (training.training, training.validation)
            for ids in pool.values()
            for name in ids
        }
        evaluated, totals, phases, occupancy, error = audit_receipts(
            target, state, protocol, settings, source, allowed, True
        )
        replay = replay_evolution(target, state, evaluated, protocol, settings, True)
    require(
        phases["training"]["solve_jobs"] == 48 and replay["completed_generations"] == 3,
        "GP工程演化不完整",
    )
    require(
        phases["validation"]["solve_jobs"] == replay["validation_candidates"] * 2,
        "GP验证未覆盖全部候选",
    )
    require(
        totals["search_tour_evaluations"] == totals["solve_jobs"] * 32 * 256
        and not any(occupancy.values()),
        "GP FE或设备观察异常",
    )
    gp_report = {
        "status": "passed",
        "run_id": state["run_id"],
        "costs": totals,
        "phases": phases,
        "replay": replay,
        "occupancy": occupancy,
        "max_cost_error": error,
        "worker_history": state["worker_history"],
        "recovery": proofs["training"],
    }
    target = directory / "static"
    static_report = summarize(
        target,
        target / "paused-checkpoint.json",
        database=args.database,
        dataset_root=args.dataset_root,
        entrypoint=Path(__file__),
    )
    static_state = load_checkpoint(target / "checkpoint.json")
    require(verify_raw_returns(target, static_state, protocol) == 20, "Static实际调用数不符")
    require(
        static_report["status"] == "passed"
        and static_report["phase_jobs"] == {"search": 16, "validation": 4}
        and static_report["costs"]["search_tour_evaluations"] == 20 * 32 * 256,
        "Static全配置搜索/完整验证不符",
    )
    require(
        all(
            static_report["gpu_occupancy"][key] == 0
            for key in ("before_foreign_samples", "after_foreign_samples", "after_unknown_samples")
        ),
        "Static设备观察异常",
    )
    report = {
        "status": "passed",
        "scope": info["scope"],
        "condition": CONDITION,
        "complete_training_audit": True,
        "complete_static_audit": True,
        "pause_resume_preserved": True,
        "execution_sources": execution_sources(),
        "protocol": p,
        "plan_sha256": plan["sha256"],
        "training": gp_report,
        "static": static_report,
        "recovery": proofs,
        "cli_resources": resources,
        "cli_resources_sha256": file_hash(args.resources),
        "run_info_sha256": file_hash(directory / "run-info.json"),
        "engineering_entrypoint_sha256": file_hash(Path(__file__)),
        "formal_test_released": False,
    }
    require(
        args.output.resolve().is_relative_to(PROJECT) and not args.output.exists(), "需全新审计产物"
    )
    atomic_json(args.output, report)
    event(
        {
            key: report[key]
            for key in (
                "status",
                "complete_training_audit",
                "complete_static_audit",
                "pause_resume_preserved",
            )
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    running = sub.add_parser("run")
    for name in ("protocol", "catalog", "database", "dataset-root", "output"):
        running.add_argument("--" + name, type=Path, required=True)
    running.add_argument("--gpu-uuid", required=True)
    checking = sub.add_parser("audit")
    for name in ("input", "database", "dataset-root", "resources", "output"):
        checking.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.stage == "run" else audit(args)


if __name__ == "__main__":
    main()
