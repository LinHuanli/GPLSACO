#!/usr/bin/env python3
"""流式独立核验正式E3全部训练/验证与原seed演化；活动快照不冒充终态。"""

import argparse
import datetime
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_execution import make_training, training_identity, worker_protocol  # noqa: E402
from gp_faco.e3_protocol import require, training_settings  # noqa: E402
from gp_faco.training import json_value, training_manifest  # noqa: E402
from gp_faco.training_audit import audit_receipts, replay_evolution, verify_panels  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402
from research_e3 import load_execution  # noqa: E402


def audit(
    directory, execution_directory, database, dataset_root, *, snapshot=False, resources=None
):
    plan, execution, values = load_execution(execution_directory, database)
    state, manifest = (
        load_checkpoint(directory / name) for name in ("checkpoint.json", "manifest.json")
    )
    require(
        manifest == state["manifest"] and content_hash(manifest) == state["run_id"],
        "正式训练身份不同",
    )
    complete = not snapshot
    if complete:
        require(
            state["phase"] in ("complete", "failed")
            and state["active_worker"] is None
            and state["pending"] is None,
            "完整审计要求真实训练/验证终态",
        )
        require(resources is not None, "完整审计需要实际CLI终态资源收据")
        cli = json.loads(resources.read_text())
        require(
            type(cli["exit_code"]) is int
            and cli["exit_code"] == (0 if state["phase"] == "complete" else 1),
            "CLI终态与训练结果不同",
        )
    selection = load_checkpoint(directory / "data_selection.json")
    identity = selection["identity"]
    condition, seed = identity["condition"], manifest["settings"]["evolution_seed"]
    settings = training_settings(plan["config"], seed)
    require(
        identity == training_identity(plan, execution, condition),
        "正式训练数据、条件或执行来源改变",
    )
    require(json_value(asdict(settings)) == manifest["settings"], "未使用完整128×50及预登记FE配置")
    p = manifest["worker_protocol"]
    protocol = WorkerProtocol(
        **{
            f.name: SolverSettings(**p[f.name]) if f.name == "settings" else p[f.name]
            for f in fields(WorkerProtocol)
            if f.init
        }
    )
    expected = worker_protocol(
        plan,
        execution,
        condition,
        p["gpu_uuid"],
        p["gpu_model"],
        p["driver_version"],
        p["execution_host"],
    )
    require(
        json_value(protocol.manifest()) == p == json_value(expected.manifest()),
        "实际硬件/图/批形状协议改变",
    )
    for worker in state["worker_history"]:
        require(
            worker["start_method"] == "spawn"
            and worker["host"] == p["execution_host"]
            and worker["protocol_sha256"] == protocol.sha256
            and worker["binary_sha256"] == p["binary_sha256"]
            and worker["device"]["name"] == p["gpu_model"]
            and worker["constraint_mode"] == p["constraint_mode"]
            and worker["graph_prior_kind"] == p["graph_prior_kind"]
            and worker["graph_catalog_sha256"] == p["graph_catalog_sha256"],
            "真实worker与冻结图/设备不符",
        )
    allowed = set()
    with IndexedDataset(database, dataset_root) as source:
        for n in p["dimensions"]:
            for role, frozen_role in (("training", "train"), ("validation", "validation")):
                ids = selection[role][str(n)]
                require(
                    ids
                    == values["members.json"][str(n)][frozen_role]
                    == source.record_ids(frozen_role, n),
                    "未使用完整主split",
                )
                require(not allowed.intersection(ids), "训练/验证成员交叉")
                allowed.update(ids)
            require(
                not allowed.intersection(source.record_ids("development", n))
                and not allowed.intersection(values["members.json"][str(n)]["test"]),
                "正式GP使用了开发/测试成员",
            )
        _, data = make_training(source, plan, execution, values["members.json"], condition, seed)
        require(training_manifest(settings, protocol, data) == manifest, "软件或训练底座来源改变")
        verify_panels(state, selection, settings, values["training_panels.json"], complete)
        evaluated, totals, phases, observations, error = audit_receipts(
            directory, state, protocol, settings, source, allowed, complete
        )
    replay = replay_evolution(directory, state, evaluated, protocol, settings, complete)
    if complete:
        require(
            phases["training"]["solve_jobs"] == plan["training_native_calls_per_run"],
            "未覆盖全部6400个体位置",
        )
        require(
            phases["validation"]["solve_jobs"]
            == replay["validation_candidates"] * len(state["validation_panels"]),
            "未完成全部代冠军/最终种群的完整验证",
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
            "实际终态摘要改变",
        )
    return {
        "status": "passed" if complete else "snapshot_passed",
        "scope": "formal E3 training integrity; no TEST inference",
        "complete_training_and_validation_audit": complete,
        "training_outcome": state["phase"],
        "run_id": state["run_id"],
        "condition": condition,
        "evolution_seed": seed,
        "plan_sha256": plan["sha256"],
        "execution_sha256": execution["sha256"],
        "checkpoint_state_sha256": content_hash(state),
        "protocol": p,
        "settings": manifest["settings"],
        "work_by_phase": phases,
        "recomputed_costs": totals,
        "recorded_costs": state["costs"],
        "resource_observations": observations,
        "max_independent_cost_error": error,
        "replay": replay,
        "selected": state["selected"],
        "formal_test_released": False,
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "audit_sources": {
            str(path.relative_to(PROJECT)): file_hash(path)
            for path in (Path(__file__), PROJECT / "python/gp_faco/training_audit.py")
        },
        "cli_resources_sha256": file_hash(resources) if resources else None,
        "limits": [
            "Snapshot covers committed prefix only",
            "All failed solve positions retained",
            "FE excludes LS move checks; those counts remain separate",
            "Hardware timings remain stratified by model/driver",
            "User-supplied reference labels lack independent optimality certificates",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "execution", "database", "dataset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    require(
        args.input.resolve().is_relative_to(PROJECT)
        and args.output.resolve().is_relative_to(PROJECT),
        "审计产物必须在工作树",
    )
    require(not args.output.exists(), "审计输出必须全新，保留旧快照")
    report = audit(
        args.input.resolve(),
        args.execution.resolve(),
        args.database,
        args.dataset_root,
        snapshot=args.snapshot,
        resources=args.resources,
    )
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "run_id", "condition", "evolution_seed", "work_by_phase")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
