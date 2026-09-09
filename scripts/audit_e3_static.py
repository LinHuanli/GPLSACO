#!/usr/bin/env python3
"""独立重建E3 Static完整160配置搜索/固定验证，检查实际图与独立原始返回。"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_execution import make_static, worker_protocol  # noqa: E402
from gp_faco.e3_protocol import require  # noqa: E402
from gp_faco.evaluation_run import json_value  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402
from research_e3 import load_execution  # noqa: E402
from summarize_configuration_search import summarize  # noqa: E402


def verify_raw_returns(directory, state, protocol):
    """逐条比较独立返回；图的完整坐标身份另由通用任务重建与score_panel核对。"""
    raw = {p.stem: p for p in (directory / "raw_returns").glob("*.json")}
    actual = set()
    for key, digest in state["completed"].items():
        record = load_checkpoint(directory / "tasks" / f"{key}.json")
        require(content_hash(record) == digest, "原始任务收据改变")
        returned = any(a["status"] == "returned" for a in record["attempts"])
        if returned:
            require(
                key in raw
                and json.loads(raw[key].read_text())
                == {k: v for k, v in record["outcome"].items() if k != "gpu_boundary_after"},
                "独立原始返回不符",
            )
            require(
                record["outcome"].get("gpu_boundary_after", {}).get("foreign_processes") == 0,
                "完成Static结果具有污染或未知GPU观察",
            )
            actual.add(key)
        require(
            record["description"]["graph_inputs"]
            and all(
                row["prior_kind"] == protocol.graph_prior_kind
                for row in record["description"]["graph_inputs"]
            ),
            "Static任务缺少匹配图身份",
        )
    require(set(raw) == actual, "Static原始返回有遗漏或额外任务")
    return len(actual)


def audit(directory, execution_directory, database, dataset_root, resources):
    plan, execution, values = load_execution(execution_directory, database)
    manifest, state = (
        load_checkpoint(directory / name) for name in ("manifest.json", "checkpoint.json")
    )
    require(
        state["phase"] in ("complete", "failed")
        and state["pending"] is None
        and state["active_worker"] is None,
        "Static必须真实结束，不能把搜索前缀当完整结果",
    )
    cli = json.loads(resources.read_text())
    require(
        type(cli["exit_code"]) is int
        and cli["exit_code"] == (0 if state["phase"] == "complete" else 1),
        "Static CLI终态不同",
    )
    identity, p = manifest["data"]["identity"], manifest["worker_protocol"]
    condition = identity["condition"]
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
        static=True,
    )
    require(
        p == json_value(expected.manifest()) == json_value(protocol.manifest()),
        "Static图/硬件/批形状改变",
    )
    with IndexedDataset(database, dataset_root) as source:
        settings, policies, data = make_static(
            source, plan, execution, values["development.json"], condition
        )
        require(
            data.identity == identity and json_value(data.pools) == manifest["data"]["pools"],
            "Static数据/来源改变",
        )
        require(
            settings.purpose == "static_tuning" and len(policies) == 160, "Static未使用完整预登记族"
        )
    result = summarize(
        directory,
        database=database,
        dataset_root=dataset_root,
        entrypoint=PROJECT / "scripts/research_e3.py",
    )
    raw_count = verify_raw_returns(directory, state, protocol)
    require(
        len(load_checkpoint(directory / "plan.json")["search"]) == 1280, "Static搜索任务未覆盖全族"
    )
    if state["phase"] == "complete":
        require(
            len(state["validation_plan"]) == 40 and set(state["selected"]) == {"static"},
            "完整Static验证或选择改变",
        )
        require(state["costs"]["solve_jobs"] == 1320, "Static完整调用次数不同")
        if not state["costs"]["failed_solves"]:
            require(
                state["costs"]["search_tour_evaluations"] == 1320 * 32 * 4096, "Static完整FE不同"
            )
    return {
        "status": "passed",
        "scope": "formal E3 Static tuning integrity; no TEST inference",
        "training_outcome": state["phase"],
        "run_id": state["run_id"],
        "condition": condition,
        "plan_sha256": plan["sha256"],
        "execution_sha256": execution["sha256"],
        "protocol": p,
        "independent_summary": result,
        "raw_returns": raw_count,
        "selected": state["selected"],
        "formal_test_released": False,
        "cli_resources_sha256": file_hash(resources),
        "audit_sources": {
            str(path.relative_to(PROJECT)): file_hash(path)
            for path in (Path(__file__), PROJECT / "scripts/summarize_configuration_search.py")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "execution", "database", "dataset-root", "resources", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(
        args.input.resolve().is_relative_to(PROJECT)
        and args.output.resolve().is_relative_to(PROJECT),
        "审计产物必须在工作树",
    )
    require(not args.output.exists(), "审计输出必须全新")
    report = audit(
        args.input.resolve(),
        args.execution.resolve(),
        args.database,
        args.dataset_root,
        args.resources,
    )
    atomic_json(args.output, report)
    print(
        json.dumps({k: report[k] for k in ("status", "run_id", "condition", "raw_returns")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
