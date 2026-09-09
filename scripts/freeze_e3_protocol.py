#!/usr/bin/env python3
"""冻结E3正式参数、全部面板和成对图准备任务；不读取测试坐标或标签。"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_protocol import (  # noqa: E402
    preparation_jobs,
    require,
    static_policies,
    training_panels,
    validate_config,
)
from gp_faco.worker import content_hash, file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/e3_protocol_v1.json")
    parser.add_argument("--e1-freeze", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--reuse-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config, args.output, args.candidate_binary, args.reuse_catalog = (
        p.resolve() for p in (args.config, args.output, args.candidate_binary, args.reuse_catalog)
    )
    require(
        all(
            p.is_relative_to(PROJECT)
            for p in (args.config, args.output, args.candidate_binary, args.reuse_catalog)
        ),
        "冻结配置、工具、缓存与输出必须位于工作树",
    )
    config = json.loads(args.config.read_text())
    validate_config(config)
    e1_plan = json.loads((args.e1_freeze / "plan.json").read_text())
    require(
        e1_plan["sha256"] == content_hash({k: v for k, v in e1_plan.items() if k != "sha256"}),
        "原E1冻结身份不符",
    )
    members = json.loads((args.e1_freeze / "members.json").read_text())
    require(
        file_hash(args.e1_freeze / "members.json") == e1_plan["members_sha256"], "原split成员改变"
    )
    require(file_hash(args.database) == e1_plan["database_sha256"], "数据库身份改变")
    panels = training_panels(config, members)
    require(
        panels == json.loads((args.e1_freeze / "training_panels.json").read_text())
        and file_hash(args.e1_freeze / "training_panels.json") == e1_plan["training_panels_sha256"],
        "E3共同面板与原始独立面板流不符",
    )
    with IndexedDataset(args.database, args.dataset_root) as source:
        development = {}
        for n in config["dimensions"]:
            for role in ("train", "validation", "test"):
                require(
                    sorted(source.record_ids(role, n)) == members[str(n)][role],
                    "数据库split成员改变",
                )
            development[str(n)] = sorted(source.record_ids("development", n))
    jobs = preparation_jobs(config, members, panels, development)
    require(
        len(jobs) == 2296
        and [sum(j["dimension"] == n for j in jobs) for n in (500, 1000)] == [1146, 1150],
        "正式图准备任务数与50代/完整验证/Static调参不符",
    )
    checks = json.loads((PROJECT / "docs/reports/graph_worker_results.json").read_text())
    require(
        checks["status"] == "passed"
        and checks["training_totals"]["solve_jobs"] == 240
        and checks["worker_matrix"]["audit"]["solve_jobs"] == 40,
        "图worker/独立开发训练尚未验收",
    )
    native = PROJECT / "build/cuda/gp_faco_ext.so"
    require(file_hash(native) == checks["native_binary_sha256"], "已验收原生二进制改变")
    budget_path = PROJECT / "provenance/fe_budget_v1.json"
    budget = json.loads(budget_path.read_text())
    calibration = PROJECT / "docs/reports/baseline_fe_calibration_results.json"
    require(
        budget["primary_fe_per_colony"] == 4096
        and budget["additional_fe_per_colony"] == [1024, 16384]
        and file_hash(calibration) == budget["calibration_report_sha256"],
        "FE校准冻结身份改变",
    )
    build_path = args.candidate_binary.parent / "build-manifest.json"
    build = json.loads(build_path.read_text())
    require(
        build["exit_code"] == 0 and build["binary_sha256"] == file_hash(args.candidate_binary),
        "候选工具身份不符",
    )
    preparation_check_path = PROJECT / "artifacts/e3/preparation-check-v1/report.json"
    preparation_check = json.loads(preparation_check_path.read_text())
    require(
        preparation_check["status"] == "passed"
        and preparation_check["instances"] == 4
        and preparation_check["new_native_prior_calls"] == 4
        and preparation_check["resumed_receipts_unchanged"] == 4,
        "真实准备/进程恢复尚未通过",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.e1_freeze / "members.json", args.output / "members.json")
    shutil.copyfile(args.e1_freeze / "training_panels.json", args.output / "training_panels.json")
    atomic_json(args.output / "development.json", development)
    atomic_json(args.output / "preparation_jobs.json", jobs)
    atomic_json(
        args.output / "static_policies.json", [p.to_dict() for p in static_policies(config)]
    )
    plan = {
        "protocol_spec_id": 1,
        "status": "E3_design_and_graph_preparation_frozen; formal_execution_gate_pending",
        "config_path": str(args.config.relative_to(PROJECT)),
        "config_sha256": file_hash(args.config),
        "config": config,
        "source_e1_plan_sha256": e1_plan["sha256"],
        "database_sha256": file_hash(args.database),
        "split_sha256": e1_plan["split_sha256"],
        "native_binary_path": str(native.relative_to(PROJECT)),
        "native_binary_sha256": file_hash(native),
        "candidate_binary_path": str(args.candidate_binary.relative_to(PROJECT)),
        "candidate_binary_sha256": file_hash(args.candidate_binary),
        "candidate_build_manifest_sha256": file_hash(build_path),
        "reuse_catalog_path": str(args.reuse_catalog.relative_to(PROJECT)),
        "reuse_catalog_sha256": file_hash(args.reuse_catalog),
        "fe_decision_sha256": file_hash(budget_path),
        "calibration_report_sha256": file_hash(calibration),
        "graph_worker_report_sha256": file_hash(PROJECT / "docs/reports/graph_worker_results.json"),
        "preparation_engineering_report_sha256": file_hash(preparation_check_path),
        "files": {p.name: file_hash(p) for p in sorted(args.output.glob("*.json"))},
        "preparation_sources": {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in [
                Path(__file__),
                PROJECT / "scripts/prepare_e3_graphs.py",
                *[
                    PROJECT / "python/gp_faco" / name
                    for name in (
                        "e3_protocol.py",
                        "e3_preparation.py",
                        "candidate_prior.py",
                        "graph_matching.py",
                        "graph_catalog.py",
                        "worker.py",
                        "data.py",
                        "dataset_index.py",
                        "checkpoint.py",
                        "baseline_policy.py",
                        "evolution.py",
                        "training.py",
                        "program_ir.py",
                        "primitives.py",
                        "fitness.py",
                        "evaluation_run.py",
                    )
                ],
            ]
        },
        "preparation_instances": len(jobs),
        "preparation_prior_graphs": 2 * len(jobs),
        "training_runs": 20,
        "training_fitness_evaluations_per_run": 6400,
        "training_native_calls_per_run": 12800,
        "training_search_tour_evaluations_per_run": 1677721600,
        "total_training_search_tour_evaluations": 33554432000,
        "static_search_tasks_per_condition": 1280,
        "static_validation_tasks_per_condition": 40,
        "formal_test_released": False,
    }
    atomic_json(args.output / "plan.json", {**plan, "sha256": content_hash(plan)})
    print(
        json.dumps(
            {
                "status": plan["status"],
                "sha256": content_hash(plan),
                "preparation_instances": len(jobs),
                "training_runs": 20,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
