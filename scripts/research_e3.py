#!/usr/bin/env python3
"""冻结正式E3执行依赖并运行四条件完整GP/Static；无算法墙钟上限。"""

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_execution import (  # noqa: E402
    RegisteredGraphStaticRun,
    RegisteredGraphTrainingRun,
    make_static,
    make_training,
    static_config,
    worker_protocol,
)
from gp_faco.e3_preparation import checked_json, load_freeze  # noqa: E402
from gp_faco.e3_protocol import CONDITIONS, require  # noqa: E402
from gp_faco.graph_catalog import GraphCatalog, project_file  # noqa: E402
from gp_faco.worker import content_hash, file_hash  # noqa: E402


def execution_sources():
    paths = [
        *sorted((PROJECT / "python/gp_faco").glob("*.py")),
        Path(__file__),
        PROJECT / "scripts/audit_e3_training.py",
        PROJECT / "scripts/audit_e3_static.py",
        PROJECT / "scripts/check_e3_execution.py",
        PROJECT / "scripts/summarize_configuration_search.py",
        PROJECT / "scripts/search_baselines.py",
    ]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}


def freeze(args):
    plan, jobs = load_freeze(args.protocol.resolve(), args.database)
    audit_path, catalog_path = args.graph_audit.resolve(), args.catalog.resolve()
    require(
        audit_path.is_relative_to(PROJECT) and catalog_path.is_relative_to(PROJECT),
        "需要工作树内完整图审计",
    )
    audit = checked_json(audit_path)
    require(
        audit["status"] == "passed"
        and audit["instances"] == len(jobs) == 2296
        and audit["graphs"] == 4592
        and audit["plan_sha256"] == plan["sha256"]
        and audit["catalog_sha256"] == file_hash(catalog_path)
        and not audit["formal_test_released"],
        "完整非测试图尚未独立审计",
    )
    catalog = GraphCatalog(PROJECT, str(catalog_path.relative_to(PROJECT)), file_hash(catalog_path))
    require(set(catalog._entries) == {job["instance_id"] for job in jobs}, "完整图目录成员改变")
    checks_path = args.engineering_check.resolve()
    require(checks_path.is_relative_to(PROJECT), "执行通路验收须在工作树")
    checks = checked_json(checks_path)
    require(
        checks["status"] == "passed"
        and checks["complete_training_audit"]
        and checks["complete_static_audit"]
        and checks["pause_resume_preserved"],
        "新执行与流式审计尚未完成真实检查",
    )
    for name, digest in checks["execution_sources"].items():
        require(file_hash(project_file(PROJECT, name)) == digest, "已验收执行源码改变")
    require(checks["execution_sources"] == execution_sources(), "执行验收没有覆盖全部当前依赖")
    output = args.output.resolve()
    require(output.is_relative_to(PROJECT), "执行冻结必须在工作树")
    output.mkdir(parents=True, exist_ok=False)
    static_configs = {}
    for name, _, _ in CONDITIONS:
        path = output / "static-configs" / f"{name}.json"
        atomic_json(path, static_config(plan, name))
        static_configs[name] = {"path": str(path.relative_to(PROJECT)), "sha256": file_hash(path)}
    execution = {
        "execution_spec_id": 1,
        "status": "frozen_before_formal_fitness",
        "plan_sha256": plan["sha256"],
        "protocol_directory": str(args.protocol.resolve().relative_to(PROJECT)),
        "plan_file_sha256": file_hash(args.protocol / "plan.json"),
        "catalog_path": str(catalog_path.relative_to(PROJECT)),
        "catalog_sha256": file_hash(catalog_path),
        "graph_audit_path": str(audit_path.relative_to(PROJECT)),
        "graph_audit_sha256": file_hash(audit_path),
        "engineering_check_path": str(checks_path.relative_to(PROJECT)),
        "engineering_check_sha256": file_hash(checks_path),
        "entrypoint_sha256": file_hash(Path(__file__)),
        "sources": execution_sources(),
        "static_configs": static_configs,
        "training_runs": 20,
        "static_runs": 4,
        "wall_clock_limit": None,
        "hardware_policy": plan["config"]["execution"]["hardware"],
        "formal_test_released": False,
    }
    atomic_json(output / "execution.json", {**execution, "sha256": content_hash(execution)})
    print(
        json.dumps(
            {
                "status": execution["status"],
                "sha256": content_hash(execution),
                "training_runs": 20,
                "static_runs": 4,
            }
        ),
        flush=True,
    )


def load_execution(directory, database):
    execution = checked_json(directory / "execution.json")
    require(
        execution["sha256"] == content_hash({k: v for k, v in execution.items() if k != "sha256"}),
        "执行冻结内容摘要改变",
    )
    require(
        type(execution["execution_spec_id"]) is int
        and execution["execution_spec_id"] == 1
        and execution["wall_clock_limit"] is None
        and not execution["formal_test_released"],
        "正式执行协议不符",
    )
    require(
        execution["entrypoint_sha256"] == file_hash(Path(__file__))
        and execution["sources"] == execution_sources(),
        "冻结执行依赖改变，不能静默恢复",
    )
    protocol_directory = project_file(PROJECT, execution["protocol_directory"])
    require(
        file_hash(protocol_directory / "plan.json") == execution["plan_file_sha256"],
        "准备计划文件改变",
    )
    plan, _ = load_freeze(protocol_directory, database)
    require(plan["sha256"] == execution["plan_sha256"], "执行与准备计划不同")
    for prefix in ("catalog", "graph_audit", "engineering_check"):
        require(
            file_hash(project_file(PROJECT, execution[prefix + "_path"]))
            == execution[prefix + "_sha256"],
            f"已验收依赖改变: {prefix}",
        )
    values = {
        name: checked_json(protocol_directory / name, digest)
        for name, digest in plan["files"].items()
    }
    return plan, execution, values


def run(args):
    plan, execution, values = load_execution(args.execution.resolve(), args.database)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "正式运行须通过UUID固定可见GPU"
    )
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
    protocol = worker_protocol(
        plan,
        execution,
        args.condition,
        args.gpu_uuid,
        model,
        driver,
        socket.gethostname(),
        static=args.stage == "static",
    )
    output = args.output.resolve()
    require(output.is_relative_to(PROJECT), "正式结果必须在工作树")
    require(args.resume or not output.exists(), "已有运行目录必须显式恢复，不得新建同名运行")
    with IndexedDataset(args.database, args.dataset_root) as source:
        # 每次启动/恢复重新枚举ID，原数据行SHA在首次加载实例时由IndexedDataset核验。
        for n in plan["config"]["dimensions"]:
            for role in ("train", "validation", "test"):
                require(
                    source.record_ids(role, n) == values["members.json"][str(n)][role],
                    "实际split成员改变",
                )
            require(
                source.record_ids("development", n) == values["development.json"][str(n)],
                "开发成员改变",
            )

        def event(value):
            print(json.dumps(value, ensure_ascii=False), flush=True)

        if args.stage == "train":
            settings, data = make_training(
                source, plan, execution, values["members.json"], args.condition, args.evolution_seed
            )
            instance = RegisteredGraphTrainingRun(
                output,
                settings,
                protocol,
                data,
                expected_panels=values["training_panels.json"],
                resume=args.resume,
                event=event,
            )
        else:
            settings, policies, data = make_static(
                source, plan, execution, values["development.json"], args.condition
            )
            instance = RegisteredGraphStaticRun(
                output, settings, protocol, policies, data, resume=args.resume, event=event
            )
        # 可选停在精确任务边界用于保留与验收前缀；完整研究规模没有改变，恢复仍须完成全量。
        result = instance.run(stop_after_tasks=args.stop_after_tasks)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    frozen = sub.add_parser("freeze")
    for name in ("protocol", "graph-audit", "catalog", "engineering-check", "database", "output"):
        frozen.add_argument("--" + name, type=Path, required=True)
    for stage in ("train", "static"):
        command = sub.add_parser(stage)
        for name in ("execution", "database", "dataset-root", "output"):
            command.add_argument("--" + name, type=Path, required=True)
        command.add_argument("--gpu-uuid", required=True)
        command.add_argument("--condition", choices=[row[0] for row in CONDITIONS], required=True)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--stop-after-tasks", type=int)
        if stage == "train":
            command.add_argument("--evolution-seed", type=int, required=True)
    args = parser.parse_args()
    freeze(args) if args.stage == "freeze" else run(args)


if __name__ == "__main__":
    main()
