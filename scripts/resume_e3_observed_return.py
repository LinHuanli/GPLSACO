#!/usr/bin/env python3
"""E3次数运行的资源偏差恢复：保存真实返回，隔离受干扰计时，不重复原生求解。"""

import argparse
import csv
import datetime
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from audit_e3_training import audit as audit_training  # noqa: E402
from gp_faco.checkpoint import atomic_json, load_checkpoint  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.e3_execution import (  # noqa: E402
    RegisteredGraphStaticRun,
    RegisteredGraphTrainingRun,
    make_static,
    make_training,
    worker_protocol,
)
from gp_faco.e3_protocol import require  # noqa: E402
from gp_faco.evaluation_run import EvaluationRun, json_value  # noqa: E402
from gp_faco.worker import content_hash, file_hash  # noqa: E402
from research_e3 import load_execution  # noqa: E402
from summarize_configuration_search import summarize  # noqa: E402

BOUNDARY_ERROR = "任务结束时GPU占用未知或有外来进程，原始结果已保留"


def resource_deviation(observation):
    if not isinstance(observation, dict) or "foreign_processes" not in observation:
        return False
    value = observation.get("foreign_processes")
    return value is None or (type(value) is int and value > 0)


def verify_deviation(directory, key, identity):
    marker = json.loads((directory / "resource_deviations" / f"{key}.json").read_text())
    original_path = directory / "resource_deviations/original" / f"{key}.json"
    raw_path = directory / "raw_returns" / f"{key}.json"
    record = load_checkpoint(directory / "tasks" / f"{key}.json")
    original = load_checkpoint(original_path)
    raw = json.loads(raw_path.read_text())
    require(
        marker["recovery_identity_sha256"] == content_hash(identity)
        and marker["run_id"] == identity["run_id"]
        and marker["key"] == key
        and marker["original_record_sha256"] == file_hash(original_path)
        and marker["raw_return_sha256"] == file_hash(raw_path),
        "资源偏差原始证据改变",
    )
    require(
        original["checked"] is None and original["evaluation_seconds"] == 0.0,
        "原始偏差记录不是待核验实际返回",
    )
    require(
        all(
            record[name] == original[name]
            for name in ("key", "kind", "description", "outcome", "attempts")
        ),
        "恢复改写了真实返回或任务",
    )
    require(
        key
        == content_hash(
            {
                "run_id": identity["run_id"],
                "kind": record["kind"],
                "description": record["description"],
            }
        ),
        "偏差任务身份不符",
    )
    require(
        raw == {k: v for k, v in record["outcome"].items() if k != "gpu_boundary_after"},
        "独立原始返回不符",
    )
    observation = record["outcome"].get("gpu_boundary_after", {})
    require(
        observation == marker["observation"] and resource_deviation(observation),
        "偏差观察被改写或并非资源偏差",
    )
    attempts = record["attempts"]
    require(
        attempts
        and attempts[-1]["status"] == "returned"
        and record["outcome"]["worker_pid"] == attempts[-1]["worker_pid"]
        and sum(a["status"] == "returned" for a in attempts) == 1,
        "不是唯一实际返回",
    )
    return marker


def verify_journals(directory, state, *, complete):
    identity = json.loads((directory / "resource_recovery.json").read_text())
    require(
        identity["run_id"] == state["run_id"]
        and identity["implementation_sha256"] == file_hash(Path(__file__))
        and identity["policy_sha256"] == content_hash(identity["policy"]),
        "恢复来源改变",
    )
    markers = {p.stem for p in (directory / "resource_deviations").glob("*.json")}
    observed, returned_keys = set(), set()
    for key in state["completed"]:
        record = load_checkpoint(directory / "tasks" / f"{key}.json")
        require(content_hash(record) == state["completed"][key], "已完成任务归档改变")
        returned = any(a["status"] == "returned" for a in record["attempts"])
        if returned:
            returned_keys.add(key)
            require("gpu_boundary_after" in record["outcome"], "实际返回缺少边界记录")
            raw = directory / "raw_returns" / f"{key}.json"
            require(
                json.loads(raw.read_text())
                == {k: v for k, v in record["outcome"].items() if k != "gpu_boundary_after"},
                "原始返回改变",
            )
            if resource_deviation(record["outcome"]["gpu_boundary_after"]):
                require(key in markers, "资源偏差没有独立处置记录")
                verify_deviation(directory, key, identity)
                observed.add(key)
    if complete:
        require(markers == observed, "终态有额外或尚未核验的偏差记录")
        require(
            {p.stem for p in (directory / "raw_returns").glob("*.json")} == returned_keys,
            "完整终态的原始返回集合不完整或存在额外返回",
        )
    else:
        for key in markers - observed:
            require(
                state["pending"] and state["pending"]["key"] == key,
                "快照偏差不属于已完成或唯一待处理任务",
            )
            verify_deviation(directory, key, identity)
    waits = state.get("resource_waits", [])
    require(
        all(
            w["task_key"] in state["completed"]
            or (state["pending"] and w["task_key"] == state["pending"]["key"])
            for w in waits
        ),
        "等待观察不属于已完成或当前待提交任务",
    )
    return {
        "policy_sha256": identity["policy_sha256"],
        "implementation_sha256": identity["implementation_sha256"],
        "completed_resource_deviations": len(observed),
        "pre_submission_wait_samples": len(waits),
        "task_keys": sorted(observed),
        "timing_class": "recorded_contention_or_unknown; exclude aggregate clean-speed comparison"
        if observed or waits
        else "no post-return deviation observed",
        "resource_recovery_file_sha256": file_hash(directory / "resource_recovery.json"),
    }


class ResourceRecoveryOperations:
    def __init__(self, *args, recovery_policy, idle_wait=time.sleep, **kwargs):
        require(
            recovery_policy["policy_spec_id"] == 1
            and recovery_policy["no_quality_dependent_recovery"]
            and recovery_policy["no_wall_clock_limit"]
            and not recovery_policy["formal_test_released"],
            "恢复政策不符",
        )
        self._recovery_policy, self._idle_wait = recovery_policy, idle_wait
        super().__init__(*args, **kwargs)
        self._recovery_identity = {
            "run_id": self.run_id,
            "policy": recovery_policy,
            "policy_sha256": content_hash(recovery_policy),
            "implementation_path": str(Path(__file__).relative_to(PROJECT)),
            "implementation_sha256": file_hash(Path(__file__)),
            "base_worker_protocol_sha256": self.protocol.sha256,
        }
        path = self.directory / "resource_recovery.json"
        if path.exists():
            require(json.loads(path.read_text()) == self._recovery_identity, "恢复政策或实现改变")
        else:
            atomic_json(path, self._recovery_identity)

    def _before_submit(self):
        # 等待只发生在尚未提交的任务上；GPU占用从不转换为FE截止或重抽seed。
        while True:
            try:
                observation = self._boundary(self.protocol, self.state["active_worker"]["pid"])
            except Exception as error:
                observation = {
                    "foreign_processes": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            count = observation.get("foreign_processes")
            if type(count) is int and count == 0:
                self.state.setdefault("admission", []).append(
                    {"task_key": self.state["pending"]["key"], **observation}
                )
                self._save()
                return
            require(count is None or (type(count) is int and count > 0), "无效GPU观察")
            # 未准入观察单独归档，不把未知占用伪装为已准入的零进程样本。
            self.state.setdefault("resource_waits", []).append(
                {"task_key": self.state["pending"]["key"], **observation}
            )
            self._save()
            self._event("waiting_for_idle_gpu_before_submission", observation=observation)
            self._idle_wait(1.0)

    def _journal(self, key):
        require(
            file_hash(Path(__file__)) == self._recovery_identity["implementation_sha256"],
            "运行中恢复实现改变",
        )
        task_path = self.directory / "tasks" / f"{key}.json"
        raw_path = self.directory / "raw_returns" / f"{key}.json"
        require(task_path.exists() and raw_path.exists(), "没有实际返回，禁止按偏差政策重新提交")
        record = load_checkpoint(task_path)
        require(
            key not in self.state["completed"] and record["checked"] is None,
            "首次偏差日志必须先于外部fitness核验",
        )
        observation = record["outcome"].get("gpu_boundary_after", {})
        require(resource_deviation(observation), "不是返回后的资源观察异常")
        original = self.directory / "resource_deviations/original" / f"{key}.json"
        original.parent.mkdir(parents=True, exist_ok=True)
        # 原始task字节另存，不覆盖；后续只允许原事务补充checked与evaluation_seconds。
        if original.exists():
            require(original.read_bytes() == task_path.read_bytes(), "原始偏差返回已经改变")
        else:
            with tempfile.NamedTemporaryFile(
                dir=original.parent, prefix=".original-return-"
            ) as stream:
                stream.write(task_path.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
                # 同目录硬链接原子发布完整字节；失败时不能留下半份original归档。
                os.link(stream.name, original)
        marker = {
            "run_id": self.run_id,
            "key": key,
            "recovery_identity_sha256": content_hash(self._recovery_identity),
            "original_record_sha256": file_hash(original),
            "raw_return_sha256": file_hash(raw_path),
            "observation": observation,
            "native_resubmission": False,
            "decision": "unchanged evaluator; resource observation never selects fitness",
            "timing_class": "contended_or_unknown",
        }
        atomic_json(self.directory / "resource_deviations" / f"{key}.json", marker)
        verify_deviation(self.directory, key, self._recovery_identity)

    def _operation(self, kind, description, submit, validate):
        key = content_hash({"run_id": self.run_id, "kind": kind, "description": description})
        marker = self.directory / "resource_deviations" / f"{key}.json"
        if marker.exists():
            verify_deviation(self.directory, key, self._recovery_identity)
            # 文件已经存在，原事务仅重读、核验与归账，不走submit分支。
            return EvaluationRun._operation(self, kind, description, submit, validate)
        try:
            return super()._operation(kind, description, submit, validate)
        except ValueError as error:
            if str(error) != BOUNDARY_ERROR:
                raise
            self._journal(key)
            return EvaluationRun._operation(self, kind, description, submit, validate)


class ResourceTrainingRun(ResourceRecoveryOperations, RegisteredGraphTrainingRun):
    pass


class ResourceStaticRun(ResourceRecoveryOperations, RegisteredGraphStaticRun):
    pass


def context(directory, execution_directory, database, policy_path):
    plan, execution, values = load_execution(execution_directory, database)
    policy = json.loads(policy_path.read_text())
    require(
        policy["base_execution_sha256"] == execution["sha256"]
        and policy["base_plan_sha256"] == plan["sha256"],
        "资源政策不属于原科学执行",
    )
    manifest = load_checkpoint(directory / "manifest.json")
    require(
        manifest["data"]["identity"]["e3_execution_sha256"] == execution["sha256"],
        "不是原冻结E3运行",
    )
    return plan, execution, values, policy, manifest


def run(args):
    directory = args.input.resolve()
    require(directory.is_relative_to(PROJECT) and directory.exists(), "须恢复既有工作树运行")
    plan, execution, values, policy, manifest = context(
        directory, args.execution.resolve(), args.database, args.policy
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "恢复须固定原GPU UUID")
    condition = manifest["data"]["identity"]["condition"]
    static = "purpose" in manifest["settings"]
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
        condition,
        args.gpu_uuid,
        model,
        driver,
        socket.gethostname(),
        static=static,
    )
    require(
        json_value(protocol.manifest()) == manifest["worker_protocol"],
        "不能静默迁移硬件、图或批形状",
    )
    with IndexedDataset(args.database, args.dataset_root) as source:
        for n in plan["config"]["dimensions"]:
            for role in ("train", "validation", "test"):
                require(
                    source.record_ids(role, n) == values["members.json"][str(n)][role],
                    "实际主split改变",
                )
            require(
                source.record_ids("development", n) == values["development.json"][str(n)],
                "实际开发split改变",
            )
        options = {
            "resume": True,
            "recovery_policy": policy,
            "event": lambda v: print(json.dumps(v, ensure_ascii=False), flush=True),
        }
        if static:
            settings, policies, data = make_static(
                source, plan, execution, values["development.json"], condition
            )
            instance = ResourceStaticRun(directory, settings, protocol, policies, data, **options)
        else:
            settings, data = make_training(
                source,
                plan,
                execution,
                values["members.json"],
                condition,
                manifest["settings"]["evolution_seed"],
            )
            instance = ResourceTrainingRun(
                directory,
                settings,
                protocol,
                data,
                expected_panels=values["training_panels.json"],
                **options,
            )
        require(json_value(asdict(settings)) == manifest["settings"], "不能缩减原完整研究规模")
        result = instance.run(stop_after_tasks=args.stop_after_tasks)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


def audit(args):
    directory = args.input.resolve()
    plan, execution, values, policy, manifest = context(
        directory, args.execution.resolve(), args.database, args.policy
    )
    state = load_checkpoint(directory / "checkpoint.json")
    identity = json.loads((directory / "resource_recovery.json").read_text())
    require(identity["policy"] == policy, "独立审计政策与实际恢复不同")
    deviations = verify_journals(directory, state, complete=not args.snapshot)
    if "purpose" not in manifest["settings"]:
        base = audit_training(
            directory,
            args.execution.resolve(),
            args.database,
            args.dataset_root,
            snapshot=args.snapshot,
            resources=args.resources,
        )
    else:
        require(not args.snapshot and args.resources is not None, "Static需要完整CLI终态审计")
        require(
            state["pending"] is None
            and state["active_worker"] is None
            and state["phase"] in ("complete", "failed"),
            "Static尚未完整结束",
        )
        cli = json.loads(args.resources.read_text())
        require(
            cli["exit_code"] == (0 if state["phase"] == "complete" else 1), "Static CLI终态不符"
        )
        condition, p = manifest["data"]["identity"]["condition"], manifest["worker_protocol"]
        protocol = worker_protocol(
            plan,
            execution,
            condition,
            p["gpu_uuid"],
            p["gpu_model"],
            p["driver_version"],
            p["execution_host"],
            static=True,
        )
        require(json_value(protocol.manifest()) == p, "Static冻结硬件/图改变")
        with IndexedDataset(args.database, args.dataset_root) as source:
            settings, policies, data = make_static(
                source, plan, execution, values["development.json"], condition
            )
            require(
                len(policies) == 160
                and data.identity == manifest["data"]["identity"]
                and json_value(data.pools) == manifest["data"]["pools"],
                "Static完整族或数据改变",
            )
        base = summarize(
            directory,
            database=args.database,
            dataset_root=args.dataset_root,
            entrypoint=PROJECT / "scripts/research_e3.py",
        )
        require(len(load_checkpoint(directory / "plan.json")["search"]) == 1280, "Static搜索缩减")
        if state["phase"] == "complete":
            require(
                len(state["validation_plan"]) == 40
                and state["costs"]["solve_jobs"] == 1320
                and set(state["selected"]) == {"static"},
                "Static验证或调用集合不完整",
            )
    report = {
        "status": "snapshot_passed" if args.snapshot else "passed",
        "scope": "explicit resource deviation amendment; native inputs/evaluator unchanged",
        "run_id": state["run_id"],
        "base_execution_sha256": execution["sha256"],
        "base_audit": base,
        "resource_deviations": deviations,
        "policy_file_sha256": file_hash(args.policy),
        "implementation_sha256": file_hash(Path(__file__)),
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal_test_released": False,
    }
    require(
        args.output.resolve().is_relative_to(PROJECT) and not args.output.exists(),
        "审计输出必须为工作树内全新产物",
    )
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": state["run_id"],
                "resource_deviations": deviations["completed_resource_deviations"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("run", "audit"):
        command = sub.add_parser(stage)
        for name in ("input", "execution", "database", "dataset-root", "policy"):
            command.add_argument("--" + name, type=Path, required=True)
        if stage == "run":
            command.add_argument("--gpu-uuid", required=True)
            command.add_argument("--stop-after-tasks", type=int)
        else:
            command.add_argument("--snapshot", action="store_true")
            command.add_argument("--resources", type=Path)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.stage == "run" else audit(args)


if __name__ == "__main__":
    main()
