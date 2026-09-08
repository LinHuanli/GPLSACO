"""完整面板评价的通用事务：同一Future等待、原始结果先落盘、终态后恢复。"""

from __future__ import annotations

import copy
import json
import math
import time
from concurrent.futures import TimeoutError
from concurrent.futures.process import BrokenProcessPool

from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.worker import content_hash


def json_value(value):
    """规范化tuple/list以便checkpoint恢复后比较身份；不允许NaN/Infinity。"""
    return json.loads(json.dumps(value, allow_nan=False))


def portable_outcome(value):
    """非法非有限返回保留显式标记，外部核验判失败；JSON仍严格禁止NaN/Infinity。"""
    if type(value) is float and not math.isfinite(value):
        return {"invalid_float": repr(value)}
    if isinstance(value, dict):
        return {key: portable_outcome(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_outcome(item) for item in value]
    return value


class EvaluationRun:
    """共享评价事务；具体研究状态机提供_save、_event和明确的任务序列。"""

    def _immutable_record(self, path, value):
        if path.exists():
            if load_checkpoint(path) != json_value(value):
                raise ValueError("不可变的程序或代际归档发生变化")
        else:
            save_checkpoint(path, value)

    def _ensure_worker(self):
        if self._worker is not None:
            return
        self._worker = self._worker_factory(self.protocol)
        while True:
            try:
                runtime = self._worker.ready(timeout=5)
                break
            except TimeoutError:
                self._event("waiting_worker_startup")
        if (
            runtime["host"] != self.protocol.execution_host
            or runtime["protocol_sha256"] != self.protocol.sha256
        ):
            raise RuntimeError("worker实际身份与训练协议不符")
        self.state["active_worker"] = runtime
        self.state["worker_history"].append({"phase": self.state["phase"], **runtime})
        self._save()
        self._event("worker_ready", pid=runtime["pid"])

    def _close_worker(self):
        if self._worker is not None:
            self._worker.close()  # shutdown等待实际终态，不能因观察超时杀进程。
            self._worker = None
        self.state["active_worker"] = None

    def _wait(self, future):
        while True:
            try:
                return future.result(timeout=5)
            except TimeoutError:
                self.state["costs"]["observer_timeouts"] += 1
                self._event("waiting_same_future", operation=self.state["pending"]["key"])

    def _operation(self, kind, description, submit, validate):
        """先记录任务，再提交；结果先原子落盘，再更新完成表，允许恢复中间窗口。"""
        key = content_hash({"run_id": self.run_id, "kind": kind, "description": description})
        path = self.directory / "tasks" / f"{key}.json"
        pending = self.state["pending"]
        if pending is not None and pending["key"] != key:
            # 已完成的前置任务可重读，不能跳过真正未完成的位置。
            if key not in self.state["completed"]:
                raise ValueError("当前任务与checkpoint待完成身份不符")
        if path.exists():
            record = load_checkpoint(path)
            if (
                record["key"] != key
                or record["kind"] != kind
                or record["description"] != json_value(description)
            ):
                raise ValueError("任务产物身份不符")
            checked, evaluation_seconds = validate(record["outcome"])
            if record["checked"] is None and key not in self.state["completed"]:
                record.update(checked=checked, evaluation_seconds=evaluation_seconds)
                save_checkpoint(path, record)
            if checked != record["checked"]:
                raise ValueError("任务产物与重新独立核验结果不符")
            if key in self.state["completed"]:
                if self.state["completed"][key] != content_hash(record):
                    raise ValueError("已完成任务产物摘要改变")
                return record, False
            self._record_completed(record)
            return record, True
        if key in self.state["completed"]:
            raise FileNotFoundError("已完成任务产物丢失，不能悄悄重新评价")
        if pending is None:
            pending = {
                "key": key,
                "kind": kind,
                "description": json_value(description),
                "attempts": [],
            }
            self.state["pending"] = pending
            self._save()
        elif pending["description"] != json_value(description):
            raise ValueError("待完成任务内容改变")
        while True:
            if len(pending["attempts"]) > self.settings.infrastructure_retries:
                outcome = {"status": "failed", "error": "基础设施重试耗尽；保留完整任务失败"}
                if kind == "solve":
                    outcome.update(
                        {
                            name: description[name]
                            for name in (
                                "protocol_sha256",
                                "program_sha256",
                                "baseline_policy_sha256",
                                "baseline_policy",
                                "controller_kind",
                                "occurrence_id",
                                "dimension",
                            )
                            if name in description
                        }
                    )
                    outcome["task_id"] = content_hash(description)
                break
            self._ensure_worker()
            self._before_submit()
            attempt = {"status": "running", "worker_pid": self.state["active_worker"]["pid"]}
            pending["attempts"].append(attempt)
            self._save()
            started = time.perf_counter()
            try:
                outcome = self._wait(submit(self._worker))
            except BrokenProcessPool as error:
                attempt.update(
                    status="broken_process_pool",
                    error=str(error),
                    observed_seconds=time.perf_counter() - started,
                )
                self.state["costs"]["infrastructure_failures"] += 1
                self._close_worker()
                self._save()
                self._event("confirmed_worker_failure", attempt=len(pending["attempts"]))
                continue
            attempt.update(status="returned", observed_seconds=time.perf_counter() - started)
            break  # 普通求解失败也是一个已完成返回，绝不挑好结果重试。
        # 先保存真实返回，再核验；无效准备/标签导致核验抛错时也不能把普通返回当作崩溃重试。
        outcome = portable_outcome(outcome)
        record = {
            "key": key,
            "kind": kind,
            "description": json_value(description),
            "outcome": outcome,
            "checked": None,
            "evaluation_seconds": 0.0,
            "attempts": copy.deepcopy(pending["attempts"]),
        }
        save_checkpoint(path, record)
        checked, evaluation_seconds = validate(outcome)
        record.update(checked=checked, evaluation_seconds=evaluation_seconds)
        save_checkpoint(path, record)
        self._record_completed(record)
        return record, True

    def _record_completed(self, record):
        key = record["key"]
        if key in self.state["completed"]:
            raise ValueError("重复归集任务")
        self.state["completed"][key] = content_hash(record)
        self.state["pending"] = None
        costs = self.state["costs"]
        costs["worker_seconds"] += record["outcome"].get("worker_seconds", 0.0)
        costs["evaluator_seconds"] += record["evaluation_seconds"]
        costs.setdefault("registration_seconds", 0.0)
        costs["registration_seconds"] += record["outcome"].get("registration_seconds", 0.0)
        if record["kind"] == "preparation":
            costs["preparation_jobs"] += 1
        else:
            costs["solve_jobs"] += 1
            costs["failed_solves"] += int(record["checked"]["error"] is not None)
            costs["valid_members"] += len(record["checked"]["members"])
            native = record["outcome"].get("native_result", {})
            evaluations = native.get("total_tour_evaluations", 0)
            if type(evaluations) is int and evaluations >= 0:
                costs["search_tour_evaluations"] += evaluations
            for field, origin in (
                ("charged_seconds", "charged_seconds"),
                ("native_actual_seconds", "actual_seconds"),
                ("overrun_seconds", "overrun_seconds"),
            ):
                value = native.get(origin, 0.0)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    costs[field] += value
            for field in (
                "completed_batches",
                "completed_construction_steps",
                "completed_ls_evaluations",
            ):
                costs.setdefault(field, 0)
                value = native.get(field, 0)
                if type(value) is int and value >= 0:
                    costs[field] += value
        self._save()

    def _before_submit(self):
        """具体实验可在任务边界记录资源/设备占用；不把观察等待当成算法截止。"""
