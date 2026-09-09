"""完整面板评价的通用事务：同一Future等待、原始结果先落盘、终态后恢复。"""

from __future__ import annotations

import copy
import json
import math
import platform
import time
from concurrent.futures import TimeoutError
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.result_journal import ResultJournal


def json_value(value):
    """规范化tuple/list以便checkpoint恢复后比较身份；不允许NaN/Infinity。"""
    return json.loads(json.dumps(value, allow_nan=False))


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
        runtime_path = self.directory / "worker_runtime.json"
        if runtime_path.exists():
            active = load_checkpoint(runtime_path)["active"]
            if active and self._previous_worker_active(active):
                raise RuntimeError("上一 worker 仍可能运行，不能重复提交已在 GPU 执行的种群")
        self._worker = self._worker_factory(self.protocol)
        while True:
            try:
                runtime = self._worker.ready(timeout=5)
                break
            except TimeoutError:
                self._event("waiting_worker_startup")
        if (
            runtime["host"] != self.protocol.execution_host
            or runtime["protocol_id"] != self.protocol.identifier
        ):
            raise RuntimeError("worker实际身份与训练协议不符")
        self.state["active_worker"] = runtime
        self.state["worker_history"].append({"phase": self.state["phase"], **runtime})
        # 只在 worker 启停写小记录；代际大快照之外也能识别首代遗留的活动进程。
        save_checkpoint(runtime_path, {"active": runtime})
        self._save()
        self._event("worker_ready", pid=runtime["pid"])

    def _close_worker(self):
        if self._worker is not None:
            self._worker.close()  # shutdown等待实际终态，不能因观察超时杀进程。
            self._worker = None
            save_checkpoint(self.directory / "worker_runtime.json", {"active": None})
        self.state["active_worker"] = None

    def _wait(self, future):
        while True:
            try:
                return future.result(timeout=5)
            except TimeoutError:
                self.state["costs"]["observer_timeouts"] += 1
                self._event("waiting_same_future", operation=self.state["pending"]["key"])

    @staticmethod
    def operation_key(kind, description):
        return (
            description["occurrence_id"]
            if kind in ("solve", "population")
            else description["preparation_id"]
        )

    def _open_journal(self):
        self._journal = ResultJournal(self.directory / "results.jsonl")
        self._returned = {}
        self._population_scores = {}
        for offset, record in self._journal.trailing(self.state.get("journal_position", 0)):
            key = record["key"]
            self._returned[key] = offset
            if record["checked"] is not None and key not in self.state["completed"]:
                self._record_completed(record, offset)

    def _completed_record(self, key):
        location = self.state["completed"][key]
        if isinstance(location, (list, tuple)):
            return self._population_member(self._journal.read(location[0]), location[1])
        return self._journal.read(location)

    def _completed_score(self, key):
        location = self.state["completed"][key]
        if not isinstance(location, (list, tuple)):
            return self._completed_record(key)["checked"]
        offset, index = location
        if offset not in self._population_scores:
            self._population_scores[offset] = self._journal.read(offset)["checked"]
        # 只缓存已完成 occurrence 的紧凑评分；不反复解析整个种群的数千条路线。
        # 不按程序去重，也不会为新的代际 occurrence 跳过 FE。
        return self._population_scores[offset][index]["checked"]

    @staticmethod
    def _population_member(record, index):
        """一个完整种群批次只落盘一次；逐个体记录用普通字节位置和成员下标读取。"""
        description = record["description"]["members"][index]
        return {
            "key": description["occurrence_id"],
            "kind": "solve",
            "description": description,
            "outcome": record["outcome"]["members"][index],
            **record["checked"][index],
            "attempts": record["attempts"],
        }

    def _operation(self, kind, description, submit, validate):
        """完整结果追加一次；恢复时直接读取已评分结果，不重复求解或路线评分。"""
        key = self.operation_key(kind, description)
        if key in self.state["completed"]:
            return self._completed_record(key), False
        record = None
        if key in self._returned:
            record = self._journal.read(self._returned[key])
        if record is None:
            pending = {
                "key": key,
                "kind": kind,
                "description": json_value(description),
                "attempts": [],
            }
            self.state["pending"] = pending
            while True:
                self._ensure_worker()
                self._before_submit()
                attempt = {"status": "running", "worker_pid": self.state["active_worker"]["pid"]}
                pending["attempts"].append(attempt)
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
                    if len(pending["attempts"]) <= self.settings.infrastructure_retries:
                        continue
                    outcome = {
                        "status": "failed",
                        "error": "基础设施重试耗尽",
                        **description,
                        "task_id": key,
                    }
                    if kind == "population":
                        outcome = {
                            "members": [
                                {
                                    "status": "failed",
                                    "error": "基础设施重试耗尽",
                                    **member,
                                    "task_id": member["occurrence_id"],
                                }
                                for member in description["members"]
                            ]
                        }
                attempt.update(status="returned", observed_seconds=time.perf_counter() - started)
                break
            record = {
                "key": key,
                "kind": kind,
                "description": json_value(description),
                "outcome": outcome,
                "checked": None,
                "evaluation_seconds": 0.0,
                "attempts": copy.deepcopy(pending["attempts"]),
            }
        try:
            checked, evaluation_seconds = validate(record["outcome"])
        except BaseException:
            # 评分异常仍保留已经完成的原生返回，恢复时不能再次求解。
            self._returned[key] = self._journal.append(record)
            raise
        record.update(checked=checked, evaluation_seconds=evaluation_seconds)
        # 共享池已经逐片保存完整原始返回。评分成功后的训练日志只保留成本、工作量与引用，
        # 不再复制数千条tour；已评分恢复使用checked，原始路线可由raw_record定位。
        members = record["outcome"].get("members", [record["outcome"]])
        for member in members:
            if member.get("raw_record"):
                native = member.get("native_result", {})
                member["native_result"] = {**native, "items": [
                    {k: v for k, v in item.items() if k != "tour"} for item in native.get("items", [])]}
        offset = self._journal.append(record)
        self._record_completed(record, offset)
        return record, True

    def _record_completed(self, record, offset):
        key = record["key"]
        if key in self.state["completed"]:
            raise ValueError("重复归集任务")
        self.state["completed"][key] = offset
        self.state["pending"] = None
        if record["kind"] == "population":
            self._population_scores[offset] = record["checked"]
            # 整批返回和评分同处一行，崩溃后全部恢复，不能只保存已遍历到的几个个体。
            for index in range(len(record["checked"])):
                self._record_completed(self._population_member(record, index), [offset, index])
            self.state["costs"].setdefault("population_batches", 0)
            self.state["costs"]["population_batches"] += 1
            return
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

    def _before_submit(self):
        """具体实验可在任务边界记录资源/设备占用；不把观察等待当成算法截止。"""

    def _previous_worker_active(self, active):
        if getattr(self.settings, "portable_a5000", False):
            from gp_faco.remote import process_alive

            # 只在启动/恢复边界确认旧 worker 终态；失联时保守等待，不能重叠求解。
            return process_alive(active)
        return active["host"] != platform.node() or Path(f"/proc/{active['pid']}").exists()
