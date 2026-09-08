"""开发集基线配置搜索/次数曲线：完整面板、原始结果归档及可恢复选择。"""

from __future__ import annotations

import csv
import fcntl
import importlib.metadata
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.evaluation_run import EvaluationRun, json_value
from gp_faco.fitness import PanelFitness, aggregate_panels, score_panel
from gp_faco.worker import (
    PROJECT,
    BaselineTask,
    PersistentGpuWorker,
    WorkerProtocol,
    content_hash,
    file_hash,
    freeze_problems,
)


@dataclass(frozen=True)
class SearchSettings:
    purpose: str = "calibration"
    evaluation_limits: tuple[int, ...] = (0, 256, 1024, 4096, 16384)
    instances_per_panel: int = 16
    solver_seeds: tuple[int, ...] = (17, 29)
    order_seed: int = 90371
    validation_shortlist_per_kind: int = 10
    preparation_mode: str = "cached"
    experiment_mask: int = 0xFFFFFFFF
    infrastructure_retries: int = 1

    def __post_init__(self):
        object.__setattr__(self, "evaluation_limits", tuple(self.evaluation_limits))
        object.__setattr__(self, "solver_seeds", tuple(self.solver_seeds))
        if self.purpose not in ("calibration", "tuning") or (
            self.purpose == "tuning" and len(self.evaluation_limits) != 1
        ):
            raise ValueError("次数校准可有多个FE档，调参必须只有一个预定主档")
        if not self.evaluation_limits or len(set(self.evaluation_limits)) != len(
            self.evaluation_limits
        ):
            raise ValueError("FE档必须非空且不能重复")
        for value in (*self.evaluation_limits, *self.solver_seeds, self.order_seed):
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError("FE和seed必须是uint64整数")
        if not self.solver_seeds or len(set(self.solver_seeds)) != len(self.solver_seeds):
            raise ValueError("求解seed必须非空且不重复")
        if any(
            type(v) is not int or v < 1
            for v in (self.instances_per_panel, self.validation_shortlist_per_kind)
        ) or (
            self.preparation_mode not in ("cached", "end_to_end")
            or type(self.infrastructure_retries) is not int
            or not 0 <= self.infrastructure_retries <= 3
            or type(self.experiment_mask) is not int
            or not 0 < self.experiment_mask <= 0xFFFFFFFF
        ):
            raise ValueError("面板形状、准备、mask或重试配置无效")


class SearchData:
    """只从显式成员池获取坐标；标签由协调端在完整返回后读取。"""

    def __init__(self, source, pools: dict[str, dict[int, list[str]]], identity: dict):
        if set(pools) not in ({"search"}, {"search", "validation"}):
            raise ValueError("配置搜索需要search池，统一验证可另列validation池")
        self.source, self.identity = source, json_value(identity)
        self.pools = {
            role: {n: tuple(sorted(ids)) for n, ids in scales.items()}
            for role, scales in pools.items()
        }
        names = [name for scales in self.pools.values() for ids in scales.values() for name in ids]
        if (
            not self.identity
            or not names
            or any(type(name) is not str or not name for name in names)
            or len(set(names)) != len(names)
            or any(not ids for scales in self.pools.values() for ids in scales.values())
        ):
            raise ValueError("数据身份/成员缺失或search与validation有重叠")
        self._allowed, self._instances = set(names), {}

    def problems(self, ids):
        if not set(ids) <= self._allowed:
            raise ValueError("任务越过已冻结的数据池")
        for name in ids:
            if name not in self._instances:
                value = self.source.load_instance(name)
                if value.instance_id != name:
                    raise ValueError("数据源返回其他实例身份")
                self._instances[name] = value
        return freeze_problems(tuple(self._instances[name] for name in ids))

    def labels(self, task):
        return {p.instance_id: self.source.load_label(p.instance_id) for p in task.problems}


def gpu_boundary(protocol: WorkerProtocol, worker_pid: int) -> dict:
    """只检查目标UUID；外来进程数量进入记录，不公开他人的命令或PID。"""
    started = time.perf_counter()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    pids = {
        int(row[1])
        for row in csv.reader(apps.splitlines())
        if row and row[0].strip() == protocol.gpu_uuid
    }
    row = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={protocol.gpu_uuid}",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        )
    )
    return {
        "foreign_processes": len(pids - {worker_pid}),
        "memory_used_mib": int(row[0]),
        "utilization_percent": int(row[1]),
        "query_seconds": time.perf_counter() - started,
    }


class ConfigurationRun(EvaluationRun):
    def __init__(
        self,
        directory: Path,
        settings: SearchSettings,
        protocol: WorkerProtocol,
        policies: tuple[BaselinePolicy, ...],
        data: SearchData,
        *,
        resume=False,
        worker_factory=PersistentGpuWorker,
        boundary=gpu_boundary,
        event=None,
    ):
        self.directory = directory.resolve()
        if not self.directory.is_relative_to(PROJECT):
            raise ValueError("配置搜索输出必须位于GPLSACO内")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lease = (self.directory / ".run.lock").open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lease.close()
            raise RuntimeError("该配置搜索仍有协调进程，不能并发恢复") from None
        self.settings, self.protocol, self.data = settings, protocol, data
        self._worker_factory, self._boundary, self._event_callback = worker_factory, boundary, event
        self._worker, self._closed = None, False
        self.path = self.directory / "checkpoint.json"
        try:
            if not policies or any(type(p) is not BaselinePolicy for p in policies):
                raise ValueError("需要显式非空基线配置族")
            self.policies = {p.sha256: p for p in sorted(policies, key=lambda p: p.sha256)}
            if len(self.policies) != len(policies):
                raise ValueError("配置族含重复身份")
            for policy in policies:
                policy.validate_mask(settings.experiment_mask)
            if settings.instances_per_panel * len(settings.solver_seeds) != protocol.colonies:
                raise ValueError("配置搜索必须使用worker完整面板")
            if any(
                limit % protocol.settings.ants or limit // protocol.settings.ants > 0xFFFFFFFF
                for limit in settings.evaluation_limits
            ):
                raise ValueError("FE档必须为完整蚂蚁批次且不超过uint32批次")
            if settings.purpose == "tuning" and (
                "validation" not in data.pools or {p.kind for p in policies} != {"static", "rule"}
            ):
                raise ValueError("正式开发调参需要两类基线和独立统一验证池")
            self.panels = {}
            for role, scales in data.pools.items():
                if set(scales) != set(protocol.dimensions):
                    raise ValueError("数据池与worker规模不符")
                self.panels[role] = []
                for n in protocol.dimensions:
                    ids, width = scales[n], settings.instances_per_panel
                    if len(ids) % width or len(ids) > protocol.maximum_registered_per_dimension:
                        raise ValueError("成员不足完整面板或超出阶段缓存容量")
                    self.panels[role].extend(
                        {"dimension": n, "ids": list(ids[i : i + width])}
                        for i in range(0, len(ids), width)
                    )
            rng = random.Random(settings.order_seed)
            self.search_plan = []
            for limit in settings.evaluation_limits:
                order = list(self.policies)
                rng.shuffle(order)
                self.search_plan.extend(
                    {"policy_sha256": sha, "limit": limit, "panel": panel}
                    for sha in order
                    for panel in range(len(self.panels["search"]))
                )
            plan = {"panels": self.panels, "search": self.search_plan}
            self.manifest = json_value(
                {
                    "search_version": 1,
                    "settings": asdict(settings),
                    "worker_protocol": protocol.manifest(),
                    "data": {"pools": data.pools, "identity": data.identity},
                    "policies": [p.to_dict() for p in self.policies.values()],
                    "plan_sha256": content_hash(plan),
                    "software": {
                        "python": platform.python_version(),
                        "numpy": importlib.metadata.version("numpy"),
                    },
                    "sources": {
                        str(p.relative_to(PROJECT)): file_hash(p)
                        for p in sorted((PROJECT / "python/gp_faco").glob("*.py"))
                    },
                    "selection": "full grid; fixed validation; macro gap then policy hash per kind",
                    "budget_kind": "search_tour_evaluations",
                    "wall_clock_limit": None,
                    "gpu_occupancy": "task boundary samples; no continuous-monitoring claim",
                }
            )
            self.run_id = content_hash(self.manifest)
            if resume:
                self.state = load_checkpoint(self.path)
                if (
                    type(self.state.get("version")) is not int
                    or self.state["version"] != 1
                    or self.state.get("phase") not in ("search", "validation", "complete", "failed")
                    or type(self.state.get("cursor")) is not int
                    or self.state["cursor"] < 0
                ):
                    raise ValueError("未知配置搜索checkpoint版本、阶段或任务游标")
                if (
                    self.state["run_id"] != self.run_id
                    or load_checkpoint(self.directory / "manifest.json") != self.manifest
                ):
                    raise ValueError("恢复时配置、数据、源码、软件或硬件身份改变")
                if load_checkpoint(self.directory / "plan.json") != json_value(plan):
                    raise ValueError("冻结任务序列改变")
                active = self.state["active_worker"]
                if active is not None:
                    if active["host"] != platform.node() or Path(f"/proc/{active['pid']}").exists():
                        raise RuntimeError("原worker仍可能活动，保留原任务")
                    self.state["active_worker"] = None
                pending = self.state["pending"]
                if pending and pending["attempts"]:
                    last = pending["attempts"][-1]
                    if (
                        last["status"] == "running"
                        and not (self.directory / "tasks" / f"{pending['key']}.json").exists()
                    ):
                        last["status"] = "worker_terminated_unobserved"
                        self.state["costs"]["infrastructure_failures"] += 1
                        self.state["costs"]["unobserved_terminated_attempts"] += 1
            else:
                if any(p.name != ".run.lock" for p in self.directory.iterdir()):
                    raise FileExistsError("新搜索要求空目录；已有结果须显式resume")
                self.state = {
                    "version": 1,
                    "run_id": self.run_id,
                    "phase": "search",
                    "cursor": 0,
                    "pending": None,
                    "completed": {},
                    "active_worker": None,
                    "worker_history": [],
                    "search_scores": {},
                    "validation_scores": {},
                    "validation_plan": [],
                    "shortlist": [],
                    "selected": {},
                    "admission": [],
                    "costs": {
                        "solve_jobs": 0,
                        "preparation_jobs": 0,
                        "failed_solves": 0,
                        "valid_members": 0,
                        "worker_seconds": 0.0,
                        "evaluator_seconds": 0.0,
                        "charged_seconds": 0.0,
                        "native_actual_seconds": 0.0,
                        "search_tour_evaluations": 0,
                        "overrun_seconds": 0.0,
                        "infrastructure_failures": 0,
                        "unobserved_terminated_attempts": 0,
                        "observer_timeouts": 0,
                    },
                }
                save_checkpoint(self.directory / "manifest.json", self.manifest)
                save_checkpoint(self.directory / "plan.json", plan)
            self._save()
        except BaseException:
            self._lease.close()
            raise

    def _save(self):
        save_checkpoint(self.path, self.state)

    def _event(self, kind, **values):
        if self._event_callback:
            self._event_callback({"event": kind, "phase": self.state["phase"], **values})

    def _before_submit(self):
        observation = self._boundary(self.protocol, self.state["active_worker"]["pid"])
        self.state["admission"].append({"task_key": self.state["pending"]["key"], **observation})
        self._save()
        if observation["foreign_processes"]:
            raise RuntimeError("目标GPU出现外来计算进程，当前任务尚未提交，保留待办")

    def task(self, phase, index, job):
        panel = self.panels["search" if phase == "search" else "validation"][job["panel"]]
        problems = self.data.problems(panel["ids"])
        if any(p.dimension != panel["dimension"] for p in problems):
            raise ValueError("已读取问题与冻结面板规模不符")
        return BaselineTask(
            f"{self.run_id}:{phase}:case{index}",
            self.policies[job["policy_sha256"]],
            problems,
            tuple((p.instance_id, seed) for p in problems for seed in self.settings.solver_seeds),
            job["limit"],
            self.settings.preparation_mode,
            self.settings.experiment_mask,
        )

    def _score(self, task, outcome):
        started = time.perf_counter()
        checked = score_panel(task, self.protocol, outcome, self.data.labels(task))
        return json_value(asdict(checked)), time.perf_counter() - started

    def _submit(self, worker, task):
        future = worker.submit(task)

        # 仍通过通用事务等待原Future；返回后再取边界资源，不据观察超时另起任务。
        class ObservedFuture:
            def result(inner, timeout=None):
                outcome = future.result(timeout=timeout)
                try:
                    observation = self._boundary(self.protocol, self.state["active_worker"]["pid"])
                except Exception as error:
                    # 实际结果已返回；资源观察失败不能令它丢失并触发第二次求解。
                    observation = {
                        "foreign_processes": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                outcome["gpu_boundary_after"] = observation
                return outcome

        return ObservedFuture()

    def _aggregate(self, phase, plan):
        grouped = {}
        for index, job in enumerate(plan):
            task = self.task(phase, index, job)
            key = content_hash(
                {
                    "run_id": self.run_id,
                    "kind": "solve",
                    "description": task.manifest(self.protocol),
                }
            )
            record = load_checkpoint(self.directory / "tasks" / f"{key}.json")
            if content_hash(record) != self.state["completed"].get(key):
                raise ValueError("配置汇总的已完成任务记录改变")
            verified, _ = self._score(task, record["outcome"])
            if verified != record["checked"]:
                raise ValueError("配置汇总的原始结果未通过再次独立核验")
            result = PanelFitness(
                **{
                    **record["checked"],
                    "members": tuple(tuple(v) for v in record["checked"]["members"]),
                }
            )
            grouped.setdefault(f"{job['limit']}/{job['policy_sha256']}", []).append(
                (task, result, key)
            )
        output = {}
        for key, records in grouped.items():
            tasks, scores = tuple(v[0] for v in records), tuple(v[1] for v in records)
            fitness = aggregate_panels(tasks, self.protocol, scores, self.protocol.dimensions)
            scales = {}
            for n in self.protocol.dimensions:
                selected = [v for v in records if v[0].dimension == n]
                value = aggregate_panels(
                    tuple(v[0] for v in selected),
                    self.protocol,
                    tuple(v[1] for v in selected),
                    (n,),
                )
                scales[str(n)] = value if math.isfinite(value) else None
            output[key] = {
                "fitness": fitness if math.isfinite(fitness) else None,
                "per_dimension": scales,
                "task_keys": [v[2] for v in records],
            }
        return output

    def _finish_search(self):
        self.state["search_scores"] = self._aggregate("search", self.search_plan)
        if self.settings.purpose == "calibration":
            self.state["phase"] = "complete"
            return
        limit = self.settings.evaluation_limits[0]
        shortlist = []
        for kind in ("static", "rule"):
            ranked = sorted(
                (score["fitness"], sha)
                for sha, p in self.policies.items()
                if p.kind == kind
                and (score := self.state["search_scores"][f"{limit}/{sha}"])["fitness"] is not None
            )
            if not ranked:
                self.state["phase"] = "failed"
                return
            shortlist.extend(
                sha for _, sha in ranked[: self.settings.validation_shortlist_per_kind]
            )
        self.state["shortlist"] = shortlist
        self.state["validation_plan"] = [
            {"policy_sha256": sha, "limit": limit, "panel": panel}
            for sha in shortlist
            for panel in range(len(self.panels["validation"]))
        ]
        self._close_worker()
        self.state["phase"], self.state["cursor"] = "validation", 0

    def _finish_validation(self):
        self.state["validation_scores"] = self._aggregate(
            "validation", self.state["validation_plan"]
        )
        limit = self.settings.evaluation_limits[0]
        selected = {}
        for kind in ("static", "rule"):
            ranked = sorted(
                (score["fitness"], sha)
                for sha in self.state["shortlist"]
                if self.policies[sha].kind == kind
                and (score := self.state["validation_scores"][f"{limit}/{sha}"])["fitness"]
                is not None
            )
            if not ranked:
                self.state["phase"] = "failed"
                return
            fitness, sha = ranked[0]
            selected[kind] = {
                "policy": self.policies[sha].to_dict(),
                "policy_sha256": sha,
                "fitness": fitness,
                **self.state["validation_scores"][f"{limit}/{sha}"],
            }
        self.state["selected"], self.state["phase"] = selected, "complete"
        self._immutable_record(
            self.directory / "selected_baselines.json",
            {
                "run_id": self.run_id,
                "manifest": self.manifest,
                "selected": selected,
                "validation_scores_sha256": content_hash(self.state["validation_scores"]),
            },
        )

    def run(self, *, stop_after_tasks=None):
        if self._closed:
            raise RuntimeError("配置搜索句柄已关闭")
        if stop_after_tasks is not None and (
            type(stop_after_tasks) is not int or stop_after_tasks < 1
        ):
            raise ValueError("暂停任务数必须为正整数")
        completed_here = 0
        try:
            if self.state["phase"] == "complete":
                if self._aggregate("search", self.search_plan) != self.state["search_scores"]:
                    raise ValueError("已完成搜索的汇总改变")
                if (
                    self.settings.purpose == "tuning"
                    and self._aggregate("validation", self.state["validation_plan"])
                    != self.state["validation_scores"]
                ):
                    raise ValueError("已完成统一验证的汇总改变")
            while self.state["phase"] in ("search", "validation"):
                phase = self.state["phase"]
                plan = self.search_plan if phase == "search" else self.state["validation_plan"]
                while self.state["cursor"] < len(plan):
                    index = self.state["cursor"]
                    task = self.task(phase, index, plan[index])
                    record, new = self._operation(
                        "solve",
                        task.manifest(self.protocol),
                        lambda worker, t=task: self._submit(worker, t),
                        lambda outcome, t=task: self._score(t, outcome),
                    )
                    self.state["cursor"] += 1
                    self._save()
                    completed_here += int(new)
                    self._event(
                        "solve_completed",
                        completed=self.state["costs"]["solve_jobs"],
                        phase_completed=self.state["cursor"],
                        phase_total=len(plan),
                        failed=record["checked"]["error"] is not None,
                    )
                    if stop_after_tasks is not None and completed_here >= stop_after_tasks:
                        return self.summary("paused")
                if phase == "search":
                    self._finish_search()
                else:
                    self._finish_validation()
                self._save()
            report = self.summary(self.state["phase"])
            save_checkpoint(self.directory / "summary.json", report)
            return report
        finally:
            try:
                self._close_worker()
                self._save()
            finally:
                self._closed = True
                self._lease.close()

    def summary(self, status):
        return {
            "status": status,
            "run_id": self.run_id,
            "purpose": self.settings.purpose,
            "costs": self.state["costs"],
            "search_scores": self.state["search_scores"],
            "validation_scores": self.state["validation_scores"],
            "selected": self.state["selected"],
            "shortlist": self.state["shortlist"],
            "completed_records": len(self.state["completed"]),
        }
