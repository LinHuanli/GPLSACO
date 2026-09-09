"""显式spawn的常驻单GPU执行器；只接收无标签坐标、IR和固定形状任务。"""

from __future__ import annotations

import csv
import importlib
import math
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.data import Instance
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.graph_catalog import GraphCatalog
from gp_faco.program_ir import Program

PROJECT = Path(__file__).resolve().parents[2]


def faco_ants(dimension: int) -> int:
    """2022 论文 §5.6 的向上取整公式，整数实现避免平方根边界误差。"""
    if type(dimension) is not int or dimension < 3:
        raise ValueError("TSP 规模必须为至少 3 的整数")
    return 64 * (math.isqrt(dimension - 1) // 16 + 1)


def preparation_id(problems) -> str:
    return f"prepare-{problems[0].dimension}-{problems[0].instance_id}"


def freeze_problems(values) -> tuple[Instance, ...]:
    problems = []
    for value in values:
        if type(value) is not Instance:
            raise TypeError("worker问题必须是无标签Instance")
        if type(value.instance_id) is not str or not value.instance_id:
            raise ValueError("实例身份必须为非空字符串")
        problems.append(value)
    problems.sort(key=lambda p: p.instance_id)
    ids = [problem.instance_id for problem in problems]
    if not ids or len(set(ids)) != len(ids) or len({p.dimension for p in problems}) != 1:
        raise ValueError("面板必须包含同规模且身份不同的问题")
    return tuple(problems)


@dataclass(frozen=True)
class SolverSettings:
    ants: int = 0
    primary_width: int = 16
    backup_width: int = 64
    ls_width: int = 20
    beta: float = 1.0
    retention: float = 0.5
    p_best: float = 0.1
    epoch_source_probability: float = 0.01
    ls_evaluation_limit: int = 100000
    initial_ls_evaluation_limit: int = 100000

    def __post_init__(self) -> None:
        for name in (
            "ants",
            "primary_width",
            "backup_width",
            "ls_width",
            "ls_evaluation_limit",
            "initial_ls_evaluation_limit",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"{name}必须是非负整数")
        if not 0 <= self.ants <= 4096 or self.primary_width < 2 or self.ls_width < 1:
            raise ValueError("蚂蚁数量或候选宽度无效")
        for name in ("primary_width", "backup_width", "ls_width"):
            if getattr(self, name) > 0xFFFFFFFF:
                raise ValueError("候选宽度超出uint32")
        for name in ("beta", "retention", "p_best", "epoch_source_probability"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name}必须是有限数值")
        if not (
            self.beta > 0
            and 0 <= self.retention < 1
            and 0 < self.p_best < 1
            and 0 <= self.epoch_source_probability <= 1
        ):
            raise ValueError("信息素参数范围无效")

    def ants_for(self, dimension: int) -> int:
        return self.ants or faco_ants(dimension)


@dataclass(frozen=True)
class WorkerProtocol:
    gpu_uuid: str
    gpu_model: str
    driver_version: str
    build_id: str = "v2-exact"
    numeric_backend: str = field(default="exact", kw_only=True)
    dimensions: tuple[int, ...] = (500, 1000)
    colonies: int = 32
    settings: SolverSettings = field(default_factory=SolverSettings)
    extension_directory: str = "build/cuda"
    maximum_registered_per_dimension: int = 1024
    execution_host: str = field(default_factory=platform.node)
    constraint_mode: str = "unrestricted"
    graph_prior_kind: str | None = None
    graph_catalog_path: str | None = None
    distributed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        if not (self.distributed and self.gpu_uuid == "pool") and not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", self.gpu_uuid):
            raise ValueError("需要完整GPU UUID")
        if not self.gpu_model or not self.driver_version:
            raise ValueError("必须显式指定同型号GPU和driver")
        if self.numeric_backend not in ("exact", "fp32", "fp32_fast"):
            raise ValueError("未知数值后端")
        if self.gpu_model != "NVIDIA RTX A5000":
            raise ValueError("当前实验只允许 RTX A5000")
        if (
            not self.dimensions
            or len(set(self.dimensions)) != len(self.dimensions)
            or any(type(n) is not int or not 3 <= n <= 10000 for n in self.dimensions)
        ):
            raise ValueError("规模必须是不同的3..10000整数")
        if type(self.colonies) is not int or not 1 <= self.colonies <= 128:
            raise ValueError("固定colony数超出范围")
        if (
            type(self.maximum_registered_per_dimension) is not int
            or self.maximum_registered_per_dimension < self.colonies
        ):
            raise ValueError("静态缓存容量必须容纳一个完整面板")
        if type(self.settings) is not SolverSettings:
            raise TypeError("需要显式SolverSettings")
        directory = (PROJECT / self.extension_directory).resolve()
        if not directory.is_relative_to(PROJECT):
            raise ValueError("扩展必须位于GPLSACO内")
        if self.constraint_mode not in ("unrestricted", "hard", "escape"):
            raise ValueError("未知图限制模式")
        catalog = None
        if self.constraint_mode == "unrestricted":
            if any(v is not None for v in (self.graph_prior_kind, self.graph_catalog_path)):
                raise ValueError("普通worker不能携带图先验或图目录")
        else:
            if self.graph_prior_kind not in ("ALPHA", "POPMUSIC"):
                raise ValueError("图worker必须固定一个先验类型")
            catalog = GraphCatalog(PROJECT, self.graph_catalog_path)
            catalog.require_settings(self.settings)
            if self.constraint_mode == "escape" and (
                self.settings.primary_width > 16
                or self.settings.backup_width > 64
                or self.settings.ls_width > 20
            ):
                raise ValueError("Escape超出已验证物理槽位上限")
        # 图目录只读取一次；任务记录实例编号与实际图参数。
        object.__setattr__(self, "_graph_catalog", catalog)

    @property
    def graph_catalog(self) -> GraphCatalog | None:
        return self._graph_catalog

    def graph_identity(self, problems) -> dict:
        if self.graph_catalog is None:
            return {}
        return {"graph_inputs": self.graph_catalog.describe(problems, self.graph_prior_kind)}

    def manifest(self) -> dict:
        return {
            **asdict(self),
            "protocol_version": 6,
            "local_search_spec": "checklist_2opt_identity_zero_v2",
            "preparation_fee_policy": "wall_clock_charged_or_count_mode_resource_only",
            "instance_key_policy": "dataset_record_ordinal_v2",
            "fitness_spec_id": "reference_gap_instance_then_scale_macro_v1",
            "submission_boundary": "whole_panel_returned_final_tours_scored_once",
            "gpu_lease_policy": "git_common_root_uuid_lock_v1",
            **(
                {
                    "graph_spec_id": 1,
                    "matching_spec_id": 2,
                    "escape_spec_id": 1 if self.constraint_mode == "escape" else 0,
                }
                if self.graph_catalog is not None
                else {}
            ),
        }

    @property
    def identifier(self) -> str:
        return f"{self.build_id}-{self.constraint_mode}-{self.graph_prior_kind or 'knn'}"


def _normalize_task_members(task):
    if type(task.occurrence_id) is not str or not task.occurrence_id:
        raise ValueError("每个实际评价位置必须有独立身份")
    object.__setattr__(task, "problems", freeze_problems(task.problems))
    object.__setattr__(task, "replicas", tuple(tuple(row) for row in task.replicas))
    ids = [problem.instance_id for problem in task.problems]
    if any(
        len(row) != 2
        or type(row[0]) is not str
        or type(row[1]) is not int
        or not 0 <= row[1] <= 0xFFFFFFFFFFFFFFFF
        for row in task.replicas
    ):
        raise ValueError("replica需要实例身份和uint64 solve seed")
    if len(set(task.replicas)) != len(task.replicas) or {row[0] for row in task.replicas} != set(
        ids
    ):
        raise ValueError("replica有遗漏、未知实例或重复实例/seed")


def _task_manifest(task, protocol):
    if task.dimension not in protocol.dimensions or len(task.replicas) != protocol.colonies:
        raise ValueError("任务与worker固定规模/形状不符")
    if task.evaluation_limit_per_colony is not None and (
        task.evaluation_limit_per_colony % protocol.settings.ants_for(task.dimension)
        or task.evaluation_limit_per_colony // protocol.settings.ants_for(task.dimension)
        > 0xFFFFFFFF
    ):
        raise ValueError("次数限额必须为ants整批且批次数不超出uint32")
    if protocol.graph_catalog is not None and task.evaluation_limit_per_colony is None:
        raise ValueError("图worker只接受无时间上限的evaluation-count任务")
    return {
        "occurrence_id": task.occurrence_id,
        **task.controller_identity(),
        "protocol_id": protocol.identifier,
        "dimension": task.dimension,
        "problems": [p.instance_id for p in task.problems],
        **protocol.graph_identity(task.problems),
        "replicas": task.replicas,
        "budget_seconds": task.budget_seconds,
        "evaluation_limit_per_colony": task.evaluation_limit_per_colony,
        "preparation_mode": task.preparation_mode,
        "experiment_mask": task.experiment_mask,
        "preparation_charges": task.preparation_charges,
        **({"behavior_spec_id": 1} if type(task) is FactorialTask and task.record_behavior else {}),
    }


@dataclass(frozen=True)
class SolveTask:
    occurrence_id: str
    program: Program
    problems: tuple[Instance, ...]
    replicas: tuple[tuple[str, int], ...]
    budget_seconds: float | None = None
    preparation_mode: str = "cached_charged"
    experiment_mask: int = 0xFFFFFFFF
    preparation_charges: tuple[tuple[str, float, float], ...] | None = None
    evaluation_limit_per_colony: int | None = None

    def __post_init__(self) -> None:
        if type(self.program) is not Program:
            raise TypeError("任务只接受验证后的Program")
        _normalize_task_members(self)
        ids = [problem.instance_id for problem in self.problems]
        counted = self.evaluation_limit_per_colony is not None
        if counted:
            if (
                type(self.evaluation_limit_per_colony) is not int
                or not 0 <= self.evaluation_limit_per_colony <= 4096 * 0xFFFFFFFF
                or self.budget_seconds is not None
                or self.program.feature_spec_id != 2
                or self.preparation_mode not in ("cached", "end_to_end")
                or self.preparation_charges is not None
            ):
                raise ValueError("次数任务需要整数FE限额、特征v2，不能附带时间上限或扣费")
        elif (
            type(self.budget_seconds) not in (int, float)
            or not math.isfinite(self.budget_seconds)
            or self.budget_seconds < 0
            or self.program.feature_spec_id != 1
        ):
            raise ValueError("时间任务需要非负有限秒数与特征v1")
        if not counted and self.preparation_mode not in ("cached_charged", "end_to_end"):
            raise ValueError("未知准备模式")
        if self.preparation_charges is not None:
            charges = tuple(tuple(row) for row in self.preparation_charges)
            if self.preparation_mode != "cached_charged" or any(
                len(row) != 3
                or type(row[0]) is not str
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in row[1:]
                )
                for row in charges
            ):
                raise ValueError("冻结费用必须为cached_charged任务中的有限非负两阶段费用")
            if len(charges) != len(ids) or {row[0] for row in charges} != set(ids):
                raise ValueError("冻结费用必须精确覆盖全部问题，不能重复或遗漏")
            try:
                total = math.fsum(value for row in charges for value in row[1:])
            except OverflowError:
                raise ValueError("冻结费用总和超出有限范围") from None
            if not math.isfinite(total):
                raise ValueError("冻结费用总和超出有限范围")
            object.__setattr__(
                self,
                "preparation_charges",
                tuple(
                    sorted(
                        (name, float(cheap) + 0.0, float(preparation) + 0.0)
                        for name, cheap, preparation in charges
                    )
                ),
            )
        if (
            type(self.experiment_mask) is not int
            or not 0 < self.experiment_mask <= 0xFFFFFFFF
            or not self.experiment_mask & 0xFFFF
        ):
            raise ValueError("mask必须保留至少一个保持模式动作")

    @property
    def dimension(self) -> int:
        return self.problems[0].dimension

    @property
    def controller_id(self) -> str:
        return self.program.identifier

    def controller_identity(self) -> dict:
        return {"program_id": self.program.identifier}

    def manifest(self, protocol: WorkerProtocol) -> dict:
        return _task_manifest(self, protocol)

    def task_id(self, protocol: WorkerProtocol) -> str:
        return self.occurrence_id


@dataclass(frozen=True)
class FactorialTask(SolveTask):
    """同时绑定程序与M00规则，不能用相同IR冒充另一格的评价。"""

    factorial_policy: FactorialPolicy = field(kw_only=True)
    record_behavior: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if self.evaluation_limit_per_colony is None:
            raise ValueError("析因任务只允许评价次数预算")
        if type(self.factorial_policy) is not FactorialPolicy:
            raise TypeError("析因任务需要已验证的FactorialPolicy")
        self.factorial_policy.validate_mask(self.experiment_mask)
        if type(self.record_behavior) is not bool:
            raise TypeError("record_behavior必须是bool")

    @property
    def controller_id(self):
        return f"{self.program.identifier}-{self.factorial_policy.identifier}"

    def controller_identity(self):
        return {
            "program_id": self.program.identifier,
            "factorial_policy_id": self.factorial_policy.identifier,
            "factorial_policy": self.factorial_policy.to_dict(),
        }


@dataclass(frozen=True)
class BaselineTask:
    """基线参数任务与GP程序任务分离，共享完整面板及FE终止契约。"""

    occurrence_id: str
    policy: BaselinePolicy
    problems: tuple[Instance, ...]
    replicas: tuple[tuple[str, int], ...]
    evaluation_limit_per_colony: int
    preparation_mode: str = "cached"
    experiment_mask: int = 0xFFFFFFFF
    budget_seconds: None = field(default=None, init=False)
    preparation_charges: None = field(default=None, init=False)

    def __post_init__(self):
        if type(self.policy) is not BaselinePolicy:
            raise TypeError("基线任务需要验证后的BaselinePolicy")
        _normalize_task_members(self)
        if (
            type(self.evaluation_limit_per_colony) is not int
            or not 0 <= self.evaluation_limit_per_colony <= 4096 * 0xFFFFFFFF
            or self.preparation_mode not in ("cached", "end_to_end")
        ):
            raise ValueError("基线任务需要整数FE限额及无扣费准备模式")
        self.policy.validate_mask(self.experiment_mask)

    @property
    def dimension(self) -> int:
        return self.problems[0].dimension

    @property
    def controller_id(self) -> str:
        return self.policy.identifier

    def controller_identity(self) -> dict:
        return {
            "baseline_policy_id": self.policy.identifier,
            "baseline_policy": self.policy.to_dict(),
            "controller_kind": self.policy.kind,
        }

    def manifest(self, protocol: WorkerProtocol) -> dict:
        return _task_manifest(self, protocol)

    def task_id(self, protocol: WorkerProtocol) -> str:
        return self.occurrence_id


@dataclass(frozen=True)
class FacoTask(BaselineTask):
    """节点重定位 FACO 固定对照：MNE 8、全路线均匀起点、不重启。"""

    @property
    def controller_id(self):
        return "gpu-faco-mne8-uniform-no-restart"

    def controller_identity(self):
        return {"controller_kind": "gpu_faco_without_gp", "mne": 8}


# 以下状态仅由spawn子进程初始化；协调进程不导入CUDA扩展。
_runtime: dict = {}
_engines: dict = {}
_registered: dict = {}
_registration_fees: dict = {}
_engine_generations: dict = {}
_assigned_charges: dict = {}
_population_engines: dict = {}
_protocol: WorkerProtocol | None = None
_native = None
_lease = None


def _settings_object(dimension):
    settings = _native.FixedFacoSettings()
    for name, value in asdict(_protocol.settings).items():
        setattr(settings, name, value)
    settings.ants = _protocol.settings.ants_for(dimension)
    return settings


def _replace_engine(dimension: int) -> None:
    # 只在任务边界替换通用缓冲；容量策略属于公开protocol，不在批内作隐式释放。
    _engines.pop(dimension, None)
    _engines[dimension] = _native.FacoBatchEngine(
        dimension, _protocol.colonies, _settings_object(dimension), _protocol.constraint_mode
    )
    _registered[dimension] = {}
    _registration_fees[dimension] = {}
    _assigned_charges[dimension] = {}
    _engine_generations[dimension] = _engine_generations.get(dimension, -1) + 1


def _wait_for_idle(protocol):
    """已持UUID锁但尚未创建CUDA上下文；等待上一进程释放后的空闲读数。"""
    observations = []
    while True:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
        )
        if any(
            row and row[0].strip() == protocol.gpu_uuid for row in csv.reader(apps.splitlines())
        ):
            raise BlockingIOError("指定GPU已有计算进程，worker不启动")
        device = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={protocol.gpu_uuid}",
                "--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        row = [value.strip() for value in next(csv.reader(device.splitlines()))]
        if (
            row[0] != protocol.gpu_uuid
            or row[1] != protocol.gpu_model
            or (row[4] != protocol.driver_version)
        ):
            raise RuntimeError("实际GPU/driver与任务硬件协议不一致")
        observations.append({"device": device.strip(), "foreign_processes": 0})
        if int(row[2]) <= 1024 and int(row[3]) <= 5:
            return device, observations
        # 观察等待不扣搜索FE、不产生求解deadline；下一次仍重查外来进程。
        time.sleep(1)


def _initialize(protocol: WorkerProtocol) -> None:
    import fcntl

    global _protocol, _native, _lease, _runtime
    started = time.perf_counter()
    _protocol = protocol
    if protocol.execution_host != platform.node():
        raise RuntimeError("执行host与任务协议不符，不能混用准备费用")
    directory = (PROJECT / protocol.extension_directory).resolve()
    binary = directory / "gp_faco_ext.so"
    for relative in (".tmp", ".cache/cuda"):
        (PROJECT / relative).mkdir(parents=True, exist_ok=True)
    # 各隔离工作树必须使用同一UUID锁，不能各自在自己的.tmp中持有无关锁。
    common_git = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=PROJECT,
            text=True,
        ).strip()
    ).resolve()
    workspace = common_git.parent
    if common_git.name != ".git" or not PROJECT.is_relative_to(workspace):
        raise RuntimeError("无法确定项目内所有工作树共享的GPU锁目录")
    lock_path = workspace / ".tmp" / f"worker-{protocol.gpu_uuid}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lease = lock_path.open("a")
    fcntl.flock(_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device, idle_observations = _wait_for_idle(protocol)
    os.environ["CUDA_VISIBLE_DEVICES"] = protocol.gpu_uuid
    os.environ["CUDA_CACHE_PATH"] = str(PROJECT / ".cache/cuda")
    os.environ["TMPDIR"] = str(PROJECT / ".tmp")
    sys.path.insert(0, str(directory))
    _native = importlib.import_module("gp_faco_ext")
    if Path(_native.__file__).resolve() != binary:
        raise RuntimeError("实际导入了其他路径的CUDA扩展")
    if _native.numeric_backend != protocol.numeric_backend:
        raise RuntimeError("构建数值后端与实验参数不符")
    device_info = _native.cuda_device_info()
    if device_info["name"] != protocol.gpu_model:
        raise RuntimeError("CUDA可见设备未绑定到指定GPU")
    for n in protocol.dimensions:
        _replace_engine(n)
    from gp_faco.remote import process_receipt

    _runtime = {
        **process_receipt(),
        "pid": os.getpid(),
        "host": platform.node(),
        "start_method": multiprocessing.get_start_method(),
        "protocol_id": protocol.identifier,
        "device_before_start": device.strip(),
        "device": device_info,
        "startup_seconds": time.perf_counter() - started,
        "build_id": protocol.build_id,
        "gpu_lock_path": str(lock_path),
        "constraint_mode": protocol.constraint_mode,
        "graph_prior_kind": protocol.graph_prior_kind,
        "idle_observations": idle_observations,
    }


def _initialize_and_ready(protocol) -> dict:
    # 初始化作为首个任务返回原始异常；资源竞争不能被进程池包装成不可区分的崩溃。
    _initialize(protocol)
    return dict(_runtime)


def _register(problems: tuple[Instance, ...]) -> tuple[dict, dict]:
    n = problems[0].dimension
    # 数据集提供稳定的行编号，配对结果不依赖注册顺序或坐标摘要。
    problem_keys = {p.instance_id: p.numeric_id for p in problems}
    if any(key <= 0 for key in problem_keys.values()):
        raise ValueError("GPU 实例必须有稳定的正 numeric_id")
    new_keys = set(problem_keys.values()) - _registered[n].keys()
    if len(_registered[n]) + len(new_keys) > _protocol.maximum_registered_per_dimension:
        _replace_engine(n)
    fees = {}
    for problem in problems:
        key = problem.numeric_id
        if key not in _registered[n]:
            coordinates = np.asarray(problem.coordinates, dtype=np.float64)
            if _protocol.graph_catalog is not None:
                graph = _protocol.graph_catalog.load_graph(problem, _protocol.graph_prior_kind)
                fee = _engines[n].register_graph_problem(key, coordinates, graph)
            else:
                fee = _engines[n].register_problem(key, coordinates)
            _registration_fees[n][key] = fee
            _registered[n][key] = problem.instance_id
        fees[problem.instance_id] = dict(_registration_fees[n][key])
    return problem_keys, fees


def _prepare(problems: tuple[Instance, ...]) -> dict:
    started = time.perf_counter()
    n = problems[0].dimension
    description = {
        "protocol_id": _protocol.identifier,
        "dimension": n,
        "problems": [p.instance_id for p in problems],
        **_protocol.graph_identity(problems),
    }
    output = {
        **description,
        "kind": "preparation",
        "preparation_id": preparation_id(problems),
        "worker_pid": os.getpid(),
    }
    try:
        _, fees = _register(problems)
        output.update(
            {
                "status": "completed",
                "registration_fees": fees,
                "engine_generation": _engine_generations[n],
            }
        )
    except Exception as error:
        output.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    output["worker_seconds"] = time.perf_counter() - started
    return output


def _execute(task: SolveTask | BaselineTask | FactorialTask) -> dict:
    started = time.perf_counter()
    identity = task.task_id(_protocol)
    output = {
        "task_id": identity,
        "protocol_id": _protocol.identifier,
        **task.controller_identity(),
        "occurrence_id": task.occurrence_id,
        "dimension": task.dimension,
        "worker_pid": os.getpid(),
        **_protocol.graph_identity(task.problems),
    }
    try:
        n = task.dimension
        registration_started = time.perf_counter()
        problem_keys, fees = _register(task.problems)
        output["registration_seconds"] = time.perf_counter() - registration_started
        if task.preparation_charges is not None:
            for name, cheap, preparation in task.preparation_charges:
                key = problem_keys[name]
                _engines[n].set_preparation_charges(key, cheap, preparation)
                _assigned_charges[n][key] = (cheap, preparation)
        elif task.preparation_mode == "cached_charged" and any(
            key in _assigned_charges[n] for key in problem_keys.values()
        ):
            raise ValueError("该实例已有冻结费用，任务必须显式携带同一费用")
        keys = np.asarray([problem_keys[name] for name, _ in task.replicas], dtype=np.uint64)
        seeds = np.asarray([seed for _, seed in task.replicas], dtype=np.uint64)
        if type(task) is FacoTask:
            native_result = _engines[n].evaluate_faco_evaluations(
                keys, seeds, task.evaluation_limit_per_colony, 8, task.preparation_mode
            )
        elif type(task) is FactorialTask:
            native_result = _engines[n].evaluate_factorial_evaluations(
                keys,
                seeds,
                task.evaluation_limit_per_colony,
                task.program.to_dict(),
                task.factorial_policy.to_dict(),
                task.preparation_mode,
                task.experiment_mask,
                record_behavior=task.record_behavior,
            )
        elif type(task) is BaselineTask:
            native_result = _engines[n].evaluate_baseline_evaluations(
                keys,
                seeds,
                task.evaluation_limit_per_colony,
                task.policy.to_dict(),
                task.preparation_mode,
                task.experiment_mask,
            )
        elif task.evaluation_limit_per_colony is not None:
            native_result = _engines[n].evaluate_program_evaluations(
                keys,
                seeds,
                task.evaluation_limit_per_colony,
                task.program.to_dict(),
                task.preparation_mode,
                task.experiment_mask,
            )
        else:
            native_result = _engines[n].evaluate_program(
                keys,
                seeds,
                task.budget_seconds,
                task.program.to_dict(),
                task.preparation_mode,
                task.experiment_mask,
            )
        output.update(
            {
                "status": "completed",
                "native_result": native_result,
                "registration_fees": fees,
                "registered_problems": len(_registered[n]),
                "engine_generation": _engine_generations[n],
            }
        )
    except Exception as error:
        # 保留整个预定任务的失败。基础设施进程崩溃仍由Future异常显式传播。
        output.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    output["worker_seconds"] = time.perf_counter() - started
    return output


def _population_members(tasks, protocol):
    if not tasks or len(tasks) > 128 or protocol.constraint_mode != "unrestricted":
        raise ValueError("种群批次需要 1..128 个普通 FACO 个体")
    first = tasks[0]

    def signature(t):
        return (
            type(t),
            t.dimension,
            t.replicas,
            t.evaluation_limit_per_colony,
            t.preparation_mode,
            t.experiment_mask,
            t.preparation_charges,
            getattr(t, "factorial_policy", None),
            tuple((p.instance_id, p.numeric_id) for p in t.problems),
        )

    expected = signature(first)
    for task in tasks:
        if type(task) not in (SolveTask, FactorialTask) or signature(task) != expected:
            raise ValueError("种群批次必须共享实例、seed、预算、mask 和析因设置")
        if task.evaluation_limit_per_colony is None or task.preparation_mode != "cached":
            raise ValueError("种群批次只接受 cached 的评价次数入口")
        if getattr(task, "record_behavior", False):
            raise ValueError("行为诊断使用独立单程序入口")
        task.manifest(protocol)
    return first


def _execute_population(tasks) -> dict:
    started = time.perf_counter()
    first = _population_members(tasks, _protocol)
    n, count = first.dimension, len(tasks)
    if count == 1 and n not in _population_engines:
        return {"members": [_execute(first)]}
    outputs = [
        {
            "task_id": task.task_id(_protocol),
            "protocol_id": _protocol.identifier,
            **task.controller_identity(),
            "occurrence_id": task.occurrence_id,
            "dimension": n,
            "worker_pid": os.getpid(),
            **_protocol.graph_identity(task.problems),
        }
        for task in tasks
    ]
    try:
        registration_started = time.perf_counter()
        cached = _population_engines.get(n)
        incoming = {p.numeric_id for p in first.problems}
        if cached is not None and (
            cached["count"] < count
            or len(cached["registered"] | incoming) > _protocol.maximum_registered_per_dimension
        ):
            del _population_engines[n]
            cached = None
        if cached is None:
            cached = {
                "count": count,
                "engine": _native.FacoBatchEngine(
                    n, _protocol.colonies, _settings_object(n), population_size=count
                ),
                "registered": set(),
                "fees": {},
            }
            _population_engines[n] = cached
        engine = cached["engine"]
        for problem in first.problems:
            key = problem.numeric_id
            if key not in cached["registered"]:
                cached["fees"][problem.instance_id] = engine.register_problem(
                    key, np.asarray(problem.coordinates, dtype=np.float64)
                )
                cached["registered"].add(key)
        keys_by_name = {p.instance_id: p.numeric_id for p in first.problems}
        keys = np.asarray([keys_by_name[name] for name, _ in first.replicas], dtype=np.uint64)
        seeds = np.asarray([seed for _, seed in first.replicas], dtype=np.uint64)
        registration = time.perf_counter() - registration_started
        results = engine.evaluate_population_evaluations(
            keys,
            seeds,
            first.evaluation_limit_per_colony,
            [task.program.to_dict() for task in tasks],
            first.experiment_mask,
            factorial_policy=first.factorial_policy.to_dict()
            if type(first) is FactorialTask
            else None,
        )
        for output, result in zip(outputs, results, strict=True):
            output.update(
                status="completed",
                native_result=result,
                registration_seconds=registration / count,
                registered_problems=len(cached["registered"]),
                registration_fees={
                    p.instance_id: cached["fees"][p.instance_id] for p in first.problems
                },
            )
    except Exception as error:
        for output in outputs:
            output.update(status="failed", error=f"{type(error).__name__}: {error}")
    elapsed = time.perf_counter() - started
    for output in outputs:
        output["worker_seconds"] = elapsed / count
        output["population_worker_seconds"] = elapsed
    return {"members": outputs}


class PersistentGpuWorker:
    """每worker一个活动任务；Future超时不会重启进程或重新提交任务。"""

    def __init__(self, protocol: WorkerProtocol):
        if "gp_faco_ext" in sys.modules:
            raise RuntimeError("协调进程不得导入CUDA扩展；请使用独立的spawn协调入口")
        self.protocol = protocol
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._ready_future = self._executor.submit(_initialize_and_ready, protocol)
        self._active: Future | None = None
        self._closed = False

    def ready(self, timeout: float | None = None) -> dict:
        return self._ready_future.result(timeout=timeout)

    def submit(self, task: SolveTask | BaselineTask | FactorialTask) -> Future:
        if type(task) not in (SolveTask, BaselineTask, FactorialTask, FacoTask):
            raise TypeError("worker仅接受已验证的GP、基线或析因任务")
        task.manifest(self.protocol)
        self._check_idle()
        self._active = self._executor.submit(_execute, task)
        return self._active

    def prepare(self, problems: tuple[Instance, ...]) -> Future:
        frozen = freeze_problems(problems)
        if (
            frozen[0].dimension not in self.protocol.dimensions
            or len(frozen) > self.protocol.colonies
        ):
            raise ValueError("准备问题超出固定worker规模或容量")
        self.protocol.graph_identity(frozen)
        self._check_idle()
        self._active = self._executor.submit(_prepare, frozen)
        return self._active

    def submit_population(self, tasks) -> Future:
        tasks = tuple(tasks)
        _population_members(tasks, self.protocol)
        self._check_idle()
        self._active = self._executor.submit(_execute_population, tasks)
        return self._active

    def _check_idle(self) -> None:
        if self._closed:
            raise RuntimeError("worker已关闭")
        if self._active is not None and not self._active.done():
            raise RuntimeError("同一worker已有活动任务；继续等待该Future，不得重复启动")

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True

    def __enter__(self) -> PersistentGpuWorker:
        return self

    def __exit__(self, *args) -> None:
        self.close()
