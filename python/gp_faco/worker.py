"""显式spawn的常驻单GPU执行器；只接收无标签坐标、IR和固定形状任务。"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
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

from gp_faco.data import Instance
from gp_faco.program_ir import Program

PROJECT = Path(__file__).resolve().parents[2]


def content_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hash() -> str:
    return content_hash(
        {
            name: file_hash(PROJECT / "python/gp_faco" / name)
            for name in ("worker.py", "fitness.py", "data.py", "program_ir.py", "primitives.py")
        }
    )


def coordinate_hash(instance: Instance) -> str:
    xy = np.asarray(instance.coordinates, dtype="<f8")
    return hashlib.sha256(b"ordered-continuous-fp64-v1\0" + xy.tobytes()).hexdigest()


def freeze_problems(values) -> tuple[Instance, ...]:
    problems = []
    for value in values:
        if type(value) is not Instance:
            raise TypeError("worker问题必须是无标签Instance")
        if type(value.instance_id) is not str or not value.instance_id:
            raise ValueError("实例身份必须为非空字符串")
        xy = tuple(tuple(0.0 if x == 0 else float(x) for x in point) for point in value.coordinates)
        problems.append(Instance(value.instance_id, xy, value.distance_spec))
    problems.sort(key=lambda p: p.instance_id)
    ids = [problem.instance_id for problem in problems]
    if not ids or len(set(ids)) != len(ids) or len({p.dimension for p in problems}) != 1:
        raise ValueError("面板必须包含同规模且身份不同的问题")
    if len({coordinate_hash(p) for p in problems}) != len(ids):
        raise ValueError("一个面板不得重复登记同一有序点集")
    return tuple(problems)


@dataclass(frozen=True)
class SolverSettings:
    ants: int = 32
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
        if not 1 <= self.ants <= 128 or self.primary_width < 2 or self.ls_width < 1:
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


@dataclass(frozen=True)
class WorkerProtocol:
    gpu_uuid: str
    gpu_model: str
    driver_version: str
    binary_sha256: str
    dimensions: tuple[int, ...] = (500, 1000)
    colonies: int = 32
    settings: SolverSettings = field(default_factory=SolverSettings)
    extension_directory: str = "build/cuda"
    maximum_registered_per_dimension: int = 1024
    execution_host: str = field(default_factory=platform.node)
    implementation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "implementation_sha256", implementation_hash())
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", self.gpu_uuid):
            raise ValueError("需要完整GPU UUID")
        if not self.gpu_model or not self.driver_version:
            raise ValueError("必须显式指定同型号GPU和driver")
        if not re.fullmatch(r"[0-9a-f]{64}", self.binary_sha256):
            raise ValueError("需要扩展二进制SHA256")
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

    def manifest(self) -> dict:
        return {
            **asdict(self),
            "protocol_version": 2,
            "preparation_fee_policy": "explicit_task_charges_or_measured_registration",
            "instance_key_policy": "uint64_prefix_ordered_continuous_fp64_v1",
            "fitness_spec_id": "reference_gap_instance_then_scale_macro_v1",
            "submission_boundary": "native_whole_panel_completed_and_verified",
        }

    @property
    def sha256(self) -> str:
        return content_hash(self.manifest())


@dataclass(frozen=True)
class SolveTask:
    occurrence_id: str
    program: Program
    problems: tuple[Instance, ...]
    replicas: tuple[tuple[str, int], ...]
    budget_seconds: float
    preparation_mode: str = "cached_charged"
    experiment_mask: int = 0xFFFFFFFF
    preparation_charges: tuple[tuple[str, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if type(self.occurrence_id) is not str or not self.occurrence_id:
            raise ValueError("每个实际评价位置必须有身份，重复树也不能共用fitness缓存")
        if type(self.program) is not Program:
            raise TypeError("任务只接受验证后的Program")
        object.__setattr__(self, "problems", freeze_problems(self.problems))
        object.__setattr__(self, "replicas", tuple(tuple(row) for row in self.replicas))
        ids = [problem.instance_id for problem in self.problems]
        if any(
            len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            or not 0 <= row[1] <= 0xFFFFFFFFFFFFFFFF
            for row in self.replicas
        ):
            raise ValueError("replica需要实例身份和uint64 solve seed")
        if len(set(self.replicas)) != len(self.replicas) or {
            row[0] for row in self.replicas
        } != set(ids):
            raise ValueError("replica有遗漏、未知实例或重复实例/seed")
        if (
            type(self.budget_seconds) not in (int, float)
            or not math.isfinite(self.budget_seconds)
            or self.budget_seconds < 0
        ):
            raise ValueError("预算必须有限且非负")
        if self.preparation_mode not in ("cached_charged", "end_to_end"):
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

    def manifest(self, protocol: WorkerProtocol) -> dict:
        if self.dimension not in protocol.dimensions or len(self.replicas) != protocol.colonies:
            raise ValueError("任务与worker固定规模/形状不符")
        return {
            "occurrence_id": self.occurrence_id,
            "program_sha256": self.program.sha256,
            "protocol_sha256": protocol.sha256,
            "dimension": self.dimension,
            "problems": [(p.instance_id, coordinate_hash(p)) for p in self.problems],
            "replicas": self.replicas,
            "budget_seconds": self.budget_seconds,
            "preparation_mode": self.preparation_mode,
            "experiment_mask": self.experiment_mask,
            "preparation_charges": self.preparation_charges,
        }

    def task_id(self, protocol: WorkerProtocol) -> str:
        return content_hash(self.manifest(protocol))


# 以下状态仅由spawn子进程初始化；协调进程不导入CUDA扩展。
_runtime: dict = {}
_engines: dict = {}
_registered: dict = {}
_engine_generations: dict = {}
_assigned_charges: dict = {}
_protocol: WorkerProtocol | None = None
_native = None
_lease = None


def _settings_object():
    settings = _native.FixedFacoSettings()
    for name, value in asdict(_protocol.settings).items():
        setattr(settings, name, value)
    return settings


def _replace_engine(dimension: int) -> None:
    # 只在任务边界替换通用缓冲；容量策略属于公开protocol，不在批内作隐式释放。
    _engines.pop(dimension, None)
    _engines[dimension] = _native.FacoBatchEngine(dimension, _protocol.colonies, _settings_object())
    _registered[dimension] = {}
    _assigned_charges[dimension] = {}
    _engine_generations[dimension] = _engine_generations.get(dimension, -1) + 1


def _initialize(protocol: WorkerProtocol) -> None:
    import fcntl

    global _protocol, _native, _lease, _runtime
    started = time.perf_counter()
    _protocol = protocol
    if implementation_hash() != protocol.implementation_sha256:
        raise RuntimeError("启动期间worker源码身份改变")
    if protocol.execution_host != platform.node():
        raise RuntimeError("执行host与任务协议不符，不能混用准备费用")
    directory = (PROJECT / protocol.extension_directory).resolve()
    binary = directory / "gp_faco_ext.so"
    if file_hash(binary) != protocol.binary_sha256:
        raise RuntimeError("CUDA扩展与冻结的任务协议指纹不同")
    for relative in (".tmp", ".cache/cuda"):
        (PROJECT / relative).mkdir(parents=True, exist_ok=True)
    _lease = (PROJECT / ".tmp" / f"worker-{protocol.gpu_uuid}.lock").open("a")
    fcntl.flock(_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if any(row and row[0].strip() == protocol.gpu_uuid for row in csv.reader(apps.splitlines())):
        raise RuntimeError("指定GPU已有计算进程，worker不启动")
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
        or row[4] != protocol.driver_version
    ):
        raise RuntimeError("实际GPU/driver与任务硬件协议不一致")
    if int(row[2]) > 1024 or int(row[3]) > 5:
        raise RuntimeError("指定GPU未满足空闲阈值")
    os.environ["CUDA_VISIBLE_DEVICES"] = protocol.gpu_uuid
    os.environ["CUDA_CACHE_PATH"] = str(PROJECT / ".cache/cuda")
    os.environ["TMPDIR"] = str(PROJECT / ".tmp")
    sys.path.insert(0, str(directory))
    _native = importlib.import_module("gp_faco_ext")
    if Path(_native.__file__).resolve() != binary:
        raise RuntimeError("实际导入了其他路径的CUDA扩展")
    device_info = _native.cuda_device_info()
    if device_info["name"] != protocol.gpu_model:
        raise RuntimeError("CUDA可见设备未绑定到指定GPU")
    for n in protocol.dimensions:
        _replace_engine(n)
    _runtime = {
        "pid": os.getpid(),
        "host": platform.node(),
        "start_method": multiprocessing.get_start_method(),
        "protocol_sha256": protocol.sha256,
        "device_before_start": device.strip(),
        "device": device_info,
        "startup_seconds": time.perf_counter() - started,
        "binary_sha256": protocol.binary_sha256,
    }


def _ready() -> dict:
    return dict(_runtime)


def _register(problems: tuple[Instance, ...]) -> tuple[dict, dict]:
    n = problems[0].dimension
    problem_keys = {p.instance_id: int(coordinate_hash(p)[:16], 16) for p in problems}
    new_keys = set(problem_keys.values()) - _registered[n].keys()
    if len(_registered[n]) + len(new_keys) > _protocol.maximum_registered_per_dimension:
        _replace_engine(n)
    fees = {}
    for problem in problems:
        key, fingerprint = problem_keys[problem.instance_id], coordinate_hash(problem)
        old = _registered[n].get(key)
        if old is not None and old != fingerprint:
            raise ValueError("不同坐标的实例key发生64位碰撞")
        fees[problem.instance_id] = _engines[n].register_problem(
            key, np.asarray(problem.coordinates, dtype=np.float64)
        )
        _registered[n][key] = fingerprint
    return problem_keys, fees


def _prepare(problems: tuple[Instance, ...]) -> dict:
    started = time.perf_counter()
    n = problems[0].dimension
    description = {
        "protocol_sha256": _protocol.sha256,
        "dimension": n,
        "problems": [(p.instance_id, coordinate_hash(p)) for p in problems],
    }
    output = {
        **description,
        "kind": "preparation",
        "preparation_id": content_hash(description),
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


def _execute(task: SolveTask) -> dict:
    started = time.perf_counter()
    identity = task.task_id(_protocol)
    output = {
        "task_id": identity,
        "protocol_sha256": _protocol.sha256,
        "program_sha256": task.program.sha256,
        "occurrence_id": task.occurrence_id,
        "dimension": task.dimension,
        "worker_pid": os.getpid(),
    }
    try:
        n = task.dimension
        problem_keys, fees = _register(task.problems)
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


class PersistentGpuWorker:
    """每worker一个活动任务；Future超时不会重启进程或重新提交任务。"""

    def __init__(self, protocol: WorkerProtocol):
        if "gp_faco_ext" in sys.modules:
            raise RuntimeError("协调进程不得导入CUDA扩展；请使用独立的spawn协调入口")
        self.protocol = protocol
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(protocol,),
        )
        self._ready_future = self._executor.submit(_ready)
        self._active: Future | None = None
        self._closed = False

    def ready(self, timeout: float | None = None) -> dict:
        return self._ready_future.result(timeout=timeout)

    def submit(self, task: SolveTask) -> Future:
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
        self._check_idle()
        self._active = self._executor.submit(_prepare, frozen)
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
