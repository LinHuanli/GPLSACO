"""开发曲线和 baseline 的常驻 A5000 求解会话；标签仅由调用方评分。"""

from __future__ import annotations

import fcntl
import importlib
import os
import subprocess
import sys
import time

import numpy as np

from gp_faco.worker import PROJECT, faco_ants


class GpuSession:
    def __init__(self, gpu_uuid, backend, *, colonies=32, extension_directory=None):
        self.gpu_uuid, self.backend, self.colonies = gpu_uuid, backend, colonies
        self.lease = (PROJECT / ".tmp" / f"worker-{gpu_uuid}.lock").open("a")
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_uuid}",
                    "--query-gpu=name,driver_version,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(", ")
        )
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        )
        if (
            info[0] != "NVIDIA RTX A5000"
            or gpu_uuid in apps
            or int(info[2]) > 1024
            or int(info[3]) > 5
        ):
            self.lease.close()
            raise RuntimeError("新会话需要实时空闲 RTX A5000；请先用 gpu-free 选择")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        os.environ["CUDA_CACHE_PATH"] = str(PROJECT / ".cache/cuda")
        build = {"exact": "v2-exact", "fp32": "v2-fp32", "fp32_fast": "v2-fp32-fast"}[backend]
        directory = (PROJECT / (extension_directory or f"build/{build}")).resolve()
        if not directory.is_relative_to(PROJECT):
            raise ValueError("扩展必须位于项目目录内")
        build = directory.name
        sys.path.insert(0, str(directory))
        self.native = importlib.import_module("gp_faco_ext")
        if self.native.numeric_backend != backend:
            raise ValueError("一个 Python 进程只加载一个数值后端")
        self.device = {"gpu_uuid": gpu_uuid, "model": info[0], "driver": info[1], "build": build}
        self.engines, self.registered = {}, {}
        self.population_engines = {}
        self.population_registered = {}

    def solve(
        self, problems, pairs, iterations, controller, *, checkpoints=(), decision_iterations=()
    ):
        if len(pairs) != self.colonies:
            raise ValueError("FACO 必须收到一个完整固定面板")
        n = problems[pairs[0][0]].dimension
        if n not in self.engines:
            settings = self.native.FixedFacoSettings()
            settings.ants = faco_ants(n)
            self.engines[n] = self.native.FacoBatchEngine(n, self.colonies, settings)
            self.registered[n] = set()
        engine = self.engines[n]
        before = time.perf_counter()
        for name, _ in pairs:
            problem = problems[name]
            if problem.dimension != n:
                raise ValueError("面板混合了不同规模")
            if name not in self.registered[n]:
                engine.register_problem(
                    problem.numeric_id, np.asarray(problem.coordinates, dtype=np.float64)
                )
                self.registered[n].add(name)
        registration_seconds = time.perf_counter() - before
        keys = np.asarray([problems[name].numeric_id for name, _ in pairs], dtype=np.uint64)
        seeds = np.asarray([seed for _, seed in pairs], dtype=np.uint64)
        before = time.perf_counter()
        if controller is None or "uniform_mne" in controller:
            result = engine.evaluate_faco_evaluations(
                keys,
                seeds,
                faco_ants(n) * iterations,
                8 if controller is None else controller["uniform_mne"],
                "cached",
                checkpoint_iterations=list(checkpoints),
            )
        elif "controller_version" in controller:
            result = engine.evaluate_controller_evaluations(
                keys, seeds, faco_ants(n) * iterations, controller,
                decision_iterations=list(decision_iterations))
        elif "opcode" in controller:
            observation = (
                {"decision_iterations": list(decision_iterations)} if decision_iterations else {}
            )
            result = engine.evaluate_program_evaluations(
                keys,
                seeds,
                faco_ants(n) * iterations,
                controller,
                "cached",
                checkpoint_iterations=list(checkpoints),
                **observation,
            )
        else:
            observation = {"decision_iterations": list(decision_iterations)} if decision_iterations else {}
            result = engine.evaluate_baseline_evaluations(
                keys,
                seeds,
                faco_ants(n) * iterations,
                controller,
                "cached",
                checkpoint_iterations=list(checkpoints),
                **observation,
            )
        return result, {
            "registration_seconds": registration_seconds,
            "solve_seconds": time.perf_counter() - before,
            "device_bytes": result["allocated_device_bytes"],
        }

    def set_colonies(self, colonies):
        if not 1 <= colonies <= 128:
            raise ValueError("GPU面板形状必须在1..128")
        if colonies != self.colonies:
            self.engines.clear(); self.registered.clear()
            self.population_engines.clear(); self.population_registered.clear()
            self.colonies = colonies

    def solve_population(self, problems, pairs, iterations, programs, *, experiment_mask=0xFFFFFFFF):
        if len(pairs) != self.colonies or not 1 <= len(programs) <= 128:
            raise ValueError("种群分片需要完整面板和 1..128 个真实程序")
        n = problems[pairs[0][0]].dimension
        key = n, len(programs)
        before = time.perf_counter()
        if key not in self.population_engines:
            settings = self.native.FixedFacoSettings()
            settings.ants = faco_ants(n)
            engine = self.native.FacoBatchEngine(
                n, self.colonies, settings, population_size=len(programs)
            )
            self.population_engines[key] = engine
            self.population_registered[key] = set()
        engine = self.population_engines[key]
        for name in dict.fromkeys(name for name, _ in pairs):
            if name not in self.population_registered[key]:
                problem = problems[name]
                engine.register_problem(
                    problem.numeric_id, np.asarray(problem.coordinates, dtype=np.float64)
                )
                self.population_registered[key].add(name)
        registration = time.perf_counter() - before
        keys = np.asarray([problems[name].numeric_id for name, _ in pairs], dtype=np.uint64)
        seeds = np.asarray([seed for _, seed in pairs], dtype=np.uint64)
        before = time.perf_counter()
        result = engine.evaluate_population_evaluations(
            keys, seeds, faco_ants(n) * iterations, programs, experiment_mask
        )
        return result, {
            "registration_seconds": registration,
            "solve_seconds": time.perf_counter() - before,
        }

    def close(self):
        self.engines.clear()
        self.population_engines.clear()
        self.lease.close()
