#!/usr/bin/env python3
"""完整控制动作的开发成本矩阵；固定批次计时、未插桩对照与逐实例/批量轨迹核验。"""

import argparse
import csv
import fcntl
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# 标准库导入后启动内部时钟；外层GNU time另记完整解释器墙钟。
PROCESS_STARTED = time.perf_counter()

import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402


def require(value, message):
    if not value:
        raise RuntimeError(message)


def conditions():
    result = []
    for restart in (False, True):
        for region in range(4):
            for level in range(4):
                keep = 4 * region + level
                result.append(
                    {
                        "name": f"{'restart' if restart else 'keep'}-r{region}-m{2 << level}",
                        "mask": (1 << keep) | ((1 << (keep + 16)) if restart else 0),
                        "program": Program((0,), (4,)).to_dict(),
                    }
                )
    result.append(
        {
            "name": "feedback-program",
            "mask": 0xFFFFFFFF,
            "program": Program((0, 0, 2, 0, 0, 3, 0, 4, 2), (4, 8, 0, 0, 2, 0, 5, 0, 0)).to_dict(),
        }
    )

    def subtree(start, count):
        if count == 1:
            return [(0, start % 12)]
        return subtree(start, count // 2) + subtree(start + count // 2, count // 2) + [(2, 0)]

    nodes = subtree(0, 32)
    result.append(
        {
            "name": "maximum-63-node-program",
            "mask": 0xFFFFFFFF,
            "program": Program(tuple(v[0] for v in nodes), tuple(v[1] for v in nodes)).to_dict(),
        }
    )
    return result


class MeasurementRun:
    def __init__(self, config, output, gpu_uuid, resume):
        self.config, self.output, self.gpu_uuid = config, output, gpu_uuid
        self.costs = {
            name: 0.0
            for name in (
                "admission",
                "metadata",
                "data_read",
                "engine_creation",
                "registration",
                "native_calls",
                "verification",
                "journal",
            )
        }
        self.single_tours = {}
        self.current = None
        self.partial_samples = []
        self._timers = []
        self.peak_gpu_mib = 0
        self.output.mkdir(parents=True, exist_ok=True)
        self._lease = (PROJECT / ".tmp" / f"worker-{gpu_uuid}.lock").open("a")
        fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.admit()
        row = [
            v.strip()
            for v in next(
                csv.reader(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--id={gpu_uuid}",
                            "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).splitlines()
                )
            )
        ]
        require(int(row[3]) <= 1024 and int(row[4]) <= 5, "目标GPU尚未满足空闲阈值")
        self.device = {"uuid": row[0], "model": row[1], "driver": row[2], "host": platform.node()}
        self.database = PROJECT / "artifacts/data/main-index-v1/instances.sqlite"
        source_files = [Path(__file__), PROJECT / "CMakeLists.txt"]
        for part in ("cpp", "cuda"):
            source_files.extend(p for p in (PROJECT / part).rglob("*") if p.is_file())
        source_files.extend(
            PROJECT / "python/gp_faco" / name
            for name in (
                "checkpoint.py",
                "worker.py",
                "data.py",
                "dataset_index.py",
                "program_ir.py",
                "primitives.py",
            )
        )
        with self.timer("metadata"):
            self.manifest = {
                "version": 1,
                "config": config,
                "device": self.device,
                "conditions": conditions(),
                "database_sha256": file_hash(self.database),
                "split_sha256": file_hash(PROJECT / "provenance/splits.v1.json"),
                "binary_sha256": file_hash(PROJECT / "build/cuda/gp_faco_ext.so"),
                "sources": {str(p.relative_to(PROJECT)): file_hash(p) for p in source_files},
            }
            self.identity = content_hash(self.manifest)
            if resume:
                self.state = load_checkpoint(output / "checkpoint.json")
                require(
                    self.state["manifest"] == self.manifest, "恢复时配置、GPU、二进制或源文件改变"
                )
            else:
                require(not any(output.iterdir()), "新剖析要求空目录；已有运行须resume")
                self.state = {
                    "manifest": self.manifest,
                    "identity": self.identity,
                    "completed": {},
                    "fees": {},
                    "status": "running",
                    "invocations": [],
                }
                (output / "runtime").mkdir()
                shutil.copy2(
                    PROJECT / "build/cuda/gp_faco_ext.so", output / "runtime/gp_faco_ext.so"
                )
                save_checkpoint(output / "manifest.json", self.manifest)
                self.save()
        require(
            file_hash(output / "runtime/gp_faco_ext.so") == self.manifest["binary_sha256"],
            "实际将加载的运行副本摘要不符",
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        os.environ["CUDA_CACHE_PATH"] = str(PROJECT / ".cache/cuda")
        os.environ["TMPDIR"] = str(PROJECT / ".tmp")
        sys.path.insert(0, str(output / "runtime"))
        import gp_faco_ext as native

        require(
            Path(native.__file__).resolve() == output / "runtime/gp_faco_ext.so",
            "CUDA扩展加载路径错误",
        )
        self.native = native

    @contextmanager
    def timer(self, name):
        started = time.perf_counter()
        frame = {"children": 0.0}
        self._timers.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._timers.pop()
            self.costs[name] += elapsed - frame["children"]
            if self._timers:
                self._timers[-1]["children"] += elapsed

    def save(self):
        with self.timer("journal"):
            save_checkpoint(self.output / "checkpoint.json", self.state)

    def admit(self):
        with self.timer("admission"):
            rows = csv.reader(
                subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                    text=True,
                ).splitlines()
            )
            foreign = [
                row[1].strip()
                for row in rows
                if row and row[0].strip() == self.gpu_uuid and int(row[1].strip()) != os.getpid()
            ]
            require(not foreign, f"目标GPU检测到其他compute PID，保留产物: {foreign}")
            used = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_uuid}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            self.peak_gpu_mib = max(self.peak_gpu_mib, float(used))

    def execute(self):
        try:
            with IndexedDataset(self.database, PROJECT.parent / "Datasets/TSP") as data:
                for n in self.config["dimensions"]:
                    with self.timer("data_read"):
                        names = data.record_ids("development", n)[
                            : self.config["instances_per_scale"]
                        ]
                        require(
                            len(names) == self.config["instances_per_scale"], "开发实例不足完整面板"
                        )
                        problems = {name: data.load_instance(name) for name in names}
                    for colonies in self.config["colonies"]:
                        self.measure_shape(n, colonies, names, problems)
            self.state["status"] = "complete"
            self.state.pop("error", None)
        except BaseException as error:
            self.state["status"] = "interrupted"
            self.state["error"] = f"{type(error).__name__}: {error}"
            if self.current is not None:
                save_checkpoint(
                    self.output / "failures" / f"{self.current}-{os.getpid()}.json",
                    {
                        "key": self.current,
                        "error": self.state["error"],
                        "partial_samples": self.partial_samples,
                    },
                )
            raise
        finally:
            # 各分项不重叠；全部计入过程时钟，解释器启动前开销由外层GNU time补充。
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.state["invocations"].append(
                {
                    "pid": os.getpid(),
                    "costs_seconds": dict(self.costs),
                    "process_wall_seconds": time.perf_counter() - PROCESS_STARTED,
                    "cpu_user_seconds": usage.ru_utime,
                    "cpu_system_seconds": usage.ru_stime,
                    "max_rss_kib": usage.ru_maxrss,
                    "boundary_sampled_peak_gpu_mib": self.peak_gpu_mib,
                    "last_condition": self.current,
                }
            )
            self.save()
            self._lease.close()

    def measure_shape(self, n, colonies, names, problems):
        self.admit()
        with self.timer("engine_creation"):
            settings = self.native.FixedFacoSettings()
            settings.ants = self.config["ants"]
            engine = self.native.FacoBatchEngine(n, colonies, settings)
        keys = {name: int(coordinate_hash(problem)[:16], 16) for name, problem in problems.items()}
        require(len(set(keys.values())) == len(keys), "64位实例键发生碰撞")
        measurements = {}
        with self.timer("registration"):
            for name in names:
                measured = engine.register_problem(
                    keys[name], np.asarray(problems[name].coordinates, dtype=np.float64)
                )
                measurements[name] = {
                    "measured": measured,
                    "phases": engine.preparation_profile(keys[name]),
                }
                if name not in self.state["fees"]:
                    self.state["fees"][name] = {
                        **measured,
                        "coordinate_sha256": coordinate_hash(problems[name]),
                    }
                frozen = self.state["fees"][name]
                require(
                    frozen["coordinate_sha256"] == coordinate_hash(problems[name]),
                    "准备费用坐标改变",
                )
                engine.set_preparation_charges(
                    keys[name], frozen["cheap_seconds"], frozen["preparation_seconds"]
                )
        self.save()
        groups = [[name] for name in names] if colonies == 1 else [names]
        for group_index, group in enumerate(groups):
            replicas = [
                (name, seed)
                for name in group
                for seed in (
                    self.config["solver_seeds"][:1]
                    if colonies == 1
                    else self.config["solver_seeds"]
                )
            ]
            require(len(replicas) == colonies, "数据面板与固定形状不符")
            native_keys = np.asarray([keys[name] for name, _ in replicas], dtype=np.uint64)
            native_seeds = np.asarray([seed for _, seed in replicas], dtype=np.uint64)
            for condition in self.manifest["conditions"]:
                description = {
                    "run_id": self.identity,
                    "dimension": n,
                    "colonies": colonies,
                    "replicas": replicas,
                    "condition": condition,
                    "problems": [(name, coordinate_hash(problems[name])) for name in group],
                }
                key = content_hash(description)
                self.current = key
                self.partial_samples = []
                path = self.output / "conditions" / f"{key}.json"
                if path.exists():
                    record = load_checkpoint(path)
                    require(
                        record["key"] == key
                        and content_hash(record["description"]) == content_hash(description),
                        "已有条件产物身份不符",
                    )
                    if key in self.state["completed"]:
                        require(
                            self.state["completed"][key] == content_hash(record),
                            "已完成条件摘要改变",
                        )
                else:
                    require(key not in self.state["completed"], "已完成条件产物丢失")
                    self.admit()
                    with self.timer("journal"):
                        save_checkpoint(
                            self.output / "pending.json", {"key": key, "description": description}
                        )
                    samples = []
                    self.partial_samples = samples
                    # 预热独立于报告样本；所有调用仍重置到同一实例/seed，固定elapsed不受计时影响。
                    with self.timer("native_calls"):
                        engine.run_program_diagnostic(
                            native_keys,
                            native_seeds,
                            condition["program"],
                            self.config["warmup_batches"],
                            self.config["elapsed_ratio"],
                            False,
                            condition["mask"],
                            budget_seconds=self.config["guard_budget_seconds"],
                        )
                    for enabled in self.config["profile_order"]:
                        with self.timer("native_calls"):
                            result = engine.run_program_diagnostic(
                                native_keys,
                                native_seeds,
                                condition["program"],
                                self.config["fixed_batches"],
                                self.config["elapsed_ratio"],
                                enabled,
                                condition["mask"],
                                budget_seconds=self.config["guard_budget_seconds"],
                            )
                        with self.timer("verification"):
                            require(
                                result["completed_batches"] == self.config["fixed_batches"]
                                and result["discarded_batches"] == 0,
                                "固定批次未全部完成",
                            )
                            for (name, _), item in zip(replicas, result["items"], strict=True):
                                require(
                                    item["has_incumbent"]
                                    and abs(tour_cost(problems[name], item["tour"]) - item["cost"])
                                    < 1e-8,
                                    "原目标路线核验失败",
                                )
                            if samples:
                                reference = samples[0]["result"]
                                require(
                                    self.signature(result) == self.signature(reference),
                                    "插桩或重复执行改变轨迹",
                                )
                        samples.append({"profile_enabled": enabled, "result": result})
                    self.admit()
                    record = {
                        "key": key,
                        "description": description,
                        "status": "completed",
                        "samples": samples,
                        "registration": {name: measurements[name] for name in group},
                        "pid": os.getpid(),
                    }
                    with self.timer("journal"):
                        save_checkpoint(path, record)
                self.state["completed"][key] = content_hash(record)
                self.check_shape(
                    n, colonies, condition["name"], replicas, record["samples"][0]["result"]
                )
                self.save()
                if len(self.state["completed"]) % 34 == 0:
                    print(
                        json.dumps(
                            {
                                "completed_conditions": len(self.state["completed"]),
                                "dimension": n,
                                "colonies": colonies,
                                "group": group_index,
                            }
                        ),
                        flush=True,
                    )
        del engine

    @staticmethod
    def signature(result):
        return (
            [(v["tour"], v["cost"]) for v in result["items"]],
            result["control_states"],
            result["completed_construction_steps"],
            result["completed_ls_evaluations"],
        )

    def check_shape(self, n, colonies, condition, replicas, result):
        for (name, seed), item in zip(replicas, result["items"], strict=True):
            if seed != self.config["solver_seeds"][0]:
                continue
            key = n, condition, name
            if colonies == 1:
                self.single_tours[key] = item["tour"], item["cost"]
            else:
                require(
                    self.single_tours[key] == (item["tour"], item["cost"]),
                    "固定elapsed下批量改变单实例轨迹",
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/profiling_v1.json")
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/gpu/profiling/cost-matrix-v1"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(
        args.output.resolve().is_relative_to(PROJECT)
        and args.config.resolve().is_relative_to(PROJECT),
        "配置与产物必须位于GPLSACO内",
    )
    run = MeasurementRun(
        json.loads(args.config.read_text()), args.output.resolve(), args.gpu_uuid, args.resume
    )
    run.execute()
    print(
        json.dumps(
            {
                "status": run.state["status"],
                "conditions": len(run.state["completed"]),
                "identity": run.identity,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
