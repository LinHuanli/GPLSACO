#!/usr/bin/env python3
"""在同GPU比较冻结主二进制与析因端点，核验真实32×32开发面板，不读取标签。"""

import argparse
import csv
import fcntl
import importlib
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

import numpy as np  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.factorial_policy import FactorialPolicy  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import SolverSettings, coordinate_hash, file_hash  # noqa: E402

ORIGINAL_NATIVE = "b4138774e24bc5d01914fc99e503311ede2859fc8aa6096273b58efae2f5836f"
PROGRAM = Program((0, 0, 2, 0, 0, 3, 0, 4, 2), (4, 8, 0, 0, 2, 0, 5, 0, 0), feature_spec_id=2)
POLICIES = (
    BaselinePolicy(region=2, mne_level=1, max_mne_level=1),
    BaselinePolicy(region=2, restart_mode="periodic", restart_period=3),
    BaselinePolicy(region=3, restart_mode="bernoulli", restart_probability=0.5),
    BaselinePolicy(
        kind="rule",
        max_mne_level=3,
        region=1,
        stagnation_step=2,
        restart_stagnation=3,
        restart_cooldown=3,
    ),
)
LIMIT = 256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sources():
    paths = [Path(__file__), PROJECT / "CMakeLists.txt"]
    for directory in ("cpp", "cuda", "python/gp_faco"):
        paths += [
            p
            for p in (PROJECT / directory).rglob("*")
            if p.suffix in (".cpp", ".cu", ".hpp", ".cuh", ".py")
        ]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(paths)}


def boundary(uuid, pid):
    rows = csv.reader(
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
    )
    return {"foreign_processes": len({int(r[1]) for r in rows if r} - {pid})}


def guard_labels(source):
    def authorize(action, first, second, database, origin):
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and first == "labels"
            else sqlite3.SQLITE_OK
        )

    source.connection.set_authorizer(authorize)


def selected(source, n):
    ids = sorted(source.record_ids("development", n))[:16]
    require(len(ids) == 16, "完整开发面板不足")
    return tuple(source.load_instance(name) for name in ids)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    native_path = args.native_dir.resolve() / "gp_faco_ext.so"
    binary = file_hash(native_path)
    if args.stage == "original":
        require(binary == ORIGINAL_NATIVE, "原端点必须使用冻结b413主二进制")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    cache = PROJECT / ".cache/cuda"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(cache)
    # 本进程只导入一个扩展，避免同名动态库缓存让跨二进制对照失效。
    sys.path.insert(0, str(native_path.parent))
    native = importlib.import_module("gp_faco_ext")
    require(Path(native.__file__).resolve() == native_path, "实际扩展路径不同")
    row = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={args.gpu_uuid}",
                    "--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        )
    )
    require(
        boundary(args.gpu_uuid, -1)["foreign_processes"] == 0
        and int(row[3]) <= 1024
        and int(row[4]) <= 5,
        "GPU不再空闲，未创建Engine",
    )
    manifest = {
        "stage": args.stage,
        "native_path": str(native_path),
        "native_sha256": binary,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "sources": sources(),
        "database_sha256": file_hash(args.database),
        "gpu_uuid": args.gpu_uuid,
        "gpu_model": row[1].strip(),
        "driver": row[2].strip(),
        "host": platform.node(),
        "colonies": 32,
        "ants": 32,
        "FE_per_colony": LIMIT,
        "policies": [p.to_dict() for p in POLICIES],
        "program": PROGRAM.to_dict(),
        "settings": asdict(SolverSettings()),
        "wall_clock_limit": None,
        "formal_test_released": False,
    }
    atomic_json(args.output / "manifest.json", manifest)
    count = 0
    with IndexedDataset(args.database, args.dataset_root) as source:
        guard_labels(source)
        for n in (500, 1000):
            problems = selected(source, n)
            settings = native.FixedFacoSettings()
            for key, value in manifest["settings"].items():
                setattr(settings, key, value)
            engine = native.FacoBatchEngine(n, 32, settings)
            keys_by_id = {p.instance_id: int(coordinate_hash(p)[:16], 16) for p in problems}
            for p in problems:
                engine.register_problem(keys_by_id[p.instance_id], np.asarray(p.coordinates))
            replicas = [(p.instance_id, s) for p in problems for s in (17, 29)]
            keys = np.asarray([keys_by_id[name] for name, _ in replicas], dtype=np.uint64)
            seeds = np.asarray([s for _, s in replicas], dtype=np.uint64)
            sequence = (
                ("M00", "M11")
                if args.stage == "original"
                else ("M00", "M11", "M10", "M01", "M10-repeat", "M01-repeat")
            )
            for j, base in enumerate(POLICIES):
                for name in sequence:
                    variant = name.split("-")[0]
                    case = f"{n}-{j}-{name}"
                    before = boundary(args.gpu_uuid, os.getpid())
                    require(before["foreign_processes"] == 0, "GPU有外来进程，当前调用尚未提交")
                    task = {
                        "case": case,
                        "dimension": n,
                        "replicas": replicas,
                        "variant": variant,
                        "policy": base.to_dict(),
                        "coordinates": {p.instance_id: coordinate_hash(p) for p in problems},
                    }
                    atomic_json(
                        args.output / f"{case}-submitted.json", {"task": task, "before": before}
                    )
                    if args.stage == "original":
                        result = (
                            engine.evaluate_baseline_evaluations(keys, seeds, LIMIT, base.to_dict())
                            if variant == "M00"
                            else engine.evaluate_program_evaluations(
                                keys, seeds, LIMIT, PROGRAM.to_dict()
                            )
                        )
                    else:
                        result = engine.evaluate_factorial_evaluations(
                            keys,
                            seeds,
                            LIMIT,
                            PROGRAM.to_dict(),
                            FactorialPolicy(variant, base).to_dict(),
                        )
                    record = {"task": task, "before": before, "result": result}
                    atomic_json(args.output / f"{case}-return.json", record)
                    try:
                        after = boundary(args.gpu_uuid, os.getpid())
                    except Exception as error:
                        after = {"foreign_processes": None, "error": str(error)}
                    atomic_json(args.output / f"{case}-observed.json", {"after": after})
                    count += 1
    require(
        sources() == manifest["sources"] and file_hash(native_path) == binary,
        "运行期间源码/二进制改变",
    )
    atomic_json(args.output / "completion.json", {"calls": count, "FE": count * 32 * LIMIT})
    print(json.dumps({"status": "complete", "calls": count, "FE": count * 32 * LIMIT}), flush=True)


def causal(result):
    return {
        "items": [(r["tour"], r["cost"]) for r in result["items"]],
        **{
            k: result[k]
            for k in (
                "control_states",
                "total_tour_evaluations",
                "completed_batches",
                "completed_construction_steps",
                "completed_ls_evaluations",
            )
        },
    }


def audit(args):
    def read(p):
        return json.loads(p.read_text())

    directories = (args.reference_output, args.output)
    manifests = [read(d / "manifest.json") for d in directories]
    require(manifests[0]["native_sha256"] == ORIGINAL_NATIVE, "原生主参照不符")
    for field in (
        "gpu_uuid",
        "gpu_model",
        "driver",
        "host",
        "sources",
        "policies",
        "program",
        "settings",
    ):
        require(manifests[0][field] == manifests[1][field], f"端点对照字段不符:{field}")
    records, members, deviations, max_error = [], 0, [], 0.0
    with IndexedDataset(args.database, args.dataset_root) as source:
        guard_labels(source)
        problems = {p.instance_id: p for n in (500, 1000) for p in selected(source, n)}
        for directory, manifest, count in zip(directories, manifests, (16, 48), strict=True):
            require(
                manifest["sources"] == sources()
                and manifest["native_sha256"] == file_hash(Path(manifest["native_path"]))
                and manifest["database_sha256"] == file_hash(args.database),
                "源码/二进制/数据身份改变",
            )
            data = {
                p.name.removesuffix("-return.json"): read(p)
                for p in directory.glob("*-return.json")
            }
            sequence = (
                ("M00", "M11")
                if count == 16
                else ("M00", "M11", "M10", "M01", "M10-repeat", "M01-repeat")
            )
            expected = {
                f"{n}-{j}-{name}" for n in (500, 1000) for j in range(4) for name in sequence
            }
            require(
                set(data) == expected
                and read(directory / "completion.json")
                == {"calls": count, "FE": count * 32 * LIMIT},
                "实际调用覆盖不符",
            )
            for case, row in data.items():
                task, result = row["task"], row["result"]
                require(
                    read(directory / f"{case}-submitted.json")
                    == {"task": task, "before": row["before"]}
                    and row["before"] == {"foreign_processes": 0},
                    "提交前资源或任务身份不符",
                )
                n, j, name = case.split("-", 2)
                base = POLICIES[int(j)]
                expected_replicas = [
                    [p.instance_id, s] for p in selected(source, int(n)) for s in (17, 29)
                ]
                require(
                    task["replicas"] == expected_replicas
                    and task["policy"] == base.to_dict()
                    and task["variant"] == name.split("-")[0],
                    "实际面板或规则不符",
                )
                if count == 48:
                    require(
                        result["factorial_policy"]
                        == FactorialPolicy(task["variant"], base).to_dict(),
                        "原生因素身份不符",
                    )
                require(
                    result["budget_seconds"] is None
                    and result["total_tour_evaluations"] == LIMIT * 32
                    and result["completed_tour_evaluations_per_colony"] == LIMIT
                    and result["launched_batches"] == result["completed_batches"] == LIMIT // 32
                    and result["discarded_batches"]
                    == result["charged_seconds"]
                    == result["overrun_seconds"]
                    == 0
                    and len(result["items"]) == 32,
                    "次数/预算/形状不符",
                )
                for (instance_id, _), item in zip(task["replicas"], result["items"], strict=True):
                    p = problems[instance_id]
                    tour = item["tour"]
                    require(
                        task["coordinates"][instance_id] == coordinate_hash(p)
                        and sorted(tour) == list(range(p.dimension)),
                        "坐标/路线排列不符",
                    )
                    xy = p.coordinates
                    cost = math.fsum(
                        math.hypot(
                            xy[tour[i]][0] - xy[tour[(i + 1) % p.dimension]][0],
                            xy[tour[i]][1] - xy[tour[(i + 1) % p.dimension]][1],
                        )
                        for i in range(p.dimension)
                    )
                    error = abs(cost - item["cost"])
                    require(
                        item["has_incumbent"] is True
                        and math.isfinite(item["cost"])
                        and error < 1e-10,
                        "独立路线长度不符",
                    )
                    max_error = max(max_error, error)
                    members += 1
                post = read(directory / f"{case}-observed.json")["after"]
                if type(post["foreign_processes"]) is not int or post["foreign_processes"] != 0:
                    deviations.append({"directory": str(directory), "case": case, **post})
            records.append(data)
    for case, row in records[0].items():
        require(
            causal(row["result"]) == causal(records[1][case]["result"]),
            "原生主底座与E2端点轨迹不同",
        )
    for case, row in records[1].items():
        if case.endswith("-repeat"):
            require(
                causal(row["result"]) == causal(records[1][case.removesuffix("-repeat")]["result"]),
                "M10/M01切换后轨迹改变",
            )
    resources = [read(p) for p in args.resources]
    require(
        len(resources) == 2 and all(v["exit_code"] == 0 for v in resources), "实际两次CLI未正常结束"
    )
    report = {
        "status": "passed",
        "scope": "E2 engineering only; no labels or formal TEST",
        "calls": 64,
        "search_tour_evaluations": 64 * LIMIT * 32,
        "valid_members": members,
        "cross_binary_endpoint_pairs": 16,
        "factorial_replay_pairs": 16,
        "max_independent_cost_error": max_error,
        "label_queries": 0,
        "resource_deviations": deviations,
        "manifests_sha256": [file_hash(d / "manifest.json") for d in directories],
        "native_sha256": [m["native_sha256"] for m in manifests],
        "cli_resources": resources,
    }
    atomic_json(args.output / "audit.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("original", "factorial", "audit"))
    for name in ("output", "database", "dataset-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--resources", type=Path, nargs="*")
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(args.output.is_relative_to(PROJECT), "产物必须位于本工作树")
    if args.stage == "audit":
        require(
            args.reference_output is not None and args.resources is not None, "审计缺少原参照与资源"
        )
        audit(args)
    else:
        require(
            args.native_dir is not None and args.gpu_uuid and args.gpu_lock is not None,
            "运行缺少原生/GPU/锁",
        )
        with args.gpu_lock.open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)


if __name__ == "__main__":
    main()
