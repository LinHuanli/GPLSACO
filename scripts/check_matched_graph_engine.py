#!/usr/bin/env python3
"""复用匹配v2真实开发输入，运行两先验×Hard/Escape的完整32×32次数型工程检查。"""

import argparse
import csv
import fcntl
import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from check_hard_engine import gpu_state, require  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.graph_matching import engine_graph_spec  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402


def load_inputs(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["status"] == "passed" and manifest["matching_spec_id"] == 2, "需要已核验匹配v2输入"
    )
    require(manifest["instances"] == 32 and len(manifest["rows"]) == 32, "固定开发面板不完整")
    for name, digest in manifest["sources"].items():
        require(file_hash(PROJECT / name) == digest, "匹配源码身份改变")
    original = PROJECT / manifest["source_input_manifest"]
    require(file_hash(original) == manifest["source_input_manifest_sha256"], "原准备身份改变")
    original_manifest = json.loads(original.read_text())
    records = {}
    for row in manifest["rows"]:
        path = directory / row["path"]
        require(file_hash(path) == row["sha256"], "匹配输入被改写")
        matched = json.loads(path.read_text())
        source = PROJECT / matched["source_instance"]
        require(file_hash(source) == matched["source_instance_sha256"], "原始坐标/共同初解资料改变")
        for name, digest in matched["source_priors_sha256"].items():
            require(file_hash(PROJECT / name) == digest, "原始先验缓存改变")
        key = (row["dimension"], row["index"])
        require(key not in records, "重复开发实例")
        records[key] = {"original": json.loads(source.read_text()), "matched": matched}
    require(set(records) == {(n, i) for n in (500, 1000) for i in range(16)}, "面板成员不完整")
    return manifest, original_manifest, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-lock-directory", type=Path, default=PROJECT / ".tmp")
    args = parser.parse_args()
    args.inputs, args.output, args.native_module = (
        p.resolve() for p in (args.inputs, args.output, args.native_module)
    )
    require(
        all(p.is_relative_to(PROJECT) for p in (args.inputs, args.output, args.native_module)),
        "输入和产物必须位于当前工作树",
    )
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "必须显式固定CUDA UUID")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.gpu_lock_directory / f"worker-{args.gpu_uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = gpu_state(args.gpu_uuid)
        row = next(csv.reader(before["device_csv"].splitlines()))
        require(int(row[2]) <= 1024 and int(row[3]) <= 5, "目标GPU不空闲")
        _, original, records = load_inputs(args.inputs)
        sys.path.insert(0, str(args.native_module.parent))
        native = importlib.import_module("gp_faco_ext")
        require(Path(native.__file__).resolve() == args.native_module, "导入了错误原生模块")
        settings = native.FixedFacoSettings()
        for name, value in original["solver"].items():
            setattr(settings, name, value)
        require(settings.ants == 32, "固定蚂蚁批形状改变")
        program = Program((0,), (4,), feature_spec_id=2).to_dict()
        baseline = BaselinePolicy(
            mne_level=3,
            max_mne_level=3,
            region=3,
            restart_mode="bernoulli",
            restart_probability=1.0,
        ).to_dict()
        manifest = {
            "status": "running",
            "scope": "engineering_development; untrained controls; no test labels",
            "input_manifest_sha256": file_hash(args.inputs / "manifest.json"),
            "database_sha256": original["database_sha256"],
            "native_binary_sha256": file_hash(args.native_module),
            "sources": {
                str(p.relative_to(PROJECT)): file_hash(p)
                for p in (
                    Path(__file__),
                    PROJECT / "scripts/check_hard_engine.py",
                    PROJECT / "python/gp_faco/graph_matching.py",
                )
            },
            "host": socket.gethostname(),
            "gpu_uuid": args.gpu_uuid,
            "before": before,
            "colonies": 32,
            "solver": original["solver"],
            "matching_spec_id": 2,
            "escape_spec_id": 1,
            "wall_clock_limit": None,
            "expected_calls": 34,
            "preparation_scope": "reuse frozen LKH priors and matching v2; "
            "native EndToEnd repeats engine preparation only",
            "registrations": [],
        }
        atomic_json(args.output / "manifest.json", manifest)
        calls = []

        def register(engine, n, constraint, key, xy, graph=None):
            resource = (
                engine.register_problem(key, xy)
                if graph is None
                else engine.register_graph_problem(key, xy, graph)
            )
            manifest["registrations"].append(
                {"dimension": n, "constraint_mode": constraint, "key": key, "resources": resource}
            )

        def call(engine, n, kind, constraint, keys, limit, mode, controller):
            index = len(calls)
            before_call = gpu_state(args.gpu_uuid)
            seeds = np.tile(np.array([17, 29], np.uint64), 16)
            started = time.perf_counter()
            value = (
                engine.evaluate_program_evaluations(keys, seeds, limit, program, mode)
                if controller == "gp"
                else engine.evaluate_baseline_evaluations(keys, seeds, limit, baseline, mode)
            )
            path = args.output / f"call-{index:02}.json"
            record = {
                "dimension": n,
                "kind": kind,
                "constraint_mode": constraint,
                "limit": limit,
                "mode": mode,
                "controller": controller,
                "controller_spec": program if controller == "gp" else baseline,
                "keys": keys.tolist(),
                "seeds": seeds.tolist(),
                "native_wall_seconds": time.perf_counter() - started,
                "before": before_call,
                "result": value,
            }
            atomic_json(path, record)  # 先保留真实返回，再做可能失败的边界观察。
            record["after"] = gpu_state(args.gpu_uuid)
            atomic_json(path, record)
            calls.append({"path": path.name, "sha256": file_hash(path)})
            print(
                json.dumps(
                    {
                        "event": "returned",
                        "call": index,
                        "dimension": n,
                        "kind": kind,
                        "constraint": constraint,
                    }
                ),
                flush=True,
            )

        for n in (500, 1000):
            ordinary = native.FacoBatchEngine(n, 32, settings)
            for i in range(16):
                register(
                    ordinary,
                    n,
                    "unrestricted",
                    i + 1,
                    np.array(records[n, i]["original"]["coordinates"], np.float64),
                )
            call(
                ordinary,
                n,
                "unrestricted",
                "unrestricted",
                np.repeat(np.arange(1, 17, dtype=np.uint64), 2),
                0,
                "cached",
                "gp",
            )
            del ordinary
            for constraint in ("hard", "escape"):
                engine = native.FacoBatchEngine(n, 32, settings, constraint)
                for i in range(16):
                    entry = records[n, i]
                    xy = np.array(entry["original"]["coordinates"], np.float64)
                    for offset, kind in enumerate(("ALPHA", "POPMUSIC")):
                        register(
                            engine,
                            n,
                            constraint,
                            i + 1 + offset * 16,
                            xy,
                            engine_graph_spec(entry["matched"]["graphs"][kind]),
                        )
                for offset, kind in enumerate(("ALPHA", "POPMUSIC")):
                    keys = np.repeat(
                        np.arange(1 + offset * 16, 17 + offset * 16, dtype=np.uint64), 2
                    )
                    for limit, mode, controller in (
                        (0, "cached", "gp"),
                        (128, "cached", "gp"),
                        (64, "cached", "baseline"),
                        (128, "end_to_end", "gp"),
                    ):
                        call(engine, n, kind, constraint, keys, limit, mode, controller)
                del engine
        require(len(calls) == 34, "调用矩阵不完整")
        require(file_hash(args.native_module) == manifest["native_binary_sha256"], "在途二进制改变")
        manifest["status"] = "returned_unverified"
        atomic_json(args.output / "manifest.json", manifest)
        atomic_json(
            args.output / "completed.json", {"status": "returned_unverified", "calls": calls}
        )


if __name__ == "__main__":
    main()
