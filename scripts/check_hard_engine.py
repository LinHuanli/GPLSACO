#!/usr/bin/env python3
"""真实开发数据的先验准备与32×32 Hard验收；原始返回先落盘，独立audit后报告。"""

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.candidate_prior import PriorSettings, prepare_candidates  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.graph_matching import engine_graph_spec, match_graphs  # noqa: E402
from gp_faco.program_ir import Program  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gpu_state(uuid):
    device = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={uuid}",
            "--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = list(
        csv.reader(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                text=True,
            ).splitlines()
        )
    )
    foreign = [
        row for row in processes if row and row[0].strip() == uuid and int(row[1]) != os.getpid()
    ]
    require(not foreign, "目标GPU出现外来计算进程")
    return {"device_csv": device, "processes": processes, "pid": os.getpid()}


def prepare(args, native, settings):
    manifest = {
        "status": "preparing",
        "scope": "engineering_development; no efficacy selection or test labels",
        "database_sha256": file_hash(args.database),
        "native_binary_sha256": file_hash(args.native_module),
        "sources": {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in (
                Path(__file__),
                PROJECT / "python/gp_faco/graph_matching.py",
                PROJECT / "python/gp_faco/candidate_prior.py",
            )
        },
        "solver": {
            name: getattr(settings, name)
            for name in (
                "ants",
                "primary_width",
                "backup_width",
                "ls_width",
                "beta",
                "retention",
                "p_best",
                "epoch_source_probability",
                "ls_evaluation_limit",
                "initial_ls_evaluation_limit",
            )
        },
        "wall_clock_limit": None,
        "files": [],
    }
    atomic_json(args.output / "manifest.json", manifest)
    with IndexedDataset(args.database, args.dataset_root) as source:
        for n in (500, 1000):
            ids = sorted(source.record_ids("development", n))[:16]
            require(len(ids) == 16, "需要每规模16个不同开发实例")
            for i, identity in enumerate(ids):
                started = time.perf_counter()
                problem = source.load_instance(identity)
                xy = np.array(problem.coordinates, dtype=np.float64)
                common = native.prepare_common_initial(xy, settings)
                priors = {
                    kind: prepare_candidates(
                        problem,
                        PriorSettings(kind=kind),
                        args.candidate_binary,
                        args.output / f"prior-{n}-{i}-{kind}",
                    )
                    for kind in ("ALPHA", "POPMUSIC")
                }
                graphs = match_graphs(problem, tuple(common["tour"]), priors)
                path = args.output / f"{n}-{i}.json"
                atomic_json(
                    path,
                    {
                        "instance_id": identity,
                        "coordinates": problem.coordinates,
                        "distance_spec": problem.distance_spec,
                        "common": common,
                        "graphs": graphs,
                        "preparation_wall_seconds": time.perf_counter() - started,
                    },
                )
                manifest["files"].append(
                    {"path": path.name, "sha256": file_hash(path), "dimension": n, "index": i}
                )
                atomic_json(args.output / "manifest.json", manifest)
                print(json.dumps({"event": "prepared", "dimension": n, "index": i}), flush=True)
    manifest["status"] = "complete"
    atomic_json(args.output / "manifest.json", manifest)


def solve(args, native, settings):
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "complete", "准备未完成")
    require(file_hash(args.native_module) == manifest["native_binary_sha256"], "原生身份改变")
    for name, digest in manifest["sources"].items():
        require(file_hash(PROJECT / name) == digest, "准备源码身份改变")
    for name, value in manifest["solver"].items():
        require(getattr(settings, name) == value, "共同求解设置改变")
    before = gpu_state(args.gpu_uuid)
    row = next(csv.reader(before["device_csv"].splitlines()))
    require(int(row[2]) <= 1024 and int(row[3]) <= 5, "目标GPU不空闲")
    atomic_json(
        args.output / "manifest.json",
        {
            "input_manifest_sha256": file_hash(manifest_path),
            "native_binary_sha256": manifest["native_binary_sha256"],
            "gpu_uuid": args.gpu_uuid,
            "before": before,
            "colonies": 32,
            "ants": 32,
            "scope": "engineering_development",
            "wall_clock_limit": None,
        },
    )
    program = Program((0,), (4,), feature_spec_id=2).to_dict()
    baseline = BaselinePolicy(
        mne_level=3,
        max_mne_level=3,
        region=3,
        restart_mode="bernoulli",
        restart_probability=1.0,
    ).to_dict()
    calls = []

    def call(engine, n, kind, keys, limit, mode, controller):
        index = len(calls)
        before_call = gpu_state(args.gpu_uuid)
        started = time.perf_counter()
        kwargs = (keys, np.tile(np.array([17, 29], np.uint64), 16), limit)
        value = (
            engine.evaluate_program_evaluations(*kwargs, program, mode)
            if controller == "gp"
            else engine.evaluate_baseline_evaluations(*kwargs, baseline, mode)
        )
        path = args.output / f"call-{index:02}.json"
        record = {
            "dimension": n,
            "kind": kind,
            "limit": limit,
            "mode": mode,
            "controller": controller,
            "controller_spec": program if controller == "gp" else baseline,
            "keys": keys.tolist(),
            "seeds": kwargs[1].tolist(),
            "native_wall_seconds": time.perf_counter() - started,
            "before": before_call,
            "result": value,
        }
        atomic_json(path, record)  # 返回与GPU边界查询独立，查询失败不能抹去已完成求解。
        record["after"] = gpu_state(args.gpu_uuid)
        atomic_json(path, record)
        calls.append({"path": path.name, "sha256": file_hash(path)})
        print(
            json.dumps({"event": "returned", "call": index, "dimension": n, "kind": kind}),
            flush=True,
        )

    for n in (500, 1000):
        entries = [e for e in manifest["files"] if e["dimension"] == n]
        require([e["index"] for e in entries] == list(range(16)), "固定开发面板改变")
        ordinary = native.FacoBatchEngine(n, 32, settings)
        hard = native.FacoBatchEngine(n, 32, settings, "hard")
        for entry in entries:
            path = args.inputs / entry["path"]
            require(file_hash(path) == entry["sha256"], "输入资料身份改变")
            problem = json.loads(path.read_text())
            xy = np.array(problem["coordinates"], dtype=np.float64)
            key = entry["index"] + 1
            ordinary.register_problem(key, xy)
            for offset, kind in enumerate(("ALPHA", "POPMUSIC")):
                hard.register_graph_problem(
                    key + offset * 16, xy, engine_graph_spec(problem["graphs"][kind])
                )
        call(
            ordinary,
            n,
            "unrestricted",
            np.repeat(np.arange(1, 17, dtype=np.uint64), 2),
            0,
            "cached",
            "gp",
        )
        del ordinary
        for offset, kind in enumerate(("ALPHA", "POPMUSIC")):
            keys = np.repeat(np.arange(1 + offset * 16, 17 + offset * 16, dtype=np.uint64), 2)
            for limit, mode, controller in (
                (0, "cached", "gp"),
                (128, "cached", "gp"),
                (64, "cached", "baseline"),
                (128, "end_to_end", "gp"),
            ):
                call(hard, n, kind, keys, limit, mode, controller)
        del hard
    atomic_json(args.output / "completed.json", {"calls": calls, "status": "returned_unverified"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "solve"))
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(args.output.is_relative_to(PROJECT), "产物必须位于项目内")
    if args.stage == "prepare":
        require(all((args.database, args.dataset_root, args.candidate_binary)), "缺少开发准备来源")
    else:
        require(args.inputs and args.gpu_uuid, "缺少冻结输入或设备UUID")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "设备绑定未显式固定")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.native_module.resolve().parent))
    native = importlib.import_module("gp_faco_ext")
    require(Path(native.__file__).resolve() == args.native_module.resolve(), "导入了错误原生模块")
    settings = native.FixedFacoSettings()
    settings.ants = 32
    (prepare if args.stage == "prepare" else solve)(args, native, settings)


if __name__ == "__main__":
    main()
