#!/usr/bin/env python3
"""真实开发实例上的状态分叉工程验收；固定FE，不读取标签或正式TEST。"""

import argparse
import csv
import datetime
import fcntl
import hashlib
import importlib
import json
import math
import os
import socket
import sqlite3
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

import numpy as np  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402

PROGRAM = {
    "ir_version": 1,
    "numeric_spec_id": 1,
    "feature_spec_id": 2,
    "opcode": [0, 0, 2, 0, 0, 3, 0, 4, 2],
    "operand": [4, 8, 0, 0, 2, 0, 5, 0, 0],
    "constant_bits": [],
}
CONTINUATION = BaselinePolicy(mne_level=2, max_mne_level=2, region=1).to_dict()
SOURCE_BASELINE = BaselinePolicy(
    mne_level=1, max_mne_level=1, region=2, restart_mode="periodic", restart_period=3
).to_dict()
PHASES = {
    "reference": (256, 32),
    "capture": (128, 32),
    "poison": (64, 32),
    "continue": (128, 32),
    "seed17-mne2": (128, 32),
    "seed17-mne16": (128, 32),
    "seed17-mne2-repeat": (128, 32),
    "seed29-mne2": (128, 32),
    "seed29-mne16": (128, 32),
    "single17-mne2": (128, 1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources():
    paths = [Path(__file__), PROJECT / "CMakeLists.txt"]
    for directory in ("cpp", "cuda"):
        paths += [
            p
            for p in (PROJECT / directory).rglob("*")
            if p.suffix in (".cpp", ".cu", ".hpp", ".cuh")
        ]
    paths += [
        PROJECT / "python/gp_faco" / (n + ".py")
        for n in ("baseline_policy", "data", "dataset_index", "checkpoint", "worker")
    ]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(paths)}


def observe(uuid):
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    row = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=" + uuid,
                    "--query-gpu=name,driver_version,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        )
    )
    model, driver, memory, utilization = [v.strip() for v in row]
    return {
        "gpu_uuid": uuid,
        "gpu_model": model,
        "driver_version": driver,
        "memory_used_mib": int(memory),
        "utilization_percent": int(utilization),
        "foreign_processes": sum(
            bool(r and r[0].strip() == uuid and int(r[1]) != os.getpid())
            for r in csv.reader(apps.splitlines())
        ),
    }


def guard_labels(source):
    def authorize(action, first, *_):
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and first == "labels"
            else sqlite3.SQLITE_OK
        )

    source.connection.set_authorizer(authorize)


def safe_observation(uuid):
    try:
        return observe(uuid)
    except Exception as error:
        return {"foreign_processes": None, "error": f"{type(error).__name__}: {error}"}


def wait_before_submission(uuid, directory, phase):
    observation = safe_observation(uuid)
    while observation["foreign_processes"] != 0:
        with (directory / "resource-waits.jsonl").open("a") as stream:
            stream.write(json.dumps({"phase": phase, "observation": observation}) + "\n")
        time.sleep(5)
        observation = safe_observation(uuid)
    return observation


def solutions(result):
    return [(item["tour"], item["cost"]) for item in result["items"]]


def same_result(a, b):
    return solutions(a) == solutions(b) and a["control_states"] == b["control_states"]


def run(args, native):
    output = args.output.resolve()
    require(output.is_relative_to(PROJECT) and not output.exists(), "需使用工作树内全新工程输出")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.gpu_uuid, "须固定实际GPU UUID")
    output.mkdir(parents=True)
    before = observe(args.gpu_uuid)
    require(
        before["foreign_processes"] == 0
        and before["memory_used_mib"] <= 1024
        and before["utilization_percent"] <= 5,
        "工程GPU当前不空闲",
    )
    manifest = {
        "engineering_spec_id": 1,
        "scope": "state-fork engineering; no formal mechanism inference",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "sources": sources(),
        "native_sha256": file_hash(Path(native.__file__)),
        "database_sha256": file_hash(args.database),
        "device_before": before,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "dimensions": [500, 1000],
        "instances_per_dimension": 16,
        "solver_seeds": [17, 29],
        "program": PROGRAM,
        "source_baseline": SOURCE_BASELINE,
        "continuation": CONTINUATION,
        "source_evaluations": 256,
        "capture_after_evaluations": 128,
        "additional_evaluations": 128,
        "ants": 32,
        "colonies": 32,
        "no_wall_clock_limit": True,
        "formal_test_released": False,
    }
    atomic_json(output / "manifest.json", manifest)
    calls = []
    with IndexedDataset(args.database, args.dataset_root) as source:
        guard_labels(source)
        for n in manifest["dimensions"]:
            ids = source.record_ids("development", n)[:16]
            require(len(ids) == 16, "开发实例不足")
            instances = {identifier: source.load_instance(identifier) for identifier in ids}
            keys_by_id = {identifier: int(identifier[:16], 16) for identifier in ids}
            require(len(set(keys_by_id.values())) == 16, "实例key碰撞")
            replicas = [(identifier, seed) for identifier in ids for seed in (17, 29)]
            keys = np.array([keys_by_id[i] for i, _ in replicas], dtype=np.uint64)
            seeds = np.array([s for _, s in replicas], dtype=np.uint64)

            def make(members=replicas, n=n, keys_by_id=keys_by_id, instances=instances):
                settings = native.FixedFacoSettings()
                engine = native.FacoBatchEngine(n, len(members), settings)
                fees = {}
                for identifier in sorted({i for i, _ in members}):
                    fees[identifier] = engine.register_problem(
                        keys_by_id[identifier],
                        np.asarray(instances[identifier].coordinates, dtype=np.float64),
                    )
                return engine, fees

            for kind, argument in (("program", PROGRAM), ("baseline", SOURCE_BASELINE)):
                case = output / f"{n}-{kind}"
                case.mkdir()
                engine, fees = make()
                atomic_json(
                    case / "input.json",
                    {
                        "dimension": n,
                        "kind": kind,
                        "replicas": replicas,
                        "keys_by_id": keys_by_id,
                        "registration": fees,
                    },
                )

                def call(phase, operation, members=replicas, case=case):
                    admission = wait_before_submission(args.gpu_uuid, case, phase)
                    atomic_json(
                        case / (phase + "-submitted.json"),
                        {
                            "phase": phase,
                            "replicas": members,
                            "planned_FE_per_colony": PHASES[phase][0],
                            "admission": admission,
                        },
                    )
                    value = operation()
                    result = value["evaluation"] if phase == "capture" else value
                    raw = {"phase": phase, "replicas": members, "native_result": result}
                    # 资源查询失败不能丢弃已经返回的原生评价。
                    atomic_json(case / (phase + "-raw.json"), raw)
                    atomic_json(
                        case / (phase + "-return.json"),
                        {
                            **raw,
                            "device_after": safe_observation(args.gpu_uuid),
                        },
                    )
                    calls.append(
                        {"case": case.name, "phase": phase, "FE": result["total_tour_evaluations"]}
                    )
                    print(json.dumps(calls[-1]), flush=True)
                    return value

                reference = call(
                    "reference",
                    partial(
                        getattr(engine, f"evaluate_{kind}_evaluations"), keys, seeds, 256, argument
                    ),
                )
                captured = call(
                    "capture",
                    partial(
                        getattr(engine, f"capture_{kind}_state"), keys, seeds, 256, 128, argument
                    ),
                )
                blob = captured["state"].to_bytes()
                (case / "snapshot.bin").write_bytes(blob)
                metadata = captured["state"].describe()
                atomic_json(
                    case / "snapshot.json",
                    {"sha256": hashlib.sha256(blob).hexdigest(), "description": metadata},
                )
                restored = native.CountedState.from_bytes((case / "snapshot.bin").read_bytes())
                del engine
                engine, fees = make()
                atomic_json(case / "rebuild-registration.json", fees)
                call(
                    "poison",
                    partial(engine.evaluate_baseline_evaluations, keys, seeds, 64, CONTINUATION),
                )
                continued = call(
                    "continue",
                    partial(getattr(engine, f"continue_{kind}_state"), restored, 128, argument),
                )
                require(same_result(reference, continued), "真实实例的完整恢复结果改变")
                for field in (
                    "total_tour_evaluations",
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                ):
                    require(
                        captured["evaluation"][field] + continued[field] == reference[field],
                        "真实恢复FE或工作量改变",
                    )
                branches = {}
                for seed, mne, suffix in (
                    (17, 2, ""),
                    (17, 16, ""),
                    (17, 2, "-repeat"),
                    (29, 2, ""),
                    (29, 16, ""),
                ):
                    phase = f"seed{seed}-mne{mne}{suffix}"
                    branches[phase] = call(
                        phase,
                        partial(
                            engine.continue_baseline_state,
                            restored,
                            128,
                            CONTINUATION,
                            fork_seed=seed,
                            region=0,
                            mne=mne,
                        ),
                    )
                require(
                    same_result(branches["seed17-mne2"], branches["seed17-mne2-repeat"]),
                    "真实实例A-B-A重放改变",
                )
                selected_members = [replicas[17]]
                solo, solo_fees = make(selected_members)
                atomic_json(case / "single-registration.json", solo_fees)
                selected = restored.select_colony(17)
                one = call(
                    "single17-mne2",
                    partial(
                        solo.continue_baseline_state,
                        selected,
                        128,
                        CONTINUATION,
                        fork_seed=17,
                        region=0,
                        mne=2,
                    ),
                    selected_members,
                )
                require(
                    solutions(one)[0] == solutions(branches["seed17-mne2"])[17],
                    "真实单colony提取改变结果",
                )
                require(restored.to_bytes() == blob, "分支改写原始完整状态")
                del engine, solo, restored, selected, captured
    require(
        manifest["sources"] == sources()
        and manifest["native_sha256"] == file_hash(Path(native.__file__)),
        "执行期间来源改变",
    )
    require(len(calls) == 40, "工程调用矩阵不完整")
    atomic_json(
        output / "result.json",
        {
            "status": "completed",
            "calls": calls,
            "total_FE": sum(c["FE"] for c in calls),
            "label_queries": 0,
            "formal_mechanism_result": False,
        },
    )


def audit(args, native):
    directory = args.output.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["sources"] == sources()
        and manifest["native_sha256"] == file_hash(Path(native.__file__)),
        "来源改变",
    )
    require(manifest["database_sha256"] == file_hash(args.database), "索引改变")
    resources = json.loads(args.resources.read_text())
    require(resources["exit_code"] == 0, "工程CLI未正常终止")
    completed = json.loads((directory / "result.json").read_text())
    require(
        completed["status"] == "completed" and len(completed["calls"]) == 40, "完整实际调用缺失"
    )
    total_FE, members, max_error, resource_deviations = 0, 0, 0.0, 0
    with IndexedDataset(args.database, args.dataset_root) as source:
        guard_labels(source)
        for n in (500, 1000):
            ids = source.record_ids("development", n)[:16]
            instances = {i: source.load_instance(i) for i in ids}
            expected_members = [[i, seed] for i in ids for seed in (17, 29)]
            for kind in ("program", "baseline"):
                case = directory / f"{n}-{kind}"
                inputs = json.loads((case / "input.json").read_text())
                require(
                    inputs["replicas"] == expected_members and inputs["dimension"] == n,
                    "实际开发面板改变",
                )
                metadata = json.loads((case / "snapshot.json").read_text())
                require(file_hash(case / "snapshot.bin") == metadata["sha256"], "快照原字节改变")
                snapshot = native.CountedState.from_bytes((case / "snapshot.bin").read_bytes())
                require(snapshot.describe() == metadata["description"], "快照元数据与真实字节不符")
                require(
                    metadata["description"]["progress_evaluation_limit"] == 256
                    and metadata["description"]["completed_batches"] == 4,
                    "源progress预算或切点错误",
                )
                require(
                    {p.name.removesuffix("-return.json") for p in case.glob("*-return.json")}
                    == set(PHASES),
                    "原始返回矩阵不完整或额外",
                )
                records = {}
                for phase, (FE, count) in PHASES.items():
                    raw = json.loads((case / (phase + "-return.json")).read_text())
                    require(
                        json.loads((case / (phase + "-raw.json")).read_text())
                        == {k: v for k, v in raw.items() if k != "device_after"},
                        "独立原始返回不符",
                    )
                    submitted = json.loads((case / (phase + "-submitted.json")).read_text())
                    require(submitted["admission"]["foreign_processes"] == 0, "未在空闲GPU上准入")
                    result = raw["native_result"]
                    records[phase] = result
                    panel = [expected_members[17]] if count == 1 else expected_members
                    require(
                        raw["replicas"] == panel and len(result["items"]) == count,
                        "实际返回成员改变",
                    )
                    require(
                        result["budget_seconds"] is None
                        and result["evaluation_limit_per_colony"] == FE
                        and result["completed_tour_evaluations_per_colony"] == FE
                        and result["total_tour_evaluations"] == FE * count
                        and result["completed_batches"] == FE // 32,
                        "实际FE账目不符",
                    )
                    require(
                        result["charged_seconds"]
                        == result["overrun_seconds"]
                        == result["discarded_batches"]
                        == 0,
                        "次数入口出现墙钟扣费或丢弃",
                    )
                    resource_deviations += raw["device_after"]["foreign_processes"] != 0
                    for item, (identifier, _) in zip(result["items"], panel, strict=True):
                        tour = item["tour"]
                        instance = instances[identifier]
                        require(
                            item["has_incumbent"] and sorted(tour) == list(range(n)), "非法完整tour"
                        )
                        cost = math.fsum(instance.distance(tour[i - 1], tour[i]) for i in range(n))
                        error = abs(cost - item["cost"])
                        max_error = max(max_error, error)
                        require(
                            math.isfinite(item["cost"]) and error <= 1e-8 + 1e-12 * cost,
                            "独立成本核验失败",
                        )
                        members += 1
                    total_FE += result["total_tour_evaluations"]
                require(same_result(records["reference"], records["continue"]), "独立恢复对照失败")
                require(
                    same_result(records["seed17-mne2"], records["seed17-mne2-repeat"]),
                    "独立A-B-A对照失败",
                )
                require(
                    solutions(records["single17-mne2"])[0] == solutions(records["seed17-mne2"])[17],
                    "独立单colony对照失败",
                )
                for field in (
                    "total_tour_evaluations",
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                ):
                    require(
                        records["capture"][field] + records["continue"][field]
                        == records["reference"][field],
                        "独立前后缀工作量不符",
                    )
    require(
        total_FE == completed["total_FE"] == 156160 and members == 1156, "工程总矩阵或成员计数不符"
    )
    result = {
        "status": "passed",
        "calls": 40,
        "FE": total_FE,
        "members": members,
        "max_independent_cost_error": max_error,
        "resource_deviations": resource_deviations,
        "label_queries": 0,
        "native_sha256": manifest["native_sha256"],
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "resources_sha256": file_hash(args.resources),
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "formal_mechanism_result": False,
        "formal_test_released": False,
    }
    target = directory / "audit.json"
    require(not target.exists(), "独立审计输出已存在")
    atomic_json(target, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "audit"))
    for name in ("output", "native-dir", "database", "dataset-root", "gpu-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    require(args.native_dir.resolve().is_relative_to(PROJECT), "只能使用隔离工作树构建")
    sys.path.insert(0, str(args.native_dir.resolve()))
    native = importlib.import_module("gp_faco_ext")
    require(
        Path(native.__file__).resolve().parent == args.native_dir.resolve(),
        "载入了其他工作树native",
    )
    if args.stage == "run":
        with args.gpu_lock.open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args, native)
    else:
        require(args.resources is not None, "独立审计需要原CLI终态资源记录")
        audit(args, native)


if __name__ == "__main__":
    main()
