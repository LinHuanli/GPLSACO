#!/usr/bin/env python3
"""真实spawn worker的记录身份/返回/fitness验证；人工坐标顺序reference仅供接口验收。"""

import argparse
import copy
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from check_e2_behavior import (  # noqa: E402
    LIMIT,
    POLICIES,
    PROGRAM,
    boundary,
    causal,
    guard_labels,
    require,
    selected,
    sources,
)
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import Label, tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.factorial_policy import FactorialPolicy  # noqa: E402
from gp_faco.fitness import score_panel  # noqa: E402
from gp_faco.worker import (  # noqa: E402
    FactorialTask,
    PersistentGpuWorker,
    SolverSettings,
    WorkerProtocol,
    file_hash,
)


def source_identity():
    return sources() | {"scripts/check_behavior_worker.py": file_hash(Path(__file__))}


def protocol_from_manifest(value):
    # implementation_sha256为派生字段；构造后仍须与保存的完整身份精确相同。
    names = {field.name for field in fields(WorkerProtocol) if field.init}
    protocol = WorkerProtocol(
        **{
            key: SolverSettings(**item) if key == "settings" else item
            for key, item in value.items()
            if key in names
        }
    )
    require(json.loads(json.dumps(asdict(protocol))) == value, "序列化worker身份不符")
    return protocol


def producer_sources(manifest):
    current = source_identity()
    commit = manifest["source_commit"]
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "来源commit无效")
    driver = "scripts/check_behavior_worker.py"
    # 允许独立审计器修正自身的读取错误；原执行脚本必须能按执行commit逐字节复原。
    archived = subprocess.check_output(["git", "show", f"{commit}:{driver}"], cwd=PROJECT)
    current[driver] = hashlib.sha256(archived).hexdigest()
    require(current == manifest["sources"], "执行来源与Git归档/当前核心不一致")
    return current


def tasks(source):
    for n in (500, 1000):
        problems = selected(source, n)
        replicas = tuple((p.instance_id, seed) for p in problems for seed in (17, 29))
        for variant in ("M00", "M10", "M01", "M11"):
            for recorded in (False, True):
                yield FactorialTask(
                    f"behavior-worker/{n}/{variant}",
                    PROGRAM,
                    problems,
                    replicas,
                    preparation_mode="cached",
                    evaluation_limit_per_colony=LIMIT,
                    factorial_policy=FactorialPolicy(variant, POLICIES[3]),
                    record_behavior=recorded,
                )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    row = [
        v.strip()
        for v in next(
            csv.reader(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={args.gpu_uuid}",
                        "--query-gpu=name,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).splitlines()
            )
        )
    ]
    native = PROJECT / "build/behavior/gp_faco_ext.so"
    protocol = WorkerProtocol(
        args.gpu_uuid,
        row[0],
        row[1],
        file_hash(native),
        dimensions=(500, 1000),
        colonies=32,
        extension_directory="build/behavior",
        settings=SolverSettings(),
        maximum_registered_per_dimension=32,
    )
    # worker自己持有共享GPU锁；协调进程不抢占同一锁造成父子自锁。
    link = PROJECT / ".tmp" / f"worker-{args.gpu_uuid}.lock"
    require(args.gpu_lock.resolve() != link, "必须显式绑定主项目共享锁")
    if link.is_symlink():
        require(link.resolve() == args.gpu_lock.resolve(), "共享GPU锁目标不同")
    else:
        require(not link.exists(), "已有独立锁不能覆盖")
        link.symlink_to(args.gpu_lock.resolve())
    manifest = {
        "scope": "engineering worker/evaluator plumbing only; no quality inference",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": source_identity(),
        "protocol": asdict(protocol),
        "protocol_sha256": protocol.sha256,
        "database_sha256": file_hash(args.database),
        "host": platform.node(),
        "native_path": str(native),
        "native_sha256": file_hash(native),
        "FE_per_colony": LIMIT,
        "calls": 16,
        "wall_clock_limit": None,
        "formal_test_released": False,
        "evaluator_reference": "coordinate-order tour built outside worker; not dataset labels",
    }
    atomic_json(args.output / "manifest.json", manifest)
    worker = PersistentGpuWorker(protocol)
    try:
        runtime = worker.ready()
        atomic_json(args.output / "runtime.json", runtime)
        with IndexedDataset(args.database, args.dataset_root) as source:
            guard_labels(source)
            for task in tasks(source):
                before = boundary(args.gpu_uuid, runtime["pid"])
                require(before == {"foreign_processes": 0}, "worker任务尚未提交，GPU出现外来进程")
                key = task.task_id(protocol)
                atomic_json(args.output / f"{key}-submitted.json", task.manifest(protocol))
                # 同一Future等待完整FE，不设算法时间预算或超时重试。
                outcome = worker.submit(task).result()
                atomic_json(args.output / f"{key}-return.json", outcome)
                try:
                    after = boundary(args.gpu_uuid, runtime["pid"])
                except Exception as error:
                    after = {"foreign_processes": None, "error": str(error)}
                atomic_json(
                    args.output / f"{key}-resource.json", {"before": before, "after": after}
                )
                require(outcome["status"] == "completed", "实际worker返回失败，保留原始结果")
    finally:
        worker.close()
    require(manifest["sources"] == source_identity(), "执行期间来源改变")
    atomic_json(args.output / "completion.json", {"calls": 16, "FE": 16 * 32 * LIMIT})
    print(json.dumps({"status": "completed", "calls": 16, "FE": 16 * 32 * LIMIT}), flush=True)


def audit(args):
    def read(path):
        return json.loads(path.read_text())

    manifest, runtime = read(args.output / "manifest.json"), read(args.output / "runtime.json")
    require(
        manifest["sources"] == producer_sources(manifest)
        and manifest["database_sha256"] == file_hash(args.database)
        and manifest["native_sha256"] == file_hash(Path(manifest["native_path"])),
        "来源/数据/原生身份改变",
    )
    protocol = protocol_from_manifest(manifest["protocol"])
    require(
        protocol.sha256 == manifest["protocol_sha256"]
        and runtime["protocol_sha256"] == protocol.sha256
        and runtime["binary_sha256"] == manifest["native_sha256"],
        "实际worker身份不符",
    )
    ids, previous, rows, rejected, deviations = set(), {}, 0, 0, []
    with IndexedDataset(args.database, args.dataset_root) as source:
        guard_labels(source)
        for task in tasks(source):
            key = task.task_id(protocol)
            ids.add(key)
            require(
                read(args.output / f"{key}-submitted.json")
                == json.loads(json.dumps(task.manifest(protocol))),
                "原提交任务改变",
            )
            outcome = read(args.output / f"{key}-return.json")
            require(outcome["worker_pid"] == runtime["pid"], "实际worker被替换")
            labels = {
                p.instance_id: Label(
                    tuple(range(p.dimension)), tour_cost(p, tuple(range(p.dimension)))
                )
                for p in task.problems
            }
            fitness = score_panel(task, protocol, outcome, labels)
            require(not fitness.failed and len(fitness.members) == 32, "独立接口fitness核验失败")
            record = read(args.output / f"{key}-resource.json")
            require(record["before"] == {"foreign_processes": 0}, "提交时GPU非空闲")
            if record["after"] != {"foreign_processes": 0}:
                deviations.append({"task_id": key, **record})
            native = outcome["native_result"]
            if task.record_behavior:
                require(
                    causal(native) == previous[task.occurrence_id], "worker记录开关改变求解结果"
                )
                rows += len(native["behavior"]["rows"])
                for mutation in ("missing", "truncated"):
                    bad = copy.deepcopy(outcome)
                    if mutation == "missing":
                        del bad["native_result"]["behavior"]
                    else:
                        bad["native_result"]["behavior"]["rows"].pop()
                    require(score_panel(task, protocol, bad, labels).failed, "缺失行为记录未被拒绝")
                    rejected += 1
            else:
                previous[task.occurrence_id] = causal(native)
    require(
        {p.name.removesuffix("-return.json") for p in args.output.glob("*-return.json")} == ids
        and len(ids) == 16
        and read(args.output / "completion.json") == {"calls": 16, "FE": 16 * 32 * LIMIT},
        "任务/FE覆盖不完整",
    )
    resource = read(args.resources)
    require(resource["exit_code"] == 0, "实际worker CLI未完成")
    report = {
        "status": "passed",
        "scope": manifest["scope"],
        "calls": 16,
        "FE": 16 * 32 * LIMIT,
        "valid_tours": 512,
        "behavior_rows": rows,
        "toggle_pairs": 8,
        "rejected_tampered_outcomes": rejected,
        "dataset_label_queries": 0,
        "resource_deviations": deviations,
        "cli_resources": resource,
        "manifest_sha256": file_hash(args.output / "manifest.json"),
        "producer_commit": manifest["source_commit"],
        "auditor_sha256": file_hash(Path(__file__)),
    }
    atomic_json(args.output / "audit.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "audit"))
    for name in ("output", "database", "dataset-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--resources", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(args.output.is_relative_to(PROJECT), "产物必须在本工作树")
    if args.stage == "run":
        require(args.gpu_uuid and args.gpu_lock is not None, "缺少GPU与共享锁")
        run(args)
    else:
        require(args.resources is not None, "审计缺少真实CLI资源")
        audit(args)


if __name__ == "__main__":
    main()
