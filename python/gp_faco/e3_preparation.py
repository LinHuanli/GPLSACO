"""E3成对图的CPU准备事务；冻结输入、逐项恢复，完全不读取标签。"""

from __future__ import annotations

import fcntl
import importlib
import json
import math
import os
import resource
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gp_faco.candidate_prior import PriorSettings, parse_candidates, prepare_candidates
from gp_faco.checkpoint import atomic_json
from gp_faco.data import Instance, tour_cost
from gp_faco.dataset_index import IndexedDataset
from gp_faco.e3_protocol import preparation_jobs, require, training_panels, validate_config
from gp_faco.graph_catalog import GraphCatalog, project_file
from gp_faco.graph_matching import GraphSettings, engine_graph_spec, match_graphs
from gp_faco.worker import PROJECT, content_hash, coordinate_hash, file_hash

KINDS = ("ALPHA", "POPMUSIC")


def checked_json(path, digest=None):
    if digest is not None:
        require(file_hash(path) == digest, f"文件身份改变: {path}")
    return json.loads(path.read_text())


@contextmanager
def exclusive_lock(path):
    """非阻塞文件锁只阻止重复准备；不能把获得父锁解释成未知子进程已退出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_freeze(directory, database):
    plan = checked_json(directory / "plan.json")
    require(
        plan["sha256"] == content_hash({k: v for k, v in plan.items() if k != "sha256"}),
        "E3冻结计划内容摘要不符",
    )
    validate_config(plan["config"])
    require(not plan["formal_test_released"], "准备阶段不得已放行TEST")
    config = checked_json(project_file(PROJECT, plan["config_path"]), plan["config_sha256"])
    require(config == plan["config"], "冻结配置与展开设置不一致")
    require(file_hash(database) == plan["database_sha256"], "准备数据库与冻结身份不符")
    for name, digest in plan["preparation_sources"].items():
        require(file_hash(project_file(PROJECT, name)) == digest, f"准备源码改变: {name}")
    for prefix in ("native_binary", "candidate_binary", "reuse_catalog"):
        require(
            file_hash(project_file(PROJECT, plan[prefix + "_path"])) == plan[prefix + "_sha256"],
            f"准备依赖改变: {prefix}",
        )
    candidate = project_file(PROJECT, plan["candidate_binary_path"])
    checked_json(candidate.parent / "build-manifest.json", plan["candidate_build_manifest_sha256"])
    values = {
        name: checked_json(directory / name, digest) for name, digest in plan["files"].items()
    }
    require(
        values["training_panels.json"] == training_panels(config, values["members.json"]),
        "独立面板流重放不同",
    )
    jobs = preparation_jobs(
        config, values["members.json"], values["training_panels.json"], values["development.json"]
    )
    require(
        jobs == values["preparation_jobs.json"] and len(jobs) == plan["preparation_instances"],
        "正式准备成员表不同",
    )
    return plan, jobs


def host_identity():
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        ),
        "process_start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        "utc_started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def check_common(problem, common):
    require(
        set(common) == {"tour", "cost", "cheap_seconds", "preparation_seconds"}, "共同初解字段不符"
    )
    cost = tour_cost(problem, common["tour"])
    require(
        math.isfinite(common["cost"])
        and math.isclose(cost, common["cost"], rel_tol=1e-12, abs_tol=1e-10),
        "共同初解成本不符",
    )
    require(
        all(
            type(common[k]) in (float, int) and math.isfinite(common[k]) and common[k] >= 0
            for k in ("cheap_seconds", "preparation_seconds")
        ),
        "共同准备资源无效",
    )
    return abs(cost - common["cost"])


def read_prior(problem, settings, binary, directory, *, finish_receipt=False):
    """只允许已有零退出收据的纯解析恢复；缺收据或非零退出从不重新调用LKH。"""
    receipt = directory / "resources.json"
    require(receipt.is_file(), f"原生尝试缺终态收据，须先独立核查原进程，不能自动重跑: {directory}")
    resources = checked_json(receipt)
    require(
        type(resources["exit_code"]) is int and resources["exit_code"] == 0,
        f"候选生成已有失败，保留现场，不按质量重试: {directory}",
    )
    manifest = checked_json(directory / "manifest.json")
    expected = {
        "instance_id": problem.instance_id,
        "coordinate_sha256": coordinate_hash(problem),
        "dimension": problem.dimension,
        "settings": asdict(settings),
        "binary_sha256": file_hash(binary),
        "build_manifest_sha256": file_hash(binary.parent / "build-manifest.json"),
        "module_sha256": file_hash(PROJECT / "python/gp_faco/candidate_prior.py"),
        "wall_clock_limit": None,
        "problem_file_sha256": file_hash(directory / "problem.tsp"),
        "parameter_file_sha256": file_hash(directory / "parameters.par"),
    }
    require(manifest == expected, "已有候选尝试的实例、设置或输入文件改变")
    require(resources["log_sha256"] == file_hash(directory / "native.log"), "原生输出日志改变")
    raw = checked_json(directory / "native-candidates.json")
    prior = parse_candidates(problem, settings, raw)
    prior.update(
        manifest_sha256=content_hash(manifest),
        native_output_sha256=file_hash(directory / "native-candidates.json"),
    )
    path = directory / "prior.json"
    if path.exists():
        require(checked_json(path) == prior, "已有候选解析产物与原生输出不符")
    else:
        require(finish_receipt, "缺少已提交先验收据")
        atomic_json(path, prior)
    files = {
        str(p.relative_to(PROJECT)): file_hash(p)
        for p in sorted(directory.iterdir())
        if p.is_file()
    }
    return prior, {
        "directory": str(directory.relative_to(PROJECT)),
        "files": files,
        "resources": resources,
    }


def ensure_prior(problem, settings, binary, directory):
    if not directory.exists():
        # 目录由prepare_candidates在启动原生程序之前独占创建；已有目录永不传给它。
        prepare_candidates(problem, settings, binary, directory)
    return read_prior(problem, settings, binary, directory, finish_receipt=True)


def reusable_entry(problem, plan):
    catalog = GraphCatalog(PROJECT, plan["reuse_catalog_path"], plan["reuse_catalog_sha256"])
    try:
        entry = catalog.entry(problem)
    except ValueError:
        # 同ID的坐标改变必须拒绝；真正不在旧目录中才属于新准备。
        require(problem.instance_id not in catalog._entries, "旧目录中同ID坐标改变")
        return None
    for kind in KINDS:
        catalog.load_graph(problem, kind)
    return entry


def reuse_payload(problem, entry, plan):
    path = project_file(PROJECT, entry.path)
    matched = checked_json(path, entry.file_sha256)
    original_path = project_file(PROJECT, matched["source_instance"])
    original = checked_json(original_path, matched["source_instance_sha256"])
    previous = Instance(
        original["instance_id"],
        tuple(map(tuple, original["coordinates"])),
        original["distance_spec"],
    )
    require(previous == problem, "旧缓存与数据库无标签坐标不符")
    priors, evidence = {}, {}
    binary = project_file(PROJECT, plan["candidate_binary_path"])
    for name, digest in matched["source_priors_sha256"].items():
        p = project_file(PROJECT, name)
        value = checked_json(p, digest)
        kind = value["settings"]["kind"]
        require(kind in KINDS and kind not in priors, "旧缓存先验种类重复")
        priors[kind], evidence[kind] = read_prior(
            problem, PriorSettings(**plan["config"]["priors"][kind]), binary, p.parent
        )
    graphs = match_graphs(
        problem, tuple(original["common"]["tour"]), priors, GraphSettings(**plan["config"]["graph"])
    )
    require(graphs == matched["graphs"], "已审计旧图不能按新准备重新解释")
    return (
        original["common"],
        graphs,
        evidence,
        {
            str(path.relative_to(PROJECT)): entry.file_sha256,
            str(original_path.relative_to(PROJECT)): matched["source_instance_sha256"],
        },
    )


def verify_completed(directory, job, plan):
    receipt = checked_json(directory / "complete.json")
    require(
        receipt["sha256"] == content_hash({k: v for k, v in receipt.items() if k != "sha256"}),
        "实例完成收据内容摘要不符",
    )
    require(
        receipt["job"] == job
        and receipt["plan_sha256"] == plan["sha256"]
        and receipt["status"] == "complete",
        "实例完成收据身份改变",
    )
    for name, digest in receipt["files"].items():
        require(file_hash(project_file(PROJECT, name)) == digest, f"已完成实例文件改变: {name}")
    return receipt


def initialize_worker(plan, database, dataset_root, output):
    global _context
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "图准备必须明确禁用GPU可见性")
    native_path = project_file(PROJECT, plan["native_binary_path"])
    sys.path.insert(0, str(native_path.parent))
    native = importlib.import_module("gp_faco_ext")
    require(
        Path(native.__file__).resolve() == native_path
        and file_hash(native_path) == plan["native_binary_sha256"],
        "共同初解导入了错误原生模块",
    )
    settings = native.FixedFacoSettings()
    for name, value in plan["config"]["solver"].items():
        setattr(settings, name, value)
    _context = (
        plan,
        IndexedDataset(Path(database), Path(dataset_root)),
        Path(output),
        native,
        settings,
    )


def run_job(job):
    plan, source, output, native, settings = _context
    directory = output / "instances" / f"{job['index']:05d}"
    with exclusive_lock(directory / ".lock"):
        if (directory / "complete.json").exists():
            receipt = verify_completed(directory, job, plan)
            return {
                "index": job["index"],
                "mode": receipt["mode"],
                "resumed": True,
                "receipt_sha256": file_hash(directory / "complete.json"),
            }
        problem = source.load_instance(job["instance_id"])
        require(problem.dimension == job["dimension"], "准备实例规模错误")
        started, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        attempts = directory / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = attempts / f"{len(list(attempts.glob('*.json'))):04d}.json"
        atomic_json(attempt, {"job": job, "plan_sha256": plan["sha256"], **host_identity()})
        entry = reusable_entry(problem, plan)
        original_files = {}
        if entry is not None:
            common, graphs, priors, original_files = reuse_payload(problem, entry, plan)
            mode = "reuse_verified_development_cache"
        else:
            common_path = directory / "common.json"
            if common_path.exists():
                saved = checked_json(common_path)
                require(
                    saved["plan_sha256"] == plan["sha256"]
                    and saved["coordinate_sha256"] == coordinate_hash(problem)
                    and saved["sha256"]
                    == content_hash({k: v for k, v in saved.items() if k != "sha256"}),
                    "已保存共同初解身份改变",
                )
                common = saved["common"]
            else:
                common = native.prepare_common_initial(
                    np.array(problem.coordinates, dtype=np.float64), settings
                )
                check_common(problem, common)
                saved = {
                    "common": common,
                    "coordinate_sha256": coordinate_hash(problem),
                    "plan_sha256": plan["sha256"],
                    "attempt": str(attempt.relative_to(PROJECT)),
                }
                atomic_json(common_path, {**saved, "sha256": content_hash(saved)})
            values, priors = {}, {}
            for kind in KINDS:
                values[kind], priors[kind] = ensure_prior(
                    problem,
                    PriorSettings(**plan["config"]["priors"][kind]),
                    project_file(PROJECT, plan["candidate_binary_path"]),
                    directory / kind,
                )
            graphs = match_graphs(
                problem, tuple(common["tour"]), values, GraphSettings(**plan["config"]["graph"])
            )
            mode = "fresh_paired_preparation"
        error = check_common(problem, common)
        for kind in KINDS:
            engine_graph_spec(graphs[kind])
        payload = {
            "instance_id": problem.instance_id,
            "dimension": problem.dimension,
            "coordinate_sha256": coordinate_hash(problem),
            "coordinates": problem.coordinates,
            "distance_spec": problem.distance_spec,
            "common": common,
            "graphs": graphs,
            "prior_receipts": priors,
            "original_files": original_files,
            "plan_sha256": plan["sha256"],
            "job": job,
            "mode": mode,
        }
        graph_path = directory / "graphs.json"
        if graph_path.exists():
            require(
                checked_json(graph_path) == json.loads(json.dumps(payload)),
                "已有未提交图与恢复重建不同；不能覆写",
            )
        else:
            atomic_json(graph_path, payload)
        after = resource.getrusage(resource.RUSAGE_SELF)
        files = {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in directory.rglob("*")
            if p.is_file() and p.name != ".lock" and not p.name.endswith(".partial")
        }
        for prior in priors.values():
            files.update(prior["files"])
        files.update(original_files)
        receipt = {
            "status": "complete",
            "job": job,
            "plan_sha256": plan["sha256"],
            "mode": mode,
            "files": dict(sorted(files.items())),
            "graph_path": str(graph_path.relative_to(PROJECT)),
            "coordinate_sha256": coordinate_hash(problem),
            "graphs": {
                kind: {"sha256": g["sha256"], "edges": len(g["edges"])}
                for kind, g in graphs.items()
            },
            "common_cost_absolute_error": error,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "latest_attempt_resources": {
                "wall_seconds": time.perf_counter() - started,
                "cpu_user_seconds": after.ru_utime - before.ru_utime,
                "cpu_system_seconds": after.ru_stime - before.ru_stime,
                "worker_lifetime_max_rss_kib": after.ru_maxrss,
                "note": "this observer attempt only; interrupted attempts retained separately; "
                "original LKH resources in prior receipts; maxima are not additive",
            },
        }
        atomic_json(directory / "complete.json", {**receipt, "sha256": content_hash(receipt)})
        return {
            "index": job["index"],
            "mode": mode,
            "resumed": False,
            "receipt_sha256": file_hash(directory / "complete.json"),
        }
