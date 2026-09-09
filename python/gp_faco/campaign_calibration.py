"""把预登记的开发曲线按控制器和实例/seed 面板分发；已有完整轨迹直接接续。"""

from __future__ import annotations

import json
import time

from gp_faco.campaign import read, source
from gp_faco.checkpoint import atomic_json
from gp_faco.data import tour_cost
from gp_faco.experiment_v2 import batches, choose_horizon, write_once
from gp_faco.gpu_session import GpuSession
from gp_faco.worker import faco_ants


def calibration_jobs(manifest):
    for panel in manifest["panels"]:
        for controller, policy in manifest["controllers"].items():
            for index, pairs in enumerate(batches(panel)):
                name = f"n{panel['dimension']}-{controller}-batch{index + 1:02d}"
                yield {
                    "id": f"calibration-{name}",
                    "source_job_id": name,
                    "kind": "calibration",
                    "resource": "gpu",
                    "dimension": panel["dimension"],
                    "pairs": pairs,
                    "iterations": manifest["iterations"],
                    "checkpoints": manifest["checkpoints"],
                    "controller": controller,
                    "policy": policy,
                    # 标定沿用正在计算的原构建，跨卡调度不切换求解实现。
                    "build": "build/v2-exact",
                }


def import_calibration(directory, origin):
    """原串行进程停止后调用；只读完整日志行，不复制已存的大型 tour。"""
    manifest = read(origin / "manifest.json")
    if manifest["stage"] != "curves" or manifest["backend"] != "exact":
        raise ValueError("本轮只接续预登记的 exact 开发曲线")
    target = directory / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    write_once(target / "manifest.json", manifest)
    jobs = {j["source_job_id"]: j for j in calibration_jobs(manifest)}
    imported, partial = [], 0
    with (origin / "results.jsonl").open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                partial = len(line)
                break
            old = json.loads(line)
            name = old["job_id"]
            if name in imported or name not in jobs:
                raise ValueError("原始标定日志有重复或未登记任务")
            if any(r["status"] != "completed" for r in old["rows"]):
                raise ValueError("已有失败的开发曲线成员，不能忽略后续算")
            job = jobs[name]
            path = directory / "jobs" / job["id"]
            path.mkdir(parents=True, exist_ok=True)
            write_once(path / "job.json", job)
            write_once(
                path / "result.json",
                {
                    "status": "completed",
                    "kind": "calibration",
                    "job": job["id"],
                    "rows": old["rows"],
                    "total_fe": old["native_result"]["total_tour_evaluations"],
                    "solve_seconds": old["timing"]["solve_seconds"],
                    "wall_seconds": old["end_to_end_seconds"],
                    "device": old["device"],
                    "inherited": True,
                    "raw_record": {"path": str(origin / "results.jsonl"), "offset": offset},
                },
            )
            imported.append(name)
    receipt = {
        "status": "imported",
        "completed_jobs": len(imported),
        "total_jobs": len(jobs),
        "incomplete_trailing_bytes": partial,
        "source_directory": str(origin),
        "recomputed_completed_jobs": 0,
    }
    atomic_json(target / "import.json", receipt)
    return receipt


def solve_calibration(directory, job, gpu):
    output = directory / "jobs" / job["id"]
    raw_path = output / "trajectory.json"
    raw = read(raw_path)
    n, pairs = job["dimension"], job["pairs"]
    with source() as dataset:
        problems = {name: dataset.load_instance(name) for name, _ in pairs}
        references = {name: dataset.load_label(name).cost for name in problems}
        if raw is None:
            session = GpuSession(
                gpu, "exact", colonies=len(pairs), extension_directory=job["build"]
            )
            try:
                native, timing = session.solve(
                    problems,
                    pairs,
                    job["iterations"],
                    job["policy"],
                    checkpoints=job["checkpoints"],
                )
                raw = {"native": native, "timing": timing, "device": session.device}
                atomic_json(raw_path, raw)
            finally:
                session.close()
        native = raw["native"]
        fe = len(pairs) * faco_ants(n) * job["iterations"]
        if native["total_tour_evaluations"] != fe:
            raise ValueError("开发曲线没有完成预登记的全部 FE")
        if [c["iterations"] for c in native["checkpoints"]] != job["checkpoints"]:
            raise ValueError("开发曲线记录点与预登记不一致")
        before = time.perf_counter()
        rows = []
        for checkpoint in native["checkpoints"]:
            for (name, seed), tour in zip(pairs, checkpoint["tours"], strict=True):
                cost = tour_cost(problems[name], tour)
                rows.append(
                    {
                        "dimension": n,
                        "controller": job["controller"],
                        "instance_id": name,
                        "seed": seed,
                        "iterations": checkpoint["iterations"],
                        "status": "completed",
                        "cost": cost,
                        "gap_percent": 100 * (cost / references[name] - 1),
                    },
                )
    return {
        "status": "completed",
        "kind": "calibration",
        "rows": rows,
        "total_fe": fe,
        "solve_seconds": raw["timing"]["solve_seconds"],
        "preparation_seconds": raw["timing"]["registration_seconds"],
        "scoring_seconds": time.perf_counter() - before,
        "device": raw["device"],
    }


def summarize_calibration(directory, results):
    manifest = read(directory / "calibration/manifest.json")
    results = list(results)
    rows = [r for result in results for r in result["rows"]]
    if any(r["status"] != "completed" for r in rows):
        raise ValueError("开发曲线有失败，停止依赖训练")
    selection = choose_horizon(
        rows,
        controllers=list(manifest["controllers"]),
        expected_panels=manifest["panels"],
        checkpoints=manifest["checkpoints"],
    )
    result = {
        "status": "complete",
        "backend": "exact",
        "stage": "curves",
        "completed_jobs": len(results),
        "rows": rows,
        "selection": selection,
        "total_seconds": sum(r.get("wall_seconds", r["solve_seconds"]) for r in results),
        "total_fe": sum(r["total_fe"] for r in results),
    }
    atomic_json(directory / "calibration/summary.json", result)
    return result
