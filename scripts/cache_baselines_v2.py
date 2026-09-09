#!/usr/bin/env python3
"""训练前预计算完整共享面板、开发监控和验证上的两类 FACO；不读取测试标签。"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.experiment_v2 import (  # noqa: E402
    GPU_BASELINE,
    NATIVE_BASELINE,
    BaselineCache,
    batches,
)
from gp_faco.gpu_session import GpuSession  # noqa: E402
from gp_faco.native_baseline import solve_native  # noqa: E402


def cache_manifest(frozen):
    return {
        "experiment_version": 2,
        "dataset": "main-index-v1",
        "distance": "continuous_euclidean_fp64",
        "numeric_backend": frozen["numeric_backend"],
        "ants": "64*ceil(sqrt(n)/16)",
        "beta": 1,
        "retention": 0.5,
        "primary_width": 16,
        "backup_width": 64,
        "ls_width": 20,
        "p_best": 0.1,
        "source_probability": 0.01,
        "mne": 8,
        "gpu_ls_evaluation_limit": 100000,
        "gpu_initialization": "node0_nn_checklist_2opt_100000_identity_fallback",
        "native_initialization": "author_nn_3opt_24_initial_routes_8_threads_edgeguard2",
        "gpu_local_search": "checklist_2opt_identity_zero_v2",
        "native_continuous_adaptation_revision": 2,
    }


def required_panels(declaration, training_iterations):
    return [
        (f"generation-{row['generation'] + 1:03d}", p, training_iterations)
        for row in declaration["training"]
        for p in row["panels"]
    ] + [(role, p, 5000) for role in ("monitor", "validation") for p in declaration[role]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT / "artifacts/v2/protocol")
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("运行输出必须在项目内")
    frozen = json.loads((directory / "frozen.json").read_text())
    declaration = json.loads((directory / "panels.json").read_text())
    if os.cpu_count() != 24:
        raise ValueError("此原始 FACO 对照固定 24 条原生初始路线；请在已登记的 cuda10 主机运行")
    cache = BaselineCache(directory / "baselines.sqlite", cache_manifest(frozen))
    source = IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    )
    session = None
    completed = 0
    try:
        for method in (GPU_BASELINE, NATIVE_BASELINE):
            completed = 0
            for role, panel, iterations in required_panels(
                declaration, frozen["training_iterations"]
            ):
                n = panel["dimension"]
                problems = {name: source.load_instance(name) for name in panel["ids"]}
                references = {name: source.load_label(name).cost for name in panel["ids"]}
                for pairs in batches(panel):
                    if method == GPU_BASELINE:
                        missing = [
                            (name, seed)
                            for name, seed in pairs
                            if cache.get(GPU_BASELINE, n, name, seed, iterations) is None
                        ]
                        if missing:
                            if len(missing) != len(pairs):
                                raise RuntimeError(
                                    "缓存事务不完整；请保留记录后排查，不重跑已完成 baseline"
                                )
                            if session is None:
                                session = GpuSession(args.gpu, frozen["numeric_backend"])
                            start = time.perf_counter()
                            result, _ = session.solve(problems, pairs, iterations, None)
                            rows = [
                                (name, seed, item["tour"], tour_cost(problems[name], item["tour"]))
                                for (name, seed), item in zip(pairs, result["items"], strict=True)
                            ]
                            seconds = (time.perf_counter() - start) / len(pairs)
                            for name, seed, tour, cost in rows:
                                cache.put(
                                    GPU_BASELINE,
                                    n,
                                    name,
                                    seed,
                                    iterations,
                                    cost=cost,
                                    gap_percent=100 * (cost / references[name] - 1),
                                    seconds=seconds,
                                    tour=tour,
                                )
                            cache.commit()  # 一个完整 GPU 返回为一个数据库事务。
                    else:
                        for name, seed in pairs:
                            if cache.get(NATIVE_BASELINE, n, name, seed, iterations) is not None:
                                continue
                            problem = problems[name]
                            run_dir = (
                                directory
                                / "native-baselines"
                                / f"instance-{problem.numeric_id}"
                                / f"seed-{seed}-it-{iterations}"
                            )
                            result = solve_native(problem, seed, iterations, run_dir)
                            cost = tour_cost(problem, result["tour"])
                            if result["initial_route_count"] != 24:
                                raise ValueError("原始 FACO 的实际初始化条数与声明不同")
                            cache.put(
                                NATIVE_BASELINE,
                                n,
                                name,
                                seed,
                                iterations,
                                cost=cost,
                                gap_percent=100 * (cost / references[name] - 1),
                                seconds=result["total_seconds"],
                                tour=result["tour"],
                            )
                            cache.commit()
                    completed += len(pairs)
                    atomic_json(
                        directory / "baseline_progress.json",
                        {
                            "status": "running",
                            "method": method,
                            "members": completed,
                            "role": role,
                            "dimension": n,
                        },
                    )
                    print(
                        json.dumps(
                            {"method": method, "role": role, "dimension": n, "members": completed}
                        ),
                        flush=True,
                    )
            if session is not None:
                session.close()
                session = None
        # 完成清单由逐成员的实际缓存产生；训练入口读取该表，不猜测缺项。
        for _, panel, iterations in required_panels(declaration, frozen["training_iterations"]):
            for method in (GPU_BASELINE, NATIVE_BASELINE):
                cache.panel(method, panel, iterations)
        atomic_json(
            directory / "baselines_ready.json",
            {
                "status": "complete",
                "manifest": cache.manifest,
                "training_iterations": frozen["training_iterations"],
                "paired_members": completed,
                "test_scores_read": False,
            },
        )
        atomic_json(
            directory / "baseline_progress.json",
            {"status": "complete", "paired_members": completed},
        )
    finally:
        if session:
            session.close()
        source.close()
        cache.close()


if __name__ == "__main__":
    main()
