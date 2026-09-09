#!/usr/bin/env python3
"""三个真实首代种群在共享A5000池中的分片吞吐；只使用独立开发数据。"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import SEEDS, source
from gp_faco.campaign_pool import PopulationScheduler
from gp_faco.checkpoint import atomic_json
from gp_faco.evolution import Evolution, EvolutionSettings
from gp_faco.program_ir import export_tree
from gp_faco.data import tour_cost
from gp_faco.remote import discover_a5000
from benchmark_tsp500 import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--hosts", nargs="*", default=[])
    parser.add_argument("--gpus", type=int, default=0, help="0使用发现时所有空闲卡；测量开始后固定这批卡")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64, 128])
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT) or directory.exists():
        parser.error("使用项目内一个新的工程测量目录")
    directory.mkdir(parents=True)
    atomic_json(directory / "campaign.json", {"scope": "engineering_campaign_only", "infrastructure_retries": 1,
        "build": "build/v2-tsp500-exact", "gpu_discovery_seconds": 60, "native_initial_routes": 24, "cpu_host": "cuda07"})
    atomic_json(directory / "panels.json", {})
    scheduler = PopulationScheduler(directory, hosts=args.hosts, gpu_limit=args.gpus or None, poll_seconds=.25)
    programs = {}
    for seed in SEEDS:
        evolution = Evolution(EvolutionSettings(feature_spec_id=2), seed)
        evolution.initialize()
        programs[seed] = [export_tree(p).to_dict() for p in evolution.population]
    with source() as dataset:
        names = dataset.record_ids("development", 500)[64:80]
        problems = {name: dataset.load_instance(name) for name in names}
    pairs = [(name, 17) for name in names]
    expected = {}
    report = {"status": "running", "scope": "development_performance_only", "rows": [],
              "population": 128, "seeds": list(SEEDS), "instances": 16, "aco_seeds": [17], "iterations": args.iterations}
    try:
        devices = discover_a5000(args.hosts)
        if not devices or (args.gpus and len(devices) < args.gpus):
            raise RuntimeError(f"本次固定卡数比较需要{args.gpus}张空闲A5000，实际{len(devices)}张")
        devices = devices[:args.gpus] if args.gpus else devices
        for device in devices:
            scheduler.launch_worker(device)
        while not all(scheduler.worker_available(d["uuid"]) for d in devices):
            time.sleep(.25)
        scheduler.next_discovery = float("inf")  # 各分片方案使用同一批卡，不能中途新增GPU影响公平比较。
        report["devices"] = devices
        # 每种分片先热身。后续轮次轮换测量顺序，降低随时间漂移的影响。
        chunks = args.chunks
        for measure in range(args.repeats + 1):
            order = chunks[measure % len(chunks):] + chunks[:measure % len(chunks)]
            for chunk in order:
                started = time.perf_counter()
                jobs = []
                for seed in SEEDS:
                    for start in range(0, 128, chunk):
                        jobs.append(scheduler.add("population", f"m{measure}-c{chunk}-s{seed}-i{start}",
                            seed=seed, generation=measure, shard_index=start // chunk, dimension=500,
                            pairs=pairs, iterations=args.iterations, experiment_mask=0xffffffff,
                            programs=programs[seed][start:start + chunk], protocol_id="development-pool-benchmark",
                            occurrences=[f"m{measure}:seed{seed}:individual{i}" for i in range(start, start + chunk)]))
                while not scheduler.finished(jobs):
                    scheduler.reconcile()
                    if any(scheduler.states[j] == "failed" for j in jobs):
                        raise RuntimeError("分片测量有失败，不能据此选取分片大小")
                    scheduler.dispatch()
                    time.sleep(.25)
                all_results = list(scheduler.results(jobs))
                for seed in SEEDS:
                    native = [member["native_result"] for job, result in zip(jobs, all_results, strict=True)
                              if scheduler.jobs[job]["seed"] == seed for member in result["members"]]
                    actual = [signature(n) for n in native]
                    if seed in expected and actual != expected[seed]:
                        raise AssertionError("共享多卡的分片改变了路线、状态或计算工作量")
                    expected[seed] = actual
                    for outcome in native:
                        for (name, _), item in zip(pairs, outcome["items"], strict=True):
                            if abs(tour_cost(problems[name], item["tour"]) - item["cost"]) > 1e-9:
                                raise AssertionError("多卡返回的路线不能复现所报成本")
                row = {"chunk": chunk, "measure": measure, "warmup": measure == 0 and args.repeats > 0,
                    "wall_seconds": time.perf_counter() - started,
                    "solve_gpu_seconds": sum(r["solve_seconds"] for r in all_results),
                    "total_fe": sum(r["total_fe"] for r in all_results),
                    "gpu_uuids": sorted({r["device"]["gpu_uuid"] for r in all_results})}
                report["rows"].append(row)
                atomic_json(directory / "benchmark.json", report)
                print(json.dumps(row), flush=True)
        medians = {chunk: statistics.median(r["wall_seconds"] for r in report["rows"]
                   if r["chunk"] == chunk and not r["warmup"]) for chunk in chunks}
        minimum = min(medians.values())
        report.update(status="passed", median_wall_seconds=medians,
                      selected_chunk=max(chunk for chunk, value in medians.items() if value <= minimum * 1.03),
                      selection_rule="相同FE的共享池完成时间；距最优3%以内选较大分片", trajectories_equal=True)
        atomic_json(directory / "benchmark.json", report)
    finally:
        atomic_json(directory / "stop_workers.json", {"reason": "benchmark_finished"})
        scheduler.lease.close()


if __name__ == "__main__":
    main()
