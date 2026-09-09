#!/usr/bin/env python3
"""真实 A5000 上验证种群训练、批次暂停恢复、独立验证与 FE 账目；仅为工程检查。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.evolution import EvolutionSettings  # noqa: E402
from gp_faco.result_journal import ResultJournal  # noqa: E402
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings  # noqa: E402
from gp_faco.worker import WorkerProtocol, faco_ants  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--instances-per-panel", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT):
        parser.error("工程产物必须位于项目内")
    if (
        not 2 <= args.population <= 128
        or not 1 <= args.instances_per_panel <= 16
        or args.generations < 1
    ):
        parser.error("工程配置需要 2..128 个体、1..16 实例和正代数")
    model, driver = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.gpu}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    protocol = WorkerProtocol(
        args.gpu,
        model,
        driver,
        "v2-population-exact",
        colonies=args.instances_per_panel * 2,
        extension_directory="build/v2-population-exact",
    )
    settings = TrainingSettings(
        evolution=EvolutionSettings(
            population=args.population,
            generations=args.generations,
            elites=min(2, args.population - 1),
            feature_spec_id=2,
        ),
        population_batch_size=args.population,
        instances_per_panel=args.instances_per_panel,
        budgets=tuple((n, faco_ants(n) * 10) for n in (500, 1000)),
        validation_budgets=tuple((n, faco_ants(n) * 20) for n in (500, 1000)),
        preparation_mode="cached",
        budget_kind="search_tour_evaluations",
        scope="engineering_population_cuda_only",
    )
    with IndexedDataset(
        PROJECT / "artifacts/data/main-index-v1/instances.sqlite", PROJECT.parent / "Datasets/TSP"
    ) as source:
        width = args.instances_per_panel * 2
        pools = {n: source.record_ids("development", n)[64 : 64 + width * 2] for n in (500, 1000)}
        data = TrainingData(
            source,
            {n: v[:width] for n, v in pools.items()},
            {n: v[width:] for n, v in pools.items()},
            {"scope": "engineering_population_cuda_only", "pools": pools},
        )
        paused = TrainingRun(directory, settings, protocol, data).run(stop_after_tasks=1)
        if paused["status"] != "paused" or paused["solve_jobs"] != args.population:
            raise AssertionError("完整种群面板没有在边界保存")
        resumed = TrainingRun(directory, settings, protocol, data, resume=True).run()
    if resumed["status"] != "complete" or resumed["costs"]["failed_solves"]:
        raise AssertionError("真实种群训练/验证未完整完成")
    journal = ResultJournal(directory / "results.jsonl")
    records = []
    for _offset, record in journal.trailing(0):
        if record["kind"] == "population":
            records.extend(
                TrainingRun._population_member(record, i) for i in range(len(record["checked"]))
            )
        elif record["kind"] == "solve":
            records.append(record)
    journal.close()
    occurrences = [r["key"] for r in records]
    if len(set(occurrences)) != len(occurrences):
        raise AssertionError("恢复重复求解了已经完成的个体")
    training = [r for r in records if ":generation" in r["key"]]
    if len(training) != args.population * args.generations * 2:
        raise AssertionError("没有完成每代的所有个体和规模")
    total_fe = sum(r["outcome"]["native_result"]["total_tour_evaluations"] for r in records)
    if total_fe != resumed["costs"]["search_tour_evaluations"]:
        raise AssertionError("FE 总账与完整成员不符")
    report = {
        "status": "passed",
        "scope": settings.scope,
        "population": args.population,
        "instances_per_panel": args.instances_per_panel,
        "generations": args.generations,
        "training_members": len(training),
        "validation_candidates": len(resumed["validation_results"]),
        "all_occurrences_unique": True,
        "costs": resumed["costs"],
        "protocol": protocol.manifest(),
    }
    atomic_json(directory / "check_result.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
