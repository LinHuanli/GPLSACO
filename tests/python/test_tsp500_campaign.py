"""单规模预算、共享GPU事务、两级选树及迁移统计的针对性验证。"""

from dataclasses import replace
from concurrent.futures import TimeoutError
import time

import pytest

from gp_faco.campaign import read, comparison_statistics, GPU_BASELINE, NATIVE_BASELINE, SEEDS
from gp_faco.campaign_pool import PopulationScheduler
from gp_faco.campaign_tsp500 import initialize_tsp500, validation_ranking, select_light_validation
from gp_faco.checkpoint import atomic_json, load_checkpoint
from gp_faco.distributed_population import PopulationClient, PopulationFuture
from gp_faco.worker import WorkerProtocol
from test_worker_protocol import task
from test_campaign import selection_fixture


def test_formal_tsp500_panels_and_validation_budget(tmp_path):
    config = initialize_tsp500(tmp_path)
    panels, frozen = [read(tmp_path / name) for name in ("panels.json", "frozen.json")]
    assert (config["population"], config["generations"], config["training_solver_seeds"]) == (128, 50, 1)
    assert len(panels["training"]) == 50
    for generation in panels["training"]:
        assert len(generation["panels"]) == 1
        panel = generation["panels"][0]
        assert (panel["dimension"], len(panel["ids"]), len(panel["seeds"])) == (500, 16, 1)
    quick, final = panels["validation"][0], panels["validation_final"][0]
    assert (len(quick["ids"]), quick["seeds"]) == (32, [17])
    assert (len(final["ids"]), final["seeds"]) == (64, [17, 29, 41])
    assert set(quick["ids"]).isdisjoint(final["ids"])
    assert {p["dimension"] for p in panels["test"]} == {500, 1000}
    assert all(len(p["ids"]) == 128 and len(p["seeds"]) == 10 for p in panels["test"])
    assert frozen["training_iterations"] == 1000 and frozen["wall_clock_limit"] is None
    training = 128 * 50 * 16 * 1000
    validation = 177 * 32 * 1000 + 4 * 64 * 3 * 5000
    assert validation / training == pytest.approx(0.0928125)


def test_population_submits_all_shards_and_restores_original_order(tmp_path):
    atomic_json(tmp_path / "campaign.json", {"population_shard_size": 2})
    protocol = WorkerProtocol("pool", "NVIDIA RTX A5000", "per_worker", "test", dimensions=(3,), colonies=4, distributed=True)
    # 六个完全相同的程序仍有六次独立评价，编号来自个体出现位置。
    program = replace(task().program, feature_spec_id=2)
    tasks = [replace(task(f"full:seed1103:generation0:individual{i}:panel0"), program=program, budget_seconds=None,
                     preparation_mode="cached", evaluation_limit_per_colony=1280) for i in range(6)]
    client = PopulationClient(tmp_path, 1103, protocol)
    future = client.submit_population(tasks)
    assert len(list((tmp_path / "requests").glob("*.json"))) == 3
    with pytest.raises(TimeoutError):
        future.result(timeout=0)
    for key in reversed(future.jobs):
        job = read(tmp_path / "requests" / f"{key}.json")
        atomic_json(tmp_path / "jobs" / key / "result.json", {"status": "completed", "members": [
            {"occurrence_id": occurrence, "native_result": {"items": [{"tour": [0, 1, 2]}]}}
            for occurrence in job["occurrences"]]})
    result = future.result(timeout=0)
    assert [m["occurrence_id"] for m in result["members"]] == [t.occurrence_id for t in tasks]
    restarted = PopulationClient(tmp_path, 1103, protocol).submit_population(tasks)
    assert restarted.result(timeout=0) == result
    assert len(list((tmp_path / "requests").glob("*.json"))) == 3


@pytest.mark.parametrize("campaign_abort", [False, True])
def test_population_dependency_failure_is_not_an_infinite_wait(tmp_path, campaign_abort):
    path = tmp_path / ("campaign_abort.json" if campaign_abort else "jobs/shard/terminal_failure.json")
    atomic_json(path, {"error": "injected terminal failure"})
    with pytest.raises(RuntimeError, match="injected terminal failure"):
        PopulationFuture(tmp_path, ["shard"]).result(timeout=0)


def test_completed_population_reuses_worker_without_cluster_rescan(tmp_path):
    atomic_json(tmp_path / "campaign.json", {"scope": "engineering_campaign_only", "infrastructure_retries": 1})
    atomic_json(tmp_path / "panels.json", {})
    scheduler = PopulationScheduler(tmp_path)
    try:
        key = scheduler.add("population", "done")
        scheduler.states[key] = "running"
        scheduler.active[key] = {"gpu_uuid": "gpu", "host": "cuda10", "attempt": 1}
        deadline = scheduler.next_discovery = time.monotonic() + 60
        atomic_json(tmp_path / "jobs" / key / "result.json", {"status": "completed"})
        assert scheduler.reconcile()
        assert scheduler.states[key] == "completed"
        assert scheduler.next_discovery == deadline
    finally:
        scheduler.lease.close()


def test_shared_pool_dispatch_is_fair_between_seeds_and_binds_mailbox(tmp_path, monkeypatch):
    atomic_json(tmp_path / "campaign.json", {"scope": "engineering_campaign_only", "infrastructure_retries": 1,
        "cpu_host": "cuda07", "gpu_discovery_seconds": 60})
    atomic_json(tmp_path / "panels.json", {})
    scheduler = PopulationScheduler(tmp_path)
    launches = []
    monkeypatch.setattr("gp_faco.campaign_scheduler.launch_remote", lambda *args: launches.append(args))
    monkeypatch.setattr("gp_faco.campaign_pool.discover_a5000", lambda _: [])
    monkeypatch.setattr("gp_faco.campaign_pool.process_alive", lambda _: True)
    for seed in SEEDS:
        key = scheduler.add("training", str(seed), seed=seed)
        assert read(tmp_path / "jobs" / key / "job.json")["resource"] == "coordinator"
        for shard in range(2):
            scheduler.add("population", f"{seed}-{shard}", seed=seed, generation=0, shard_index=shard)
    for index in range(3):
        uuid = f"gpu-{index}"
        scheduler.workers[uuid] = {"host": "cuda10", "uuid": uuid}
        atomic_json(tmp_path / "workers" / uuid / "runtime.json", {"host": "cuda10", "pid": index + 1,
            "start_ticks": 100 + index, "status": "idle"})
    try:
        scheduler.dispatch()
        assert len(launches) == 3
        assigned = [scheduler.jobs[k] for k in scheduler.active if scheduler.jobs[k]["kind"] == "population"]
        assert {r["seed"] for r in assigned} == set(SEEDS)
        assert {r["shard_index"] for r in assigned} == {0}
        for index in range(3):
            mail = read(tmp_path / "workers" / f"gpu-{index}" / "request.json")
            assert (mail["worker_pid"], mail["worker_start_ticks"]) == (index + 1, 100 + index)
    finally:
        scheduler.lease.close()


def test_two_stage_selection_uses_only_finalist_rows_and_keeps_quick_curve(tmp_path):
    programs, _, rows = selection_fixture(tmp_path)
    quick_panel = {"dimension": 500, "ids": ["n500-a", "n500-b"], "seeds": [17, 29]}
    final_panel = {**quick_panel, "ids": ["n500-c", "n500-d"]}
    atomic_json(tmp_path / "panels.json", {"validation": [quick_panel], "validation_final": [final_panel]})
    atomic_json(tmp_path / "campaign.json", {"scope": "unit", "population": 4, "generations": 2, "instances_per_panel": 2})
    atomic_json(tmp_path / "frozen.json", {"training_iterations": 1000})
    quick_rows = [r for r in rows if r["dimension"] == 500]
    quick_result = {"rows": quick_rows, "total_fe": 100, "solve_seconds": 1}
    scores, order = validation_ranking(tmp_path, 1103, [quick_result], programs, "validation")
    finalists = [next(p for p in programs if p["program_id"] == key) for key in order[:2]]
    atomic_json(tmp_path / "training/full-seed1103/validation_screen.json", {"evaluations": scores, "finalists": finalists})
    final_rows = [{**r, "instance_id": "n500-c" if r["instance_id"] == "n500-a" else "n500-d",
                   "gap_percent": 0.0 if r["method"] == order[1] else 2.0}
                  for r in quick_rows if r["method"] in order[:2]]
    final = {"rows": final_rows, "total_fe": 200, "solve_seconds": 2}
    report = select_light_validation(tmp_path, 1103, [quick_result], [final], programs)
    assert report["selected"]["program_id"] == order[1]
    assert len(report["champion_validation_curve"]) == 2
    assert len(report["validation_results"]) == 2
    assert report["validation_costs"]["search_tour_evaluations"] == 300
    assert load_checkpoint(tmp_path / "training/full-seed1103/checkpoint.json")["phase"] == "complete"


def test_tsp500_is_primary_and_tsp1000_is_descriptive_transfer():
    panels = [{"dimension": n, "ids": [f"{n}-a", f"{n}-b"], "seeds": [17, 29]} for n in (500, 1000)]
    gaps = {NATIVE_BASELINE: 2., GPU_BASELINE: 1.5, **{f"gp-{s}": .1 for s in SEEDS}}
    rows = [{"method": method, "dimension": p["dimension"], "instance_id": name, "seed": seed,
             "cost": 100 + gap, "gap_percent": gap, "status": "completed"}
            for method, gap in gaps.items() for p in panels for name in p["ids"] for seed in p["seeds"]]
    result = comparison_statistics(rows, {"seeds": SEEDS, "statistics_seed": 91, "bootstrap_replicates": 100,
        "primary_dimension": 500, "report_macro": False}, panels)
    assert {r["dimension"] for r in result["rows"]} == {500, 1000}
    assert all("holm_p_value" in r for r in result["rows"] if r["dimension"] == 500)
    assert all("holm_p_value" not in r for r in result["rows"] if r["dimension"] == 1000)
