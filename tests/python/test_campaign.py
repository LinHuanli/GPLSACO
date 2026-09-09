"""队列恢复、测试隔离、分片完整性和选树的一致性。"""

import os
import platform
import subprocess
import sys
import time
from dataclasses import replace

import pytest
from gp_faco.campaign import (
    GPU_BASELINE,
    NATIVE_BASELINE,
    SEEDS,
    comparison_statistics,
    freeze_methods,
    select_validation,
    validate_rows,
)
from gp_faco.campaign_jobs import solve_job
from gp_faco.campaign_scheduler import CampaignScheduler
from gp_faco.checkpoint import atomic_json, load_checkpoint, save_checkpoint
from gp_faco.evolution import Evolution, EvolutionSettings
from gp_faco.remote import parse_idle_a5000
from gp_faco.training import TrainingRun
from test_population_runtime import PopulationFarm
from test_v2_runtime import setup


def test_discovery_accepts_only_idle_a5000():
    text = """
IDLE cuda03 1 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
SINGLE cuda10 1 0.3 / 24.0 0% 1 another NVIDIA RTX A5000
IDLE cuda22 0 0.0 / 48.0 0% 0 - NVIDIA RTX 6000 Ada Generation
IDLE piccolo 0 0.0 / 24.0 0% 0 - NVIDIA RTX A5000
"""
    assert parse_idle_a5000(text) == [
        {"host": "cuda03", "index": 1},
        {"host": "piccolo", "index": 0},
    ]


@pytest.mark.parametrize("busy", [True, False])
def test_worker_startup_preserves_resource_and_configuration_errors(tmp_path, busy):
    # 在真实 spawn 边界模拟启动时已被占用；不创建 CUDA 上下文或使用真实 GPU。
    uuid = "GPU-11111111-1111-1111-1111-111111111111"
    command = tmp_path / "nvidia-smi"
    command.write_text(f"#!/bin/sh\nprintf '%s\\n' '{uuid}, 99999999'\n")
    command.chmod(0o755)
    code = (
        "from gp_faco.worker import PersistentGpuWorker, WorkerProtocol\n"
        "import platform\n"
        f"p=WorkerProtocol({uuid!r}, 'NVIDIA RTX A5000', 'test', 'test', "
        f"execution_host=platform.node() if {busy!r} else 'invalid.example')\n"
        "with PersistentGpuWorker(p) as w:\n"
        " try: w.ready(timeout=10)\n"
        " except Exception as e:\n"
        f"  assert type(e) is {'BlockingIOError' if busy else 'RuntimeError'}, repr(e)\n"
        f"  assert {'已有计算进程' if busy else '执行host与任务协议不符'!r} in str(e)\n"
        " else: raise AssertionError('Expected startup rejection')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_boundary_and_portable_resume_preserve_occurrences(tmp_path):
    settings, protocol, data = setup()
    settings = replace(settings, population_batch_size=6, portable_a5000=True)
    farm = PopulationFarm()
    path = tmp_path / "portable"
    TrainingRun(path, settings, protocol, data, worker_factory=farm).run(stop_after_tasks=1)
    moved = replace(
        protocol, gpu_uuid="GPU-22222222-2222-2222-2222-222222222222", driver_version="new"
    )
    done = TrainingRun(path, settings, moved, data, resume=True, worker_factory=farm).run(
        through="training"
    )
    assert done["status"] == "validation"
    assert len(farm.calls) == settings.evolution.population * settings.evolution.generations * 2
    assert len(farm.calls) == len(set(farm.calls))
    assert all(":generation" in key for key in farm.calls)
    assert not (path / "selected_program.json").exists()
    changed = replace(settings, budgets=tuple((n, value * 2) for n, value in settings.budgets))
    with pytest.raises(ValueError, match="参数发生变化"):
        TrainingRun(path, changed, moved, data, resume=True, worker_factory=farm)
    result = TrainingRun(path, settings, moved, data, resume=True, worker_factory=farm).run()
    assert result["status"] == "complete"
    assert len(farm.calls) == len(set(farm.calls))


@pytest.mark.parametrize("kind", ["test_gpu", "test_native", "explain"])
def test_test_labels_are_unavailable_before_all_selections(tmp_path, monkeypatch, kind):
    atomic_json(tmp_path / "campaign.json", {"scope": "test"})
    monkeypatch.setattr(
        "gp_faco.campaign_jobs.source",
        lambda: pytest.fail("Test labels were accessed before method freeze"),
    )
    with pytest.raises(ValueError, match="冻结"):
        solve_job(tmp_path, {"kind": kind})


def selection_fixture(directory):
    evolution = Evolution(
        EvolutionSettings(population=4, generations=2, feature_spec_id=2, elites=1), 1103
    )
    evolution.initialize()
    for generation in range(2):
        evolution.begin_panel(f"panel-{generation}")
        for i in range(4):
            evolution.assign(i, i / 10, evolution.panel_id, evolution.population[i].program_id)
        evolution.finish_generation()
        if generation == 0:
            evolution.advance()
    programs = [p.to_dict() for p in evolution.shortlist()]
    run = directory / "training/full-seed1103"
    run.mkdir(parents=True)
    save_checkpoint(
        run / "checkpoint.json",
        {
            "phase": "validation",
            "run_id": run.name,
            "evolution": evolution.state_dict(),
            "shortlist": programs,
            "manifest": {},
            "costs": {},
        },
    )
    save_checkpoint(
        run / "training_summary.json", {"scope": "unit", "costs": {}, "status": "validation"}
    )
    panels = [
        {"dimension": n, "ids": [f"n{n}-a", f"n{n}-b"], "seeds": [17, 29]} for n in (500, 1000)
    ]
    atomic_json(directory / "panels.json", {"validation": panels})
    rows = [
        {
            "method": p["program_id"],
            "dimension": panel["dimension"],
            "instance_id": name,
            "seed": seed,
            "status": "completed",
            "cost": 101.0,
            "gap_percent": 1.0,
        }
        for p in programs
        for panel in panels
        for name in panel["ids"]
        for seed in panel["seeds"]
    ]
    return programs, panels, rows


def test_distributed_selection_independent_of_arrival_order_and_idempotent(tmp_path):
    programs, _panels, rows = selection_fixture(tmp_path)
    chunks = [
        {"rows": rows[: len(rows) // 2], "total_fe": 100, "solve_seconds": 1},
        {"rows": rows[len(rows) // 2 :], "total_fe": 200, "solve_seconds": 2},
    ]
    first = select_validation(tmp_path, 1103, iter(chunks))
    expected = min(programs, key=lambda p: (len(p["opcode"]), p["program_id"]))
    assert first["selected"]["program_id"] == expected["program_id"]
    assert first["validation_costs"]["search_tour_evaluations"] == 300
    assert len(first["champion_validation_curve"]) == 2
    second = select_validation(tmp_path, 1103, reversed(chunks))
    assert first == second
    assert (
        load_checkpoint(tmp_path / "training/full-seed1103/checkpoint.json")["phase"] == "complete"
    )


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "failed"])
def test_validation_rejects_incomplete_or_failed_shards(tmp_path, alteration):
    programs, panels, rows = selection_fixture(tmp_path)
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["status"] = "failed"
    with pytest.raises(ValueError):
        validate_rows(rows, methods=[p["program_id"] for p in programs], panels=panels)


def test_method_freeze_requires_three_selected_programs(tmp_path):
    atomic_json(tmp_path / "campaign.json", {"seeds": list(SEEDS)})
    with pytest.raises(FileNotFoundError):
        freeze_methods(tmp_path)
    assert not (tmp_path / "methods_frozen.json").exists()


def scheduler_fixture(path):
    atomic_json(
        path / "campaign.json",
        {"scope": "engineering_campaign_only", "infrastructure_retries": 1},
    )
    atomic_json(path / "panels.json", {})
    return CampaignScheduler(path)


def test_completed_jobs_are_not_relaunched_after_coordinator_restart(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path)
    job = scheduler.add("training", "1103", seed=1103)
    path = tmp_path / "jobs" / job
    atomic_json(path / "assigned.json", {"host": "old", "attempt": 1})
    atomic_json(path / "result.json", {"status": "completed"})
    scheduler.lease.close()
    monkeypatch.setattr(
        "gp_faco.campaign_scheduler.process_alive", lambda _: pytest.fail("Finished job probed")
    )
    resumed = CampaignScheduler(tmp_path)
    assert resumed.states[job] == "completed"
    assert not resumed.active
    resumed.lease.close()


def test_lost_launch_response_preserves_remote_live_lease(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path)
    job = scheduler.add("training", "1103", seed=1103)
    scheduler.active[job] = {
        "host": "remote",
        "attempt": 1,
        "gpu_uuid": "gpu",
        "launched_unix": time.time() - 120,
    }
    scheduler.states[job] = "running"
    monkeypatch.setattr("gp_faco.campaign_scheduler.job_lock_active", lambda *_: True)
    assert not scheduler.reconcile()
    assert scheduler.states[job] == "running"
    scheduler.lease.close()


def test_resource_wait_does_not_consume_solver_failure_retry(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path)
    job = scheduler.add("training", "1103", seed=1103)
    scheduler.active[job] = {"host": "remote", "attempt": 1, "gpu_uuid": "gpu"}
    scheduler.states[job] = "running"
    atomic_json(
        tmp_path / "jobs" / job / "runtime.json",
        {"status": "resource_wait", "host": platform.node(), "pid": 99999999},
    )
    monkeypatch.setattr("gp_faco.campaign_scheduler.process_alive", lambda _: False)
    assert scheduler.reconcile()
    assert scheduler.states[job] == "pending"
    assert not (tmp_path / "jobs" / job / "failures.json").exists()
    scheduler.lease.close()


def test_only_one_coordinator_can_claim_campaign(tmp_path):
    scheduler = scheduler_fixture(tmp_path)
    with pytest.raises(BlockingIOError):
        CampaignScheduler(tmp_path)
    scheduler.lease.close()


def test_fixed_baselines_start_before_horizon_without_changing_job_identity(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path)
    scheduler.config.update(
        scope="full_three_seed_v2", instances_per_panel=1, validation_iterations=5000
    )

    def panel(name):
        return {"dimension": 500, "ids": [name], "seeds": [17, 29]}

    scheduler.declaration = {
        "training": [{"generation": 0, "panels": [panel("train")]}],
        "monitor": [panel("monitor")],
        "validation": [panel("validation")],
    }
    monkeypatch.setattr(scheduler, "inherited_baselines", lambda: {})
    monkeypatch.setattr(scheduler, "prepare_horizon", lambda: None)
    scheduler.advance()
    first = dict(scheduler.jobs)
    assert len(first) == 4
    assert {j["iterations"] for j in first.values()} == {5000}
    assert {p[0] for j in first.values() for p in j["pairs"]} == {"monitor", "validation"}
    assert not scheduler.group(("training", "test_gpu", "test_native"))
    monkeypatch.setattr(scheduler, "prepare_horizon", lambda: {"training_iterations": 100})
    scheduler.advance()
    assert len(scheduler.jobs) == 6
    assert all(scheduler.jobs[k] == v for k, v in first.items())
    assert {j["iterations"] for k, j in scheduler.jobs.items() if k not in first} == {100}
    scheduler.lease.close()


def test_three_evolution_seeds_dispatch_concurrently_across_hosts(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path)
    scheduler.config.update(
        scope="full_three_seed_v2", gpu_discovery_seconds=60, native_initial_routes=24
    )
    for seed in SEEDS:
        scheduler.add("training", str(seed), seed=seed)
    devices = [
        {"host": h, "uuid": f"gpu-{i}"} for i, h in enumerate(("cuda03", "cuda08", "piccolo"))
    ]
    requested = []
    monkeypatch.setattr(
        "gp_faco.campaign_scheduler.discover_a5000",
        lambda hosts: requested.append(tuple(hosts)) or devices,
    )
    launches = []
    monkeypatch.setattr(
        "gp_faco.campaign_scheduler.launch_remote", lambda *args: launches.append(args)
    )
    scheduler.dispatch()
    assert requested == [()]
    assert len(launches) == len(scheduler.active) == 3
    assert {a["host"] for a in scheduler.active.values()} == {"cuda03", "cuda08", "piccolo"}
    scheduler.dispatch()
    assert len(launches) == 3
    scheduler.lease.close()


def test_native_baselines_spread_only_to_hosts_with_same_initial_route_count(tmp_path, monkeypatch):
    scheduler = scheduler_fixture(tmp_path)
    scheduler.config.update(
        scope="full_three_seed_v2", gpu_discovery_seconds=60, native_initial_routes=24
    )
    for index in range(4):
        scheduler.add("baseline_native", str(index))
    devices = [
        {"host": "cuda08", "uuid": "gpu-1", "host_cpus": 24},
        {"host": "cuda10", "uuid": "gpu-2", "host_cpus": 24},
        {"host": "cuda03", "uuid": "gpu-3", "host_cpus": 16},
        {"host": "piccolo", "uuid": "gpu-4", "host_cpus": 20},
    ]
    monkeypatch.setattr("gp_faco.campaign_scheduler.discover_a5000", lambda _: devices)
    monkeypatch.setattr("gp_faco.campaign_scheduler.launch_remote", lambda *args: None)
    scheduler.dispatch()
    assert {a["host"] for a in scheduler.active.values()} == {"cuda07", "cuda08", "cuda10"}
    assert list(scheduler.states.values()).count("pending") == 1
    scheduler.lease.close()


def test_paired_statistics_include_all_three_evolution_seeds_and_both_scales():
    panels = [{"dimension": n, "ids": [f"{n}-a", f"{n}-b"], "seeds": [17, 29]} for n in (500, 1000)]
    gaps = {NATIVE_BASELINE: 2.0, GPU_BASELINE: 1.5, "gp-1103": 0, "gp-2207": 0.1, "gp-3313": 0.2}
    rows = [
        {
            "method": method,
            "dimension": p["dimension"],
            "instance_id": name,
            "seed": seed,
            "cost": 100 + gap,
            "gap_percent": gap,
            "status": "completed",
        }
        for method, gap in gaps.items()
        for p in panels
        for name in p["ids"]
        for seed in p["seeds"]
    ]
    config = {"seeds": SEEDS, "statistics_seed": 91001, "bootstrap_replicates": 100}
    result = comparison_statistics(rows, config, panels)
    assert len(result["per_evolution_seed"]) == 6
    overall = [r for r in result["rows"] if r["dimension"] == "macro"]
    assert [r["improvement_pp"] for r in overall] == pytest.approx([1.9, 1.4])
    assert all(r["wins"] == 4 and r["holm_p_value"] >= r["p_value"] for r in overall)
    assert result == comparison_statistics(list(reversed(rows)), config, panels)
