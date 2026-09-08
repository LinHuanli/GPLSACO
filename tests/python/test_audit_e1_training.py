"""独立审计的完整/中间事务重放和错误注入；假worker不产生研究性能证据。"""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.fitness import PanelFitness
from gp_faco.worker import content_hash
from test_training import WorkerFarm
from test_training import setup as shared_setup

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_e1_training import (  # noqa: E402
    audit_receipts,
    replay_evolution,
    verify_attempts,
    verify_costs,
    verify_panels,
)
from research_e1 import RegisteredTrainingRun, training_panels  # noqa: E402


class ReceiptFarm(WorkerFarm):
    def __call__(self, protocol):
        worker = super().__call__(protocol)

        def identified(operation):
            def call(*args):
                future = operation(*args)
                future.result()["worker_pid"] = worker.pid
                return future

            return call

        worker.prepare = identified(worker.prepare)
        worker.submit = identified(worker.submit)
        return worker


def fixture_run(tmp_path, *, stop=None, invalid=None):
    settings, protocol, source = shared_setup.__wrapped__()
    settings = replace(
        settings,
        evolution=replace(settings.evolution, feature_spec_id=2),
        budgets=((5, 8), (7, 8)),
        budget_kind="search_tour_evaluations",
        preparation_mode="cached",
        scope="E1_formal_training",
    )
    members = {str(n): {"train": source.training[n]} for n in (5, 7)}
    config = {
        "training": {
            "panel_seed": settings.panel_seed,
            "instances_per_panel": 2,
            "solver_seeds_per_instance": 2,
        },
        "evolution": {"generations": settings.evolution.generations},
        "dimensions": [5, 7],
    }
    panels = training_panels(config, members)
    farm = ReceiptFarm(invalid=invalid)
    run = RegisteredTrainingRun(
        tmp_path,
        settings,
        protocol,
        source.data(),
        worker_factory=farm,
        expected_panels=panels,
        boundary=lambda *_: {"foreign_processes": 0},
    )
    run.run(stop_after_tasks=stop)
    state = load_checkpoint(run.path)
    evaluated = {}
    for task in farm.submissions:
        key = content_hash(
            {"run_id": state["run_id"], "kind": "solve", "description": task.manifest(protocol)}
        )
        record = load_checkpoint(tmp_path / "tasks" / f"{key}.json")
        checked = record["checked"]
        evaluated[task.occurrence_id] = (
            task,
            PanelFitness(**{**checked, "members": tuple(tuple(v) for v in checked["members"])}),
        )
    return state, settings, protocol, source, evaluated, panels


def test_complete_replay_and_streamed_receipt_costs(tmp_path):
    state, settings, protocol, source, evaluated, panels = fixture_run(tmp_path)
    selection = load_checkpoint(tmp_path / "data_selection.json")
    verify_panels(state, selection, settings, panels, True)
    recomputed, totals, phases, observations, error = audit_receipts(
        tmp_path, state, protocol, settings, source, set(source.problems), True
    )
    assert set(recomputed) == set(evaluated) and totals["solve_jobs"] == len(evaluated)
    assert phases["training"]["solve_jobs"] == 36 and error == 0
    assert not any(observations.values())
    report = replay_evolution(tmp_path, state, recomputed, protocol, settings, True)
    assert report["completed_generations"] == 3
    assert report["completed_validation_candidates"] == report["validation_candidates"]


@pytest.mark.parametrize("stop", [4, 5])
def test_snapshot_handles_completed_receipt_before_fitness_assignment(tmp_path, stop):
    state, settings, protocol, _, evaluated, _ = fixture_run(tmp_path, stop=stop)
    report = replay_evolution(tmp_path, state, evaluated, protocol, settings, False)
    assert report["completed_generations"] == 0
    assert len(evaluated) == stop
    assert report["assigned_current_individuals"] == (stop - 1) // 2


def test_missing_task_and_changed_selection_are_rejected(tmp_path):
    state, settings, protocol, _, evaluated, _ = fixture_run(tmp_path)
    missing = dict(evaluated)
    del missing[next(iter(missing))]
    with pytest.raises(RuntimeError, match="缺失完整"):
        replay_evolution(tmp_path, state, missing, protocol, settings, True)
    changed = copy.deepcopy(state)
    changed["selected"]["fitness"] += 1
    with pytest.raises(RuntimeError, match="最终选择"):
        replay_evolution(tmp_path, changed, evaluated, protocol, settings, True)


def test_all_failed_tasks_still_cover_training_and_full_validation(tmp_path):
    state, settings, protocol, source, _, _ = fixture_run(tmp_path, invalid="missing")
    evaluated, totals, _, _, _ = audit_receipts(
        tmp_path, state, protocol, settings, source, set(source.problems), True
    )
    assert state["phase"] == "failed" and totals["failed_solves"] == totals["solve_jobs"]
    report = replay_evolution(tmp_path, state, evaluated, protocol, settings, True)
    assert report["completed_generations"] == 3
    assert not (tmp_path / "selected_program.json").exists()


def test_consistent_hashes_do_not_hide_tampered_route_cost(tmp_path):
    state, settings, protocol, source, _, _ = fixture_run(tmp_path)
    for key in state["completed"]:
        path = tmp_path / "tasks" / f"{key}.json"
        record = load_checkpoint(path)
        if record["kind"] == "solve":
            record["outcome"]["native_result"]["items"][0]["cost"] *= 2
            save_checkpoint(path, record)
            state["completed"][key] = content_hash(record)
            break
    with pytest.raises(RuntimeError, match="重算的外部fitness"):
        audit_receipts(tmp_path, state, protocol, settings, source, set(source.problems), True)


def test_returned_task_is_never_retried_as_infrastructure_failure():
    record = {
        "attempts": [
            {"status": "broken_process_pool", "worker_pid": 7},
            {"status": "returned", "worker_pid": 8},
        ],
        "outcome": {"status": "completed", "worker_pid": 8},
    }
    assert verify_attempts(record, {7, 8}, 1) == (1, 0, True)
    record["attempts"][0]["status"] = "returned"
    with pytest.raises(RuntimeError, match="普通返回"):
        verify_attempts(record, {7, 8}, 1)


def test_large_work_counters_require_exact_integers():
    recorded = {"completed_ls_evaluations": 10**13, "worker_seconds": 3.1}
    verify_costs({**recorded, "worker_seconds": 3.1 + 1e-12}, recorded)
    with pytest.raises(RuntimeError, match="整数工作量"):
        verify_costs({**recorded, "completed_ls_evaluations": 10**13 + 1}, recorded)
