"""资源偏差只改变计时分类；真实返回、FE与演化恢复不得改变。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from gp_faco.checkpoint import load_checkpoint  # noqa: E402
from gp_faco.e3_execution import RegisteredGraphTrainingRun  # noqa: E402
from gp_faco.training_audit import audit_receipts, replay_evolution, verify_panels  # noqa: E402
from gp_faco.worker import PROJECT, file_hash  # noqa: E402
from resume_e3_observed_return import (  # noqa: E402
    ResourceStaticRun,
    ResourceTrainingRun,
    verify_journals,
)
from test_e3_execution import counted as counted_fixture  # noqa: E402
from test_e3_execution import free_boundary, rich_farm  # noqa: E402
from test_training import WorkerFarm  # noqa: E402


@pytest.fixture
def counted():
    return counted_fixture.__wrapped__()


@pytest.fixture
def policy():
    return json.loads((PROJECT / "provenance/e3_resource_recovery_v1.json").read_text())


def one_after_deviation(kind="foreign"):
    calls = 0

    def boundary(*_args):
        nonlocal calls
        calls += 1
        if calls == 6:
            if kind == "unknown":
                raise OSError("injected observation failure")
            return {**free_boundary(), "foreign_processes": 1}
        return free_boundary()

    return boundary


@pytest.mark.parametrize("kind", ["foreign", "unknown"])
def test_saved_return_recovered_without_new_native_and_full_replay(counted, policy, tmp_path, kind):
    settings, protocol, source, panels = counted
    directory = tmp_path / kind
    farm = WorkerFarm(timeout=True)
    factory = rich_farm(farm)
    run = RegisteredGraphTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=one_after_deviation(kind),
        worker_factory=factory,
    )
    with pytest.raises(ValueError, match="任务结束"):
        run.run()
    before = load_checkpoint(directory / "checkpoint.json")
    raw_hashes = {p.name: file_hash(p) for p in (directory / "raw_returns").glob("*.json")}
    assert len(farm.submissions) == 1 and before["costs"]["solve_jobs"] == 0
    key = before["pending"]["key"]
    observed = load_checkpoint(directory / "tasks" / f"{key}.json")["outcome"]
    run = ResourceTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=factory,
        recovery_policy=policy,
        resume=True,
    )
    assert run.run(stop_after_tasks=1)["status"] == "paused"
    recovered = load_checkpoint(directory / "checkpoint.json")
    assert len(farm.submissions) == 1 and recovered["costs"]["solve_jobs"] == 1
    assert all(recovered["completed"][k] == v for k, v in before["completed"].items())
    assert all(
        file_hash(directory / "raw_returns" / name) == sha for name, sha in raw_hashes.items()
    )
    assert load_checkpoint(directory / "tasks" / f"{key}.json")["outcome"] == observed
    run = ResourceTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=factory,
        recovery_policy=policy,
        resume=True,
    )
    assert run.run()["status"] == "complete"
    state = load_checkpoint(directory / "checkpoint.json")
    journals = verify_journals(directory, state, complete=True)
    assert journals["completed_resource_deviations"] == 1
    verify_panels(state, load_checkpoint(directory / "data_selection.json"), settings, panels, True)
    evaluated, totals, phases, observations, _ = audit_receipts(
        directory,
        state,
        protocol,
        settings,
        source,
        set(source.problems),
        True,
    )
    replay = replay_evolution(directory, state, evaluated, protocol, settings, True)
    assert phases["training"]["solve_jobs"] == 48 and replay["completed_generations"] == 3
    assert totals["solve_jobs"] == len(farm.submissions) and not totals["failed_solves"]
    assert observations["after_foreign" if kind == "foreign" else "after_unknown"] == 1
    # 干净参考使用同一原seed、面板与评价次数；恢复不能改变后续演化或选择。
    clean_dir = tmp_path / "clean"
    clean = RegisteredGraphTrainingRun(
        clean_dir,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=rich_farm(WorkerFarm()),
    )
    assert clean.run()["status"] == "complete"
    control = load_checkpoint(clean_dir / "checkpoint.json")
    assert state["evolution"] == control["evolution"]
    assert state["selected"] == control["selected"]
    for field in ("solve_jobs", "failed_solves", "search_tour_evaluations", "valid_members"):
        assert state["costs"][field] == control["costs"][field]


def test_occupied_or_unknown_gpu_waits_before_any_submission(counted, policy, tmp_path):
    settings, protocol, source, panels = counted
    farm = WorkerFarm()
    observations = iter([1, None, 0])
    waits = []

    def boundary(*_args):
        value = next(observations, 0)
        if value is None:
            raise OSError("injected query failure before submission")
        return {**free_boundary(), "foreign_processes": value}

    def wait(seconds):
        assert not farm.preparations and not farm.submissions
        waits.append(seconds)

    run = ResourceTrainingRun(
        tmp_path / "wait",
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=boundary,
        worker_factory=rich_farm(farm),
        recovery_policy=policy,
        idle_wait=wait,
    )
    assert run.run(stop_after_tasks=1)["status"] == "paused"
    assert waits == [1.0, 1.0] and len(farm.submissions) == 1
    state = load_checkpoint(run.path)
    assert [a["foreign_processes"] for a in state["resource_waits"]] == [1, None]
    assert all(a["foreign_processes"] == 0 for a in state["admission"])


def test_invalid_return_stays_failed_and_is_not_retried(counted, policy, tmp_path):
    settings, protocol, source, panels = counted
    farm = WorkerFarm(invalid="nan")
    run = ResourceTrainingRun(
        tmp_path / "failed",
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=one_after_deviation(),
        worker_factory=rich_farm(farm),
        recovery_policy=policy,
    )
    assert run.run(stop_after_tasks=1)["status"] == "paused"
    state = load_checkpoint(run.path)
    assert state["costs"]["failed_solves"] == 1 and len(farm.submissions) == 1
    assert (
        verify_journals(run.directory, state, complete=False)["completed_resource_deviations"] == 1
    )


def test_changed_raw_return_rejected_before_further_native(counted, policy, tmp_path):
    settings, protocol, source, panels = counted
    farm = WorkerFarm()
    directory = tmp_path / "tampered"
    run = ResourceTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=one_after_deviation(),
        worker_factory=rich_farm(farm),
        recovery_policy=policy,
    )
    assert run.run(stop_after_tasks=1)["status"] == "paused"
    marker = next((directory / "resource_deviations").glob("*.json"))
    key = marker.stem
    raw = directory / "raw_returns" / f"{key}.json"
    value = json.loads(raw.read_text())
    value["worker_pid"] += 1
    raw.write_text(json.dumps(value))
    run = ResourceTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=rich_farm(farm),
        recovery_policy=policy,
        resume=True,
    )
    with pytest.raises(ValueError, match="原始证据"):
        run.run()
    assert len(farm.submissions) == 1


def test_static_return_uses_same_deviation_policy(policy, tmp_path):
    from gp_faco.baseline_policy import BaselinePolicy
    from gp_faco.configuration_search import SearchSettings
    from gp_faco.worker import SolverSettings, WorkerProtocol
    from test_configuration_search import Source
    from test_configuration_search import WorkerFarm as StaticFarm

    settings = SearchSettings(
        purpose="static_tuning",
        evaluation_limits=(16,),
        instances_per_panel=2,
        validation_shortlist_per_kind=2,
    )
    protocol = WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "test-only",
        "0",
        "0" * 64,
        dimensions=(5, 7),
        colonies=4,
        settings=SolverSettings(ants=4),
        maximum_registered_per_dimension=16,
    )
    policies = tuple(BaselinePolicy(mne_level=n, max_mne_level=n) for n in (0, 1))
    farm = StaticFarm()
    calls = 0

    def boundary(*_args):
        nonlocal calls
        calls += 1
        return {**free_boundary(), "foreign_processes": int(calls == 2)}

    run = ResourceStaticRun(
        tmp_path / "static",
        settings,
        protocol,
        policies,
        Source().data(),
        worker_factory=farm,
        boundary=boundary,
        recovery_policy=policy,
    )
    result = run.run()
    assert result["status"] == "complete" and len(farm.submissions) == 12
    assert (
        verify_journals(run.directory, load_checkpoint(run.path), complete=True)[
            "completed_resource_deviations"
        ]
        == 1
    )


def test_interrupted_journal_publication_reuses_original_return(
    counted, policy, tmp_path, monkeypatch
):
    import resume_e3_observed_return as recovery

    settings, protocol, source, panels = counted
    farm = WorkerFarm()
    directory = tmp_path / "journal"
    original_atomic = recovery.atomic_json

    def interrupted(path, value):
        if path.parent == directory / "resource_deviations":
            raise OSError("injected interruption before journal publication")
        return original_atomic(path, value)

    monkeypatch.setattr(recovery, "atomic_json", interrupted)
    run = ResourceTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=one_after_deviation(),
        worker_factory=rich_farm(farm),
        recovery_policy=policy,
    )
    with pytest.raises(OSError, match="journal publication"):
        run.run()
    assert len(farm.submissions) == 1
    original = next((directory / "resource_deviations/original").glob("*.json"))
    original_sha = file_hash(original)
    assert not list((directory / "resource_deviations").glob("*.json"))
    monkeypatch.setattr(recovery, "atomic_json", original_atomic)
    run = ResourceTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=rich_farm(farm),
        recovery_policy=policy,
        resume=True,
    )
    assert run.run(stop_after_tasks=1)["status"] == "paused"
    assert len(farm.submissions) == 1 and file_hash(original) == original_sha
    assert (
        verify_journals(run.directory, load_checkpoint(run.path), complete=False)[
            "completed_resource_deviations"
        ]
        == 1
    )
