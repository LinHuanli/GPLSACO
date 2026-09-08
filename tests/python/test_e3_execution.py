"""正式组装和原始返回事务测试；假worker只证明编排，不提供性能结论。"""

import copy
import json
from types import SimpleNamespace

import pytest
from gp_faco.checkpoint import atomic_json, load_checkpoint
from gp_faco.e3_execution import (
    RegisteredGraphStaticRun,
    RegisteredGraphTrainingRun,
    make_static,
    make_training,
    static_config,
)
from gp_faco.e3_protocol import CONDITIONS, training_panels
from gp_faco.evolution import EvolutionSettings
from gp_faco.training import TrainingSettings
from gp_faco.training_audit import audit_receipts, replay_evolution, verify_panels
from gp_faco.worker import PROJECT, SolverSettings, WorkerProtocol, file_hash
from test_configuration_search import Source as StaticSource
from test_configuration_search import WorkerFarm as StaticFarm
from test_training import Source, WorkerFarm


def free_boundary(*_args):
    return {
        "foreign_processes": 0,
        "memory_used_mib": 0,
        "utilization_percent": 0,
        "query_seconds": 0.0,
    }


@pytest.fixture
def counted():
    settings = TrainingSettings(
        evolution=EvolutionSettings(population=8, generations=3, feature_spec_id=2),
        instances_per_panel=2,
        budgets=((5, 16), (7, 16)),
        preparation_mode="cached",
        budget_kind="search_tour_evaluations",
        scope="E3_unit_orchestration",
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
    source = Source()
    panels = training_panels(
        {
            "dimensions": [5, 7],
            "evolution": {"generations": 3},
            "training": {
                "panel_seed": settings.panel_seed,
                "instances_per_panel": 2,
                "solver_seeds_per_instance": 2,
            },
        },
        {str(n): {"train": ids} for n, ids in source.training.items()},
    )
    return settings, protocol, source, panels


def rich_farm(farm):
    def factory(protocol):
        worker = farm(protocol)

        def richer(method):
            def call(*args):
                future = method(*args)

                class WithPid:
                    def result(self, timeout=None):
                        outcome = future.result(timeout=timeout)
                        outcome["worker_pid"] = worker.pid
                        return outcome

                return WithPid()

            return call

        worker.prepare, worker.submit = richer(worker.prepare), richer(worker.submit)
        return worker

    return factory


def test_formal_builders_keep_all_seeds_population_and_static_grid(tmp_path):
    config = json.loads((PROJECT / "configs/e3_protocol_v1.json").read_text())
    members = {
        str(n): {
            role: [f"{n}-{role}-{i:06d}" for i in range(count)]
            for role, count in config["data"]["expected_counts"][str(n)].items()
        }
        for n in config["dimensions"]
    }
    plan = {
        "config": config,
        "sha256": "a" * 64,
        "database_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "config_sha256": "d" * 64,
    }
    execution = {
        "sha256": "e" * 64,
        "entrypoint_sha256": "f" * 64,
        "catalog_sha256": "0" * 64,
        "static_configs": {},
    }

    def no_read(*_args):
        pytest.fail("组装阶段不得读取坐标或标签")

    source = SimpleNamespace(load_instance=no_read, load_label=no_read)
    development = {str(n): [f"dev-{n}-{i:04d}" for i in range(128)] for n in config["dimensions"]}
    total_fe = 0
    for condition, _, _ in CONDITIONS:
        for seed in config["evolution_seeds"]:
            settings, data = make_training(source, plan, execution, members, condition, seed)
            assert settings.evolution.population == 128 and settings.evolution.generations == 50
            assert not settings.evolution.no_feedback and settings.evolution_seed == seed
            assert settings.budget_kind == "search_tour_evaluations"
            assert settings.budgets == ((500, 4096), (1000, 4096))
            assert len(data.training[500]) == 63744 and len(data.training[1000]) == 63648
            assert len(data.validation[500]) == len(data.validation[1000]) == 256
            total_fe += 128 * 50 * 2 * 32 * 4096
        path = tmp_path / f"{condition}.json"
        atomic_json(path, static_config(plan, condition))
        execution["static_configs"][condition] = {
            "path": str(path.relative_to(PROJECT)),
            "sha256": file_hash(path),
        }
        settings, policies, data = make_static(source, plan, execution, development, condition)
        assert settings.selection_kinds == ("static",) and len(policies) == 160
        assert len({p.sha256 for p in policies}) == 160
        assert all(
            len(data.pools["search"][n]) == 64 and len(data.pools["validation"][n]) == 32
            for n in config["dimensions"]
        )
    assert total_fe == 33554432000
    with pytest.raises(ValueError):
        make_training(source, plan, execution, members, "alpha-Hard", 9999)
    with pytest.raises(ValueError):
        static_config(plan, "unregistered")


def test_guarded_training_resume_and_streaming_replay(counted, tmp_path):
    settings, protocol, source, panels = counted
    farm = WorkerFarm(timeout=True)
    factory = rich_farm(farm)
    directory = tmp_path / "resumed"
    run = RegisteredGraphTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=factory,
    )
    assert run.run(stop_after_tasks=5)["status"] == "paused"
    before = load_checkpoint(directory / "checkpoint.json")
    hashes = {p.name: file_hash(p) for p in (directory / "raw_returns").glob("*.json")}
    run = RegisteredGraphTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=factory,
        resume=True,
    )
    assert run.run()["status"] == "complete"
    state = load_checkpoint(directory / "checkpoint.json")
    assert all(state["completed"][key] == value for key, value in before["completed"].items())
    assert all(
        file_hash(directory / "raw_returns" / name) == digest for name, digest in hashes.items()
    )
    assert set(state["completed"]) == {p.stem for p in (directory / "raw_returns").glob("*.json")}
    for key in state["completed"]:
        record = load_checkpoint(directory / "tasks" / f"{key}.json")
        raw = json.loads((directory / "raw_returns" / f"{key}.json").read_text())
        assert "gpu_boundary_after" not in raw
        assert raw == {k: v for k, v in record["outcome"].items() if k != "gpu_boundary_after"}
    selection = load_checkpoint(directory / "data_selection.json")
    verify_panels(state, selection, settings, panels, True)
    evaluated, totals, phases, observations, _ = audit_receipts(
        directory, state, protocol, settings, source, set(source.problems), True
    )
    replay = replay_evolution(directory, state, evaluated, protocol, settings, True)
    assert phases["training"]["solve_jobs"] == 48 and replay["completed_generations"] == 3
    assert totals["solve_jobs"] == len(farm.submissions) and not any(observations.values())


def test_wrong_registered_panel_stops_before_native(counted, tmp_path):
    settings, protocol, source, panels = counted
    wrong = copy.deepcopy(panels)
    wrong[0][0]["seeds"][0] ^= 1
    farm = WorkerFarm()
    run = RegisteredGraphTrainingRun(
        tmp_path / "wrong",
        settings,
        protocol,
        source.data(),
        expected_panels=wrong,
        boundary=free_boundary,
        worker_factory=rich_farm(farm),
    )
    with pytest.raises(ValueError, match="代面板"):
        run.run()
    assert not farm.submissions and not farm.preparations


def test_unknown_after_boundary_keeps_return_without_resubmission(counted, tmp_path):
    settings, protocol, source, panels = counted
    farm = WorkerFarm()
    calls = 0

    def boundary(*_args):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("injected GPU observer failure after first solve")
        return free_boundary()

    directory = tmp_path / "observer"
    run = RegisteredGraphTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=boundary,
        worker_factory=rich_farm(farm),
    )
    with pytest.raises(ValueError, match="任务结束"):
        run.run()
    assert len(farm.submissions) == 1
    before = {p.name: file_hash(p) for p in (directory / "raw_returns").glob("*.json")}
    run = RegisteredGraphTrainingRun(
        directory,
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=rich_farm(farm),
        resume=True,
    )
    with pytest.raises(ValueError, match="任务结束"):
        run.run()
    assert len(farm.submissions) == 1
    assert before == {p.name: file_hash(p) for p in (directory / "raw_returns").glob("*.json")}


def test_nonfinite_native_return_is_retained_as_failed_fitness(counted, tmp_path):
    settings, protocol, source, panels = counted
    farm = WorkerFarm(invalid="nan")
    run = RegisteredGraphTrainingRun(
        tmp_path / "invalid",
        settings,
        protocol,
        source.data(),
        expected_panels=panels,
        boundary=free_boundary,
        worker_factory=rich_farm(farm),
    )
    result = run.run(stop_after_tasks=1)
    assert result["status"] == "paused" and len(farm.submissions) == 1
    state = load_checkpoint(run.path)
    assert state["costs"]["failed_solves"] == 1
    record = next(
        load_checkpoint(p)
        for p in (run.directory / "tasks").glob("*.json")
        if load_checkpoint(p)["kind"] == "solve"
    )
    assert record["outcome"]["native_result"]["items"][0]["cost"] == {"invalid_float": "nan"}


def test_static_wrapper_saves_raw_before_boundary_and_resumes(tmp_path):
    from gp_faco.baseline_policy import BaselinePolicy
    from gp_faco.configuration_search import SearchSettings

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
    farm = StaticFarm(timeout=True)
    policies = tuple(BaselinePolicy(mne_level=n, max_mne_level=n) for n in (0, 1))
    directory = tmp_path / "static"
    run = RegisteredGraphStaticRun(
        directory,
        settings,
        protocol,
        policies,
        StaticSource().data(),
        worker_factory=farm,
        boundary=free_boundary,
    )
    assert run.run(stop_after_tasks=5)["status"] == "paused"
    preserved = {p.name: file_hash(p) for p in (directory / "raw_returns").glob("*.json")}
    run = RegisteredGraphStaticRun(
        directory,
        settings,
        protocol,
        policies,
        StaticSource().data(),
        worker_factory=farm,
        boundary=free_boundary,
        resume=True,
    )
    result = run.run()
    assert (
        result["status"] == "complete"
        and result["costs"]["solve_jobs"] == len(farm.submissions) == 12
    )
    assert all(
        file_hash(directory / "raw_returns" / name) == digest for name, digest in preserved.items()
    )
    assert len(list((directory / "raw_returns").glob("*.json"))) == 12
    assert all(
        "gpu_boundary_after" not in json.loads(p.read_text())
        for p in (directory / "raw_returns").glob("*.json")
    )
