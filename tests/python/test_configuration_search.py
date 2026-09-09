"""基线身份、完整配置搜索及恢复中断窗口；GPU生产调用另有实际开发数据验收。"""

import copy
import math
from concurrent.futures import TimeoutError
from dataclasses import replace

import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.checkpoint import load_checkpoint
from gp_faco.configuration_search import ConfigurationRun, SearchData, SearchSettings
from gp_faco.data import Instance, Label, tour_cost
from gp_faco.fitness import score_panel
from gp_faco.worker import BaselineTask, SolverSettings, WorkerProtocol


class Source:
    def __init__(self):
        self.instances = {}
        self.label_reads = []
        for n in (5, 7):
            for i in range(6):
                name = f"{n}-{i}"
                points = tuple(
                    (
                        i * 3 + (1 + i / 10) * math.cos(2 * math.pi * j / n + i / 3),
                        (1 + i / 10) * math.sin(2 * math.pi * j / n + i / 3),
                    )
                    for j in range(n)
                )
                self.instances[name] = Instance(name, points)

    def load_instance(self, name):
        return self.instances[name]

    def load_label(self, name):
        self.label_reads.append(name)
        problem = self.instances[name]
        tour = tuple(range(problem.dimension))
        return Label(tour, tour_cost(problem, tour))

    def data(self, *, calibration=False):
        pools = {"search": {n: [f"{n}-{i}" for i in range(4)] for n in (5, 7)}}
        if not calibration:
            pools["validation"] = {n: [f"{n}-{i}" for i in (4, 5)] for n in (5, 7)}
        return SearchData(self, pools, {"source": "unit-circle-boundary-reference"})


def free_boundary(*_):
    return {"foreign_processes": 0, "memory_used_mib": 1, "query_seconds": 0.0}


class WorkerFarm:
    def __init__(self, *, timeout=False, fail=False):
        self.timeout, self.fail = timeout, fail
        self.submissions, self.workers, self.outcomes = [], [], []

    def __call__(self, protocol):
        farm = self

        class Worker:
            def __init__(self):
                self.pid = 900000000 + len(farm.workers)
                self.closed = False
                farm.workers.append(self)

            def ready(self, timeout=None):
                return {
                    "pid": self.pid,
                    "host": protocol.execution_host,
                    "protocol_id": protocol.identifier,
                    "start_method": "spawn",
                }

            def submit(self, task):
                assert isinstance(task, BaselineTask)
                farm.submissions.append(task)
                by_name = {p.instance_id: p for p in task.problems}
                items = []
                for name, _ in task.replicas:
                    problem = by_name[name]
                    tour = list(range(problem.dimension))
                    # 正多边形边界给出手工明确的最优配置；非最优配置产生交叉边。
                    best = 1 if task.policy.kind == "static" else 2
                    if task.policy.mne_level != best and task.evaluation_limit_per_colony:
                        tour[1], tour[2] = tour[2], tour[1]
                    items.append(
                        {
                            "has_incumbent": not farm.fail,
                            "tour": tour,
                            "cost": tour_cost(problem, tour),
                            "completed_seconds": 1.0,
                        }
                    )
                limit = task.evaluation_limit_per_colony
                result = {
                    "baseline_policy": task.policy.to_dict(),
                    "items": items,
                    "budget_kind": "search_tour_evaluations",
                    "budget_seconds": None,
                    "evaluation_limit_per_colony": limit,
                    "completed_tour_evaluations_per_colony": limit,
                    "total_tour_evaluations": limit * protocol.colonies,
                    "preparation_mode": task.preparation_mode,
                    "preparation_completed": True,
                    "actual_seconds": 1.0,
                    "elapsed_seconds": 1.0,
                    "charged_seconds": 0.0,
                    "overrun_seconds": 0.0,
                    "discarded_batches": 0,
                    "launched_batches": limit // protocol.settings.ants,
                    "completed_batches": limit // protocol.settings.ants,
                    "completed_ls_evaluations": 10 * limit,
                    "completed_construction_steps": 2 * limit,
                }
                outcome = {
                    "status": "completed",
                    "task_id": task.task_id(protocol),
                    "protocol_id": protocol.identifier,
                    **task.controller_identity(),
                    "dimension": task.dimension,
                    "occurrence_id": task.occurrence_id,
                    "native_result": result,
                    "worker_seconds": 1.1,
                    "registration_seconds": 0.1,
                    "worker_pid": self.pid,
                }
                farm.outcomes.append(outcome)

                class Future:
                    observed = False

                    def result(inner, timeout=None):
                        if farm.timeout and not inner.observed:
                            inner.observed = True
                            raise TimeoutError
                        return outcome

                return Future()

            def close(self):
                self.closed = True

        return Worker()


@pytest.fixture
def setup():
    settings = SearchSettings(
        purpose="tuning",
        evaluation_limits=(16,),
        instances_per_panel=2,
        validation_shortlist_per_kind=2,
    )
    protocol = WorkerProtocol(
        "GPU-34b223c6-7502-b097-19e0-a411b1708f06",
        "NVIDIA RTX A5000",
        "0",
        "test-build",
        dimensions=(5, 7),
        colonies=4,
        settings=SolverSettings(ants=4),
        maximum_registered_per_dimension=16,
    )
    policies = tuple(
        BaselinePolicy(kind=kind, mne_level=level, max_mne_level=level)
        for kind, levels in (("static", (0, 1)), ("rule", (0, 2)))
        for level in levels
    )
    return settings, protocol, policies


def test_baseline_policy_task_identity_and_actual_configuration_verification(setup):
    settings, protocol, policies = setup
    for policy in policies:
        assert BaselinePolicy.from_dict(policy.to_dict()) == policy
        assert policy.identifier != replace(policy, region=1).identifier
    for changed in (
        {"policy_spec_id": True},
        {"kind": "gp"},
        {"mne_level": -1},
        {"kind": "rule", "restart_stagnation": 8},
        {"restart_probability": True},
    ):
        with pytest.raises(ValueError):
            BaselinePolicy(**changed)
    source = Source()
    problems = source.data().problems(["5-0", "5-1"])
    task = BaselineTask(
        "position",
        policies[0],
        problems,
        tuple((p.instance_id, seed) for p in problems for seed in (17, 29)),
        16,
    )
    farm = WorkerFarm()
    worker = farm(protocol)
    outcome = worker.submit(task).result()
    labels = {p.instance_id: source.load_label(p.instance_id) for p in problems}
    assert not score_panel(task, protocol, outcome, labels).failed
    changed = copy.deepcopy(outcome)
    changed["native_result"]["baseline_policy"] = policies[1].to_dict()
    assert score_panel(task, protocol, changed, labels).failed
    assert "program_id" not in task.manifest(protocol)
    assert replace(task, evaluation_limit_per_colony=20).manifest(protocol) != task.manifest(
        protocol
    )
    with pytest.raises(ValueError):
        replace(task, evaluation_limit_per_colony=True)
    with pytest.raises(ValueError):
        replace(task, evaluation_limit_per_colony=17).manifest(protocol)


@pytest.mark.parametrize("purpose,calls,shortlist", [("tuning", 24, 4), ("static_tuning", 12, 2)])
def test_full_grid_fixed_validation_and_pause_resume(setup, tmp_path, purpose, calls, shortlist):
    settings, protocol, policies = setup
    settings = replace(settings, purpose=purpose)
    policies = tuple(p for p in policies if p.kind in settings.selection_kinds)
    first, second = WorkerFarm(), WorkerFarm(timeout=True)
    baseline = ConfigurationRun(
        tmp_path / "full",
        settings,
        protocol,
        policies,
        Source().data(),
        worker_factory=first,
    )
    expected = baseline.run()
    paused = ConfigurationRun(
        tmp_path / "resumed",
        settings,
        protocol,
        policies,
        Source().data(),
        worker_factory=second,
    )
    assert paused.run(stop_after_tasks=5)["status"] == "paused"
    before = load_checkpoint(paused.path)
    resumed = ConfigurationRun(
        paused.directory,
        settings,
        protocol,
        policies,
        Source().data(),
        worker_factory=second,
        resume=True,
    )
    result = resumed.run()
    assert result["status"] == "complete" and without_task_locations(
        result["selected"]
    ) == without_task_locations(expected["selected"])
    assert len(result["shortlist"]) == shortlist
    assert set(result["selected"]) == set(settings.selection_kinds)
    assert result["selected"]["static"]["policy"]["mne_level"] == 1
    if "rule" in settings.selection_kinds:
        assert result["selected"]["rule"]["policy"]["mne_level"] == 2
    assert len(second.submissions) == result["costs"]["solve_jobs"] == calls
    assert result["costs"]["search_tour_evaluations"] == calls * 16 * 4
    assert result["costs"]["observer_timeouts"] == calls
    assert result["costs"]["charged_seconds"] == result["costs"]["overrun_seconds"] == 0
    assert [t.task_id(protocol).split(":", 1)[1] for t in first.submissions] == [
        t.task_id(protocol).split(":", 1)[1] for t in second.submissions
    ]
    assert all(resumed.state["completed"][k] == value for k, value in before["completed"].items())
    assert all(worker.closed for worker in second.workers) and len(second.workers) == 3
    assert (
        load_checkpoint(resumed.directory / "selected_baselines.json")["selected"]
        == result["selected"]
    )


def test_static_tuning_rejects_extra_policy_kind_or_multiple_budgets(setup, tmp_path):
    settings, protocol, policies = setup
    with pytest.raises(ValueError, match="预登记基线种类"):
        ConfigurationRun(
            tmp_path / "wrong-kind",
            replace(settings, purpose="static_tuning"),
            protocol,
            policies,
            Source().data(),
            worker_factory=WorkerFarm(),
        )
    with pytest.raises(ValueError, match="预定主档"):
        replace(settings, purpose="static_tuning", evaluation_limits=(16, 32))


def test_all_failed_search_is_retained_and_exports_no_selected_baseline(setup, tmp_path):
    settings, protocol, policies = setup
    farm = WorkerFarm(fail=True)
    run = ConfigurationRun(
        tmp_path / "failed",
        settings,
        protocol,
        policies,
        Source().data(),
        worker_factory=farm,
    )
    result = run.run()
    assert result["status"] == "failed" and result["costs"]["failed_solves"] == 16
    assert result["selected"] == {} and len(result["search_scores"]) == 4
    assert not (run.directory / "selected_baselines.json").exists()


def test_calibration_keeps_all_limits_and_does_not_select_methods(setup, tmp_path):
    settings, protocol, policies = setup
    settings = replace(settings, purpose="calibration", evaluation_limits=(0, 8, 16))
    farm = WorkerFarm()
    run = ConfigurationRun(
        tmp_path / "curves",
        settings,
        protocol,
        policies,
        Source().data(calibration=True),
        worker_factory=farm,
    )
    result = run.run()
    assert result["status"] == "complete" and len(farm.submissions) == 48
    assert len(result["search_scores"]) == 12 and not result["selected"]
    assert result["costs"]["search_tour_evaluations"] == (0 + 8 + 16) * 4 * 4 * 4
    assert all(task.budget_seconds is None for task in farm.submissions)


def without_task_locations(value):
    if isinstance(value, dict):
        return {
            k: without_task_locations(v)
            for k, v in value.items()
            if k not in ("task_ids", "task_keys")
        }
    if isinstance(value, list):
        return [without_task_locations(v) for v in value]
    return value
