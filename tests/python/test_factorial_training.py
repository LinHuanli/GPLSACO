"""假worker只验证任务身份、DEAP完整重训和恢复，不作为研究效果证据。"""

import copy
from dataclasses import replace

import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.checkpoint import load_checkpoint
from gp_faco.evolution import EvolutionSettings
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.factorial_training import FactorialTrainingRun
from gp_faco.fitness import score_panel
from gp_faco.program_ir import Program
from gp_faco.training import TrainingSettings
from gp_faco.worker import FactorialTask, SolverSettings, WorkerProtocol
from test_training import ObservedFuture, Source, WorkerFarm


class FactorialFarm(WorkerFarm):
    def __call__(self, protocol):
        worker = super().__call__(protocol)
        original = worker.submit

        def submit(task):
            assert type(task) is FactorialTask
            future = original(task)
            # 延续已验证的合成返回，只增加原生要求的因素配置回执。
            value = copy.deepcopy(future._result)
            value.update(task.controller_identity())
            value["native_result"]["factorial_policy"] = task.factorial_policy.to_dict()
            return ObservedFuture(value=value, timeouts=int(self.timeout))

        worker.submit = submit
        return worker


def settings_and_protocol():
    settings = TrainingSettings(
        evolution=EvolutionSettings(population=6, generations=3, elites=2, feature_spec_id=2),
        instances_per_panel=2,
        budgets=((5, 16), (7, 32)),
        budget_kind="search_tour_evaluations",
        preparation_mode="cached",
    )
    protocol = WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "NVIDIA RTX A5000",
        "0",
        "test-build",
        dimensions=(5, 7),
        colonies=4,
        settings=SolverSettings(ants=4),
        maximum_registered_per_dimension=16,
    )
    return settings, protocol


@pytest.mark.parametrize("variant", ["M10", "M01"])
def test_independent_training_full_evaluations_resume_and_policy_identity(tmp_path, variant):
    settings, protocol = settings_and_protocol()
    source = Source()
    policy = FactorialPolicy(
        variant,
        BaselinePolicy(
            kind="rule",
            max_mne_level=3,
            stagnation_step=2,
            restart_stagnation=3,
            restart_cooldown=3,
        ),
    )
    uninterrupted_farm, farm = FactorialFarm(), FactorialFarm(timeout=True)
    full = FactorialTrainingRun(
        tmp_path / "full",
        settings,
        protocol,
        source.data(),
        factorial_policy=policy,
        worker_factory=uninterrupted_farm,
    )
    full.run()
    paused = FactorialTrainingRun(
        tmp_path / "resumed",
        settings,
        protocol,
        source.data(),
        factorial_policy=policy,
        worker_factory=farm,
    )
    assert paused.run(stop_after_tasks=5)["status"] == "paused"
    before = load_checkpoint(paused.path)
    for bad_policy in (
        replace(policy, variant="M01" if variant == "M10" else "M10"),
        replace(policy, baseline_policy=replace(policy.baseline_policy, region=1)),
    ):
        with pytest.raises(ValueError, match="恢复"):
            FactorialTrainingRun(
                paused.directory,
                settings,
                protocol,
                source.data(),
                resume=True,
                factorial_policy=bad_policy,
                worker_factory=farm,
            )
    restored = FactorialTrainingRun(
        paused.directory,
        settings,
        protocol,
        source.data(),
        resume=True,
        factorial_policy=policy,
        worker_factory=farm,
    )
    restored.run()
    final, expected = load_checkpoint(restored.path), load_checkpoint(full.path)
    assert final["phase"] == "complete"
    assert final["evolution"] == expected["evolution"]
    assert without_task_locations(final["validation_results"]) == without_task_locations(
        expected["validation_results"]
    )
    assert without_task_locations(final["selected"]) == without_task_locations(expected["selected"])
    assert all(final["completed"][k] == v for k, v in before["completed"].items())
    assert len(farm.submissions) == final["costs"]["solve_jobs"]
    assert len({t.occurrence_id for t in farm.submissions}) == len(farm.submissions)
    assert all(t.factorial_policy == policy for t in farm.submissions)
    training_tasks = [t for t in farm.submissions if ":generation" in t.occurrence_id]
    assert len(training_tasks) == 6 * 3 * 2
    assert all(
        sum(f":generation{g}:" in t.occurrence_id for t in training_tasks) == 6 * 2
        for g in range(3)
    )
    assert len(final["training_panels"]) == 3
    assert final["costs"]["failed_solves"] == 0
    assert len(farm.submissions) >= 6 * 3 * 2
    assert final["costs"]["search_tour_evaluations"] == sum(
        t.evaluation_limit_per_colony * protocol.colonies for t in farm.submissions
    )


def test_task_identity_native_policy_tamper_and_no_wall_clock():
    _, protocol = settings_and_protocol()
    source = Source()
    problems = tuple(source.problems[k] for k in source.training[5][:2])
    policy = FactorialPolicy("M10", BaselinePolicy())
    task = FactorialTask(
        "g0:p0",
        Program((0,), (4,), feature_spec_id=2),
        problems,
        tuple((p.instance_id, seed) for p in problems for seed in (17, 29)),
        preparation_mode="cached",
        evaluation_limit_per_colony=16,
        factorial_policy=policy,
    )
    other = replace(task, factorial_policy=replace(policy, variant="M01"))
    assert task.manifest(protocol) != other.manifest(protocol)
    assert task.controller_id != other.controller_id
    farm = FactorialFarm()
    outcome = farm(protocol).submit(task).result()
    labels = {p.instance_id: source.load_label(p.instance_id) for p in problems}
    assert not score_panel(task, protocol, outcome, labels).failed
    outcome["native_result"]["factorial_policy"] = other.factorial_policy.to_dict()
    assert score_panel(task, protocol, outcome, labels).failed
    with pytest.raises(ValueError):
        replace(task, evaluation_limit_per_colony=None, budget_seconds=1.0)
    with pytest.raises(ValueError):
        FactorialPolicy.from_dict(policy.to_dict() | {"extra": 0})


def test_retraining_rejects_wrong_grammar_and_nonlearned_variant(tmp_path):
    settings, protocol = settings_and_protocol()
    for variant, no_feedback in (("M00", False), ("M11", False), ("M01", True)):
        with pytest.raises(ValueError):
            FactorialTrainingRun(
                tmp_path / variant,
                replace(settings, evolution=replace(settings.evolution, no_feedback=no_feedback)),
                protocol,
                Source().data(),
                factorial_policy=FactorialPolicy(variant, BaselinePolicy()),
                worker_factory=FactorialFarm(),
            )


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
