"""假worker仅核验编排/恢复，绝不提供研究性能结论；真实GPU pilot由独立脚本执行。"""

import copy
import math
import platform
import random
from concurrent.futures import Future, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace

import pytest
from gp_faco.checkpoint import load_checkpoint
from gp_faco.data import Instance, Label, tour_cost
from gp_faco.evolution import EvolutionSettings
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings, json_value
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, coordinate_hash


class Source:
    def __init__(self):
        self.problems = {}
        self.training, self.validation = {}, {}
        for n in (5, 7):
            ids = []
            for j in range(10):
                name = f"n{n}-{j:02d}"
                xy = tuple((float(i * i + 7 * j), float(i % 3 + j)) for i in range(n))
                self.problems[name] = Instance(name, xy)
                ids.append(name)
            self.training[n], self.validation[n] = ids[:6], ids[6:]

    def load_instance(self, name):
        return self.problems[name]

    def load_label(self, name):
        problem = self.problems[name]
        tour = tuple(range(problem.dimension))
        return Label(tour, tour_cost(problem, tour))

    def data(self):
        return TrainingData(
            self,
            self.training,
            self.validation,
            {
                "scope": "synthetic orchestration only",
                "records": content_hash(
                    {name: coordinate_hash(p) for name, p in self.problems.items()}
                ),
            },
        )


class ObservedFuture(Future):
    def __init__(self, *, value=None, error=None, timeouts=0):
        super().__init__()
        self.timeouts = timeouts
        self.observations = 0
        if error:
            self.set_exception(error)
        else:
            self.set_result(value)

    def result(self, timeout=None):
        self.observations += 1
        if self.timeouts:
            self.timeouts -= 1
            raise TimeoutError("injected observer timeout")
        return super().result(timeout)


class WorkerFarm:
    def __init__(self, *, broken=0, invalid=None, timeout=False, bad_prepare=False):
        self.workers, self.submissions, self.futures, self.preparations = [], [], [], []
        self.broken, self.invalid, self.timeout, self.bad_prepare = (
            broken,
            invalid,
            timeout,
            bad_prepare,
        )

    def __call__(self, protocol):
        farm = self

        class Worker:
            def __init__(self):
                self.pid = 90000000 + len(farm.workers)
                self.closed = False

            def ready(self, timeout=None):
                return {
                    "pid": self.pid,
                    "host": platform.node(),
                    "start_method": "test_fake",
                    "protocol_sha256": protocol.sha256,
                    "startup_seconds": 0.0,
                }

            def prepare(self, problems):
                farm.preparations.append(problems)
                description = {
                    "protocol_sha256": protocol.sha256,
                    "dimension": problems[0].dimension,
                    "problems": [(p.instance_id, coordinate_hash(p)) for p in problems],
                }
                value = {
                    **description,
                    "kind": "preparation",
                    "status": "completed",
                    "preparation_id": content_hash(description),
                    "worker_seconds": 0.01,
                    "registration_fees": {
                        p.instance_id: {"cheap_seconds": 0.001, "preparation_seconds": 0.01}
                        for p in problems
                    },
                }
                if farm.bad_prepare:
                    value.update(status="failed", error="ordinary preparation failure")
                return ObservedFuture(value=value)

            def submit(self, task):
                assert all(type(p) is Instance for p in task.problems)
                farm.submissions.append(task)
                if farm.broken:
                    farm.broken -= 1
                    future = ObservedFuture(
                        error=BrokenProcessPool("injected confirmed pool failure")
                    )
                else:
                    items = []
                    problems = {p.instance_id: p for p in task.problems}
                    for name, seed in task.replicas:
                        p = problems[name]
                        tour = list(range(p.dimension))
                        random.Random(int(task.program.sha256[:16], 16) ^ seed).shuffle(tour)
                        items.append(
                            {
                                "has_incumbent": True,
                                "tour": tour,
                                "cost": tour_cost(p, tour),
                                "completed_seconds": 0.2,
                            }
                        )
                    charge = math.fsum(a + b for _, a, b in task.preparation_charges or ())
                    result = {
                        "items": items,
                        "budget_seconds": task.budget_seconds,
                        "preparation_mode": task.preparation_mode,
                        "preparation_completed": True,
                        "actual_seconds": 0.5,
                        "charged_seconds": charge,
                        "elapsed_seconds": 0.5 + charge,
                        "overrun_seconds": 0,
                        "launched_batches": 1,
                        "completed_batches": 1,
                        "discarded_batches": 0,
                    }
                    if task.evaluation_limit_per_colony is not None:
                        limit = task.evaluation_limit_per_colony
                        result.update(
                            budget_kind="search_tour_evaluations",
                            evaluation_limit_per_colony=limit,
                            completed_tour_evaluations_per_colony=limit,
                            total_tour_evaluations=limit * protocol.colonies,
                            launched_batches=limit // protocol.settings.ants,
                            completed_batches=limit // protocol.settings.ants,
                        )
                    if farm.invalid == "missing":
                        items[0]["has_incumbent"] = False
                    elif farm.invalid == "nan":
                        items[0]["cost"] = float("nan")
                    value = {
                        "status": "completed",
                        "task_id": task.task_id(protocol),
                        "protocol_sha256": protocol.sha256,
                        "program_sha256": task.program.sha256,
                        "occurrence_id": task.occurrence_id,
                        "dimension": task.dimension,
                        "worker_seconds": 0.5,
                        "native_result": result,
                    }
                    future = ObservedFuture(value=value, timeouts=int(farm.timeout))
                farm.futures.append(future)
                return future

            def close(self):
                self.closed = True

        worker = Worker()
        self.workers.append(worker)
        return worker


@pytest.fixture
def setup():
    settings = TrainingSettings(
        evolution=EvolutionSettings(population=6, generations=3, elites=2),
        instances_per_panel=2,
        budgets=((5, 1.0), (7, 1.0)),
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
    return settings, protocol, Source()


def test_count_mode_resume_preserves_limits_and_full_fitness_evaluations(tmp_path, setup):
    settings, protocol, source = setup
    settings = replace(
        settings,
        evolution=replace(settings.evolution, feature_spec_id=2),
        budget_kind="search_tour_evaluations",
        budgets=((5, 16), (7, 32)),
        preparation_mode="cached",
    )
    baseline_farm, resumed_farm = WorkerFarm(), WorkerFarm(timeout=True)
    baseline = TrainingRun(
        tmp_path / "counts-full", settings, protocol, source.data(), worker_factory=baseline_farm
    )
    expected = baseline.run()
    paused = TrainingRun(
        tmp_path / "counts-resume", settings, protocol, source.data(), worker_factory=resumed_farm
    )
    assert paused.run(stop_after_tasks=5)["status"] == "paused"
    before = load_checkpoint(paused.path)
    resumed = TrainingRun(
        paused.directory,
        settings,
        protocol,
        source.data(),
        worker_factory=resumed_farm,
        resume=True,
    )
    result = resumed.run()
    assert result["status"] == "complete" and result["selected"] == expected["selected"]
    assert resumed.evolution.state_dict() == baseline.evolution.state_dict()
    assert [t.task_id(protocol) for t in baseline_farm.submissions] == [
        t.task_id(protocol) for t in resumed_farm.submissions
    ]
    assert all(resumed.state["completed"][k] == v for k, v in before["completed"].items())
    assert all(
        t.budget_seconds is None
        and t.program.feature_spec_id == 2
        and t.preparation_charges is None
        and t.evaluation_limit_per_colony == dict(settings.budgets)[t.dimension]
        for t in resumed_farm.submissions
    )
    assert result["costs"]["charged_seconds"] == result["costs"]["overrun_seconds"] == 0
    assert result["costs"]["search_tour_evaluations"] == sum(
        t.evaluation_limit_per_colony * protocol.colonies for t in resumed_farm.submissions
    )


def test_every_occurrence_common_panels_and_all_validation_candidates(tmp_path, setup):
    settings, protocol, source = setup
    farm = WorkerFarm()
    settings = replace(
        settings,
        evolution=replace(settings.evolution, crossover_probability=0, mutation_probability=0),
    )
    run = TrainingRun(tmp_path / "common", settings, protocol, source.data(), worker_factory=farm)
    # 全部重复仍实际评价每个位置；验证阶段才按程序身份去重。
    run.evolution.population = [copy.deepcopy(run.evolution.population[0]) for _ in range(6)]
    report = run.run()
    assert report["status"] == "complete"
    training = [t for t in farm.submissions if ":generation" in t.occurrence_id]
    validation = [t for t in farm.submissions if ":validation:" in t.occurrence_id]
    assert len(training) == 3 * 6 * 2 and len(validation) == 4
    assert len({t.task_id(protocol) for t in farm.submissions}) == len(farm.submissions)
    assert len({t.program.sha256 for t in training}) == 1
    panel_seeds = set()
    for generation in range(3):
        for n in protocol.dimensions:
            tasks = [
                t
                for t in training
                if f":generation{generation}:" in t.occurrence_id and t.dimension == n
            ]
            assert len(tasks) == 6 and len({t.replicas for t in tasks}) == 1
            seeds = tuple(seed for _, seed in tasks[0].replicas[:2])
            assert seeds not in panel_seeds
            panel_seeds.add(seeds)
    assert len(farm.workers) == 2 and all(w.closed for w in farm.workers)
    assert all(v.fitness.valid for v in run.evolution.population)
    assert len(run.evolution.winners) == 3 and len(run.evolution.shortlist()) == 1
    assert report["costs"]["valid_members"] == 4 * len(farm.submissions)
    exported = load_checkpoint(run.directory / "selected_program.json")
    assert exported["selection"]["program_sha256"] == run.evolution.shortlist()[0].sha256
    assert exported["manifest"]["settings"]["scope"] == "engineering_development"


def test_partial_individual_resume_matches_uninterrupted_ir_rng_and_tasks(tmp_path, setup):
    settings, protocol, source = setup
    original_rng = random.getstate()
    baseline_farm = WorkerFarm()
    baseline = TrainingRun(
        tmp_path / "baseline", settings, protocol, source.data(), worker_factory=baseline_farm
    )
    baseline_report = baseline.run()
    farm = WorkerFarm(timeout=True)
    interrupted = TrainingRun(
        tmp_path / "resumed", settings, protocol, source.data(), worker_factory=farm
    )
    assert interrupted.run(stop_after_tasks=5)["status"] == "paused"
    before = load_checkpoint(interrupted.path)
    assert sum(v["valid"] for v in before["evolution"]["population"]) == 2
    assert before["costs"]["solve_jobs"] == 5 and before["pending"] is None
    resumed = TrainingRun(
        tmp_path / "resumed", settings, protocol, source.data(), worker_factory=farm, resume=True
    )
    report = resumed.run()
    assert report["status"] == "complete" and report["selected"] == baseline_report["selected"]
    assert json_value(resumed.evolution.state_dict()) == json_value(baseline.evolution.state_dict())
    assert resumed.state["panel_rng"] == baseline.state["panel_rng"]
    assert [t.task_id(protocol) for t in farm.submissions] == [
        t.task_id(protocol) for t in baseline_farm.submissions
    ]
    assert len({t.task_id(protocol) for t in farm.submissions}) == len(farm.submissions)
    assert set(before["completed"]) <= set(resumed.state["completed"])
    assert all(resumed.state["fees"][name] == fee for name, fee in before["fees"].items())
    assert all(f.observations == 2 for f in farm.futures)
    assert report["costs"]["observer_timeouts"] == report["costs"]["solve_jobs"]
    assert random.getstate() == original_rng
    validation = [t for t in farm.submissions if ":validation:" in t.occurrence_id]
    by_program = {}
    for task in validation:
        by_program.setdefault(task.program.sha256, []).append(
            (task.dimension, task.replicas, task.preparation_charges)
        )
    assert len(by_program) == len(resumed.evolution.shortlist())
    assert all(v == next(iter(by_program.values())) for v in by_program.values())


@pytest.mark.parametrize("window", ["raw_receipt", "verified_receipt", "completion_checkpoint"])
def test_result_journal_recovers_without_repeating_solve(tmp_path, setup, monkeypatch, window):
    import gp_faco.training as module

    settings, protocol, source = setup
    farm = WorkerFarm()
    path = tmp_path / window
    run = TrainingRun(path, settings, protocol, source.data(), worker_factory=farm)
    original = module.save_checkpoint

    def write_then_crash(target, value):
        is_solve = target.parent.name == "tasks" and value.get("kind") == "solve"
        if (
            window == "completion_checkpoint"
            and target.name == "checkpoint.json"
            and value["costs"]["solve_jobs"] == 1
        ):
            raise RuntimeError("injected coordinator interruption")
        original(target, value)
        if is_solve and (
            (window == "raw_receipt" and value["checked"] is None)
            or (window == "verified_receipt" and value["checked"] is not None)
        ):
            raise RuntimeError("injected coordinator interruption")

    with monkeypatch.context() as patch:
        patch.setattr(module, "save_checkpoint", write_then_crash)
        with pytest.raises(RuntimeError, match="injected coordinator"):
            run.run()
    assert len(farm.submissions) == 1
    saved = load_checkpoint(run.path)
    assert saved["pending"]["kind"] == "solve" and saved["costs"]["solve_jobs"] == 0
    resumed = TrainingRun(path, settings, protocol, source.data(), worker_factory=farm, resume=True)
    assert resumed.run()["status"] == "complete"
    assert len(farm.submissions) == len({t.task_id(protocol) for t in farm.submissions})
    assert resumed.state["costs"]["solve_jobs"] == len(farm.submissions)
    assert all(w.closed for w in farm.workers)


def test_confirmed_process_failure_retries_original_task_once(tmp_path, setup):
    settings, protocol, source = setup
    farm = WorkerFarm(broken=1)
    run = TrainingRun(tmp_path / "retry", settings, protocol, source.data(), worker_factory=farm)
    assert run.run()["status"] == "complete"
    assert farm.submissions[0] == farm.submissions[1]
    assert len(farm.submissions) == run.state["costs"]["solve_jobs"] + 1
    assert run.state["costs"]["infrastructure_failures"] == 1
    records = [load_checkpoint(p) for p in (run.directory / "tasks").glob("*.json")]
    retried = [r for r in records if len(r["attempts"]) == 2]
    assert len(retried) == 1
    assert [a["status"] for a in retried[0]["attempts"]] == ["broken_process_pool", "returned"]


def test_retry_exhaustion_marks_full_individual_failed_and_still_evaluates_all_scales(
    tmp_path, setup
):
    settings, protocol, source = setup
    farm = WorkerFarm(broken=2)
    run = TrainingRun(
        tmp_path / "retry-limit", settings, protocol, source.data(), worker_factory=farm
    )
    report = run.run()
    assert report["status"] == "complete"
    assert report["costs"]["failed_solves"] == 1 and report["costs"]["infrastructure_failures"] == 2
    assert farm.submissions[0] == farm.submissions[1]
    assert len(farm.submissions) == report["costs"]["solve_jobs"] + 1
    first = load_checkpoint(run.directory / "generations/0000.json")
    assert first["evolution"]["population"][0]["valid"]
    assert first["evolution"]["population"][0]["fitness"] is None
    assert all(v["valid"] for v in first["evolution"]["population"])


@pytest.mark.parametrize("invalid", ["missing", "nan"])
def test_no_finite_validation_retains_failures_and_exports_no_program(tmp_path, setup, invalid):
    settings, protocol, source = setup
    farm = WorkerFarm(invalid=invalid)
    run = TrainingRun(tmp_path / invalid, settings, protocol, source.data(), worker_factory=farm)
    report = run.run()
    assert report["status"] == "failed" and report["selected"] is None
    assert report["costs"]["failed_solves"] == report["costs"]["solve_jobs"]
    assert report["costs"]["valid_members"] == 0
    assert all(
        v.fitness.valid and v.fitness.values == (math.inf,) for v in run.evolution.population
    )
    assert not (run.directory / "selected_program.json").exists()
    assert len(farm.submissions) == len({t.task_id(protocol) for t in farm.submissions})


def test_ordinary_preparation_failure_is_saved_and_never_retried(tmp_path, setup):
    settings, protocol, source = setup
    farm = WorkerFarm(bad_prepare=True)
    run = TrainingRun(tmp_path / "prepare", settings, protocol, source.data(), worker_factory=farm)
    with pytest.raises(RuntimeError, match="ordinary preparation failure"):
        run.run()
    assert len(farm.preparations) == 1 and not farm.submissions
    resumed = TrainingRun(
        run.directory, settings, protocol, source.data(), worker_factory=farm, resume=True
    )
    with pytest.raises(RuntimeError, match="ordinary preparation failure"):
        resumed.run()
    assert len(farm.preparations) == 1 and not farm.submissions


def test_resume_rejects_changed_protocol_and_missing_completed_result(tmp_path, setup):
    settings, protocol, source = setup
    farm = WorkerFarm()
    run = TrainingRun(tmp_path / "identity", settings, protocol, source.data(), worker_factory=farm)
    run.run(stop_after_tasks=1)
    with pytest.raises(ValueError, match="身份发生变化"):
        TrainingRun(
            run.directory,
            replace(settings, panel_seed=settings.panel_seed + 1),
            protocol,
            source.data(),
            worker_factory=farm,
            resume=True,
        )
    with pytest.raises(ValueError, match="身份发生变化"):
        TrainingRun(
            run.directory,
            settings,
            replace(protocol, driver_version="changed"),
            source.data(),
            worker_factory=farm,
            resume=True,
        )
    for path in (run.directory / "tasks").glob("*.json"):
        if load_checkpoint(path)["kind"] == "solve":
            path.unlink()
    resumed = TrainingRun(
        run.directory, settings, protocol, source.data(), worker_factory=farm, resume=True
    )
    with pytest.raises(FileNotFoundError, match="已完成任务产物丢失"):
        resumed.run()
    assert len(farm.submissions) == 1


def test_shape_capacity_and_data_disjointness_are_required(tmp_path, setup):
    settings, protocol, source = setup
    with pytest.raises(ValueError, match="重复实例"):
        TrainingData(source, source.training, source.training, {"identity": "bad"})
    with pytest.raises(ValueError, match="完整固定形状"):
        TrainingRun(
            tmp_path / "shape",
            replace(settings, instances_per_panel=1),
            protocol,
            source.data(),
            worker_factory=WorkerFarm(),
        )
    with pytest.raises(ValueError, match="缓存不足"):
        TrainingRun(
            tmp_path / "capacity",
            settings,
            replace(protocol, maximum_registered_per_dimension=4),
            source.data(),
            worker_factory=WorkerFarm(),
        )
