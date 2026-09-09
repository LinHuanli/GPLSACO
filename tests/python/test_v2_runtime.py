"""次数预算、完整种群重评和中断恢复；假 worker 不用于算法质量结论。"""

import json
import platform
import random
from collections import Counter
from concurrent.futures import Future
from dataclasses import replace

import pytest
from gp_faco.data import Instance, Label, tour_cost
from gp_faco.evolution import EvolutionSettings
from gp_faco.result_journal import ResultJournal
from gp_faco.training import TrainingData, TrainingRun, TrainingSettings
from gp_faco.worker import SolverSettings, WorkerProtocol, faco_ants, preparation_id


class Source:
    def __init__(self):
        self.instances = {}
        training, validation = {}, {}
        rng = random.Random(19)
        for n in (5, 7):
            names = []
            for i in range(8):
                name = f"n{n}-{i}"
                self.instances[name] = Instance(
                    name,
                    tuple((rng.random(), rng.random()) for _ in range(n)),
                    numeric_id=n * 100 + i + 1,
                )
                names.append(name)
            training[n], validation[n] = names[:4], names[4:]
        self.data = TrainingData(self, training, validation, {"data_id": "toy-runtime-v2"})

    def load_instance(self, name):
        return self.instances[name]

    def load_label(self, name):
        p = self.instances[name]
        tour = tuple(range(p.dimension))
        return Label(tour, tour_cost(p, tour))


class Farm:
    def __init__(self):
        self.calls = []

    def __call__(self, protocol):
        farm = self

        class Worker:
            def ready(self, timeout=None):
                return {
                    "pid": 90000001,
                    "host": platform.node(),
                    "protocol_id": protocol.identifier,
                }

            def prepare(self, problems):
                future = Future()
                future.set_result(
                    {
                        "status": "completed",
                        "protocol_id": protocol.identifier,
                        "dimension": problems[0].dimension,
                        "problems": [p.instance_id for p in problems],
                        "preparation_id": preparation_id(problems),
                        "worker_seconds": 0,
                        "registration_fees": {
                            p.instance_id: {"cheap_seconds": 0, "preparation_seconds": 0}
                            for p in problems
                        },
                    }
                )
                return future

            def submit(self, task):
                farm.calls.append(task.occurrence_id)
                problems = {p.instance_id: p for p in task.problems}
                items = []
                for name, seed in task.replicas:
                    p = problems[name]
                    tour = list(range(p.dimension))
                    random.Random(seed).shuffle(tour)
                    items.append(
                        {
                            "has_incumbent": True,
                            "tour": tour,
                            "cost": tour_cost(p, tour),
                            "completed_seconds": 0.01,
                        }
                    )
                limit = task.evaluation_limit_per_colony
                batches = limit // protocol.settings.ants_for(task.dimension)
                future = Future()
                future.set_result(
                    {
                        "status": "completed",
                        "task_id": task.occurrence_id,
                        "protocol_id": protocol.identifier,
                        **task.controller_identity(),
                        "occurrence_id": task.occurrence_id,
                        "dimension": task.dimension,
                        "worker_seconds": 0.02,
                        "native_result": {
                            "items": items,
                            "budget_seconds": None,
                            "preparation_mode": "cached",
                            "preparation_completed": True,
                            "actual_seconds": 0.02,
                            "elapsed_seconds": 0.02,
                            "charged_seconds": 0,
                            "overrun_seconds": 0,
                            "launched_batches": batches,
                            "completed_batches": batches,
                            "discarded_batches": 0,
                            "budget_kind": "search_tour_evaluations",
                            "evaluation_limit_per_colony": limit,
                            "completed_tour_evaluations_per_colony": limit,
                            "total_tour_evaluations": limit * protocol.colonies,
                        },
                    }
                )
                return future

            def close(self):
                pass

        return Worker()


def setup():
    settings = TrainingSettings(
        evolution=EvolutionSettings(population=6, generations=3, elites=2, feature_spec_id=2),
        instances_per_panel=2,
        budgets=((5, 64), (7, 128)),
        preparation_mode="cached",
        budget_kind="search_tour_evaluations",
    )
    protocol = WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "NVIDIA RTX A5000",
        "test",
        dimensions=(5, 7),
        colonies=4,
        settings=SolverSettings(),
        maximum_registered_per_dimension=16,
    )
    return settings, protocol, Source().data


def test_paper_ant_boundaries_and_scale_budget():
    assert [faco_ants(n) for n in (100, 256, 257, 500, 1000, 1024, 1025, 10000)] == [
        64,
        64,
        128,
        128,
        128,
        128,
        192,
        448,
    ]
    assert SolverSettings().ants_for(10000) * 5000 == 2240000
    with pytest.raises(ValueError):
        faco_ants(True)


def test_only_a5000():
    _, protocol, _ = setup()
    with pytest.raises(ValueError, match="A5000"):
        replace(protocol, gpu_model="NVIDIA RTX PRO 5000 Blackwell")


def test_resume_does_not_repeat_completed_solves(tmp_path):
    settings, protocol, data = setup()
    farm = Farm()
    path = tmp_path / "v2-training"
    assert (
        TrainingRun(path, settings, protocol, data, worker_factory=farm).run(stop_after_tasks=5)[
            "status"
        ]
        == "paused"
    )
    result = TrainingRun(path, settings, protocol, data, worker_factory=farm, resume=True).run()
    assert result["status"] == "complete"
    assert all(n == 1 for n in Counter(farm.calls).values())
    assert len([name for name in farm.calls if ":generation" in name]) == 6 * 3 * 2
    assert result["costs"]["failed_solves"] == 0
    assert len(result["training_panels"]) == 3
    assert not (path / "tasks").exists()


def test_crash_after_result_append_before_snapshot(tmp_path, monkeypatch):
    settings, protocol, data = setup()
    farm = Farm()
    original = TrainingRun._record_completed
    crashed = False

    def interrupt(self, record, offset):
        nonlocal crashed
        if record["kind"] == "solve" and not crashed:
            crashed = True
            raise RuntimeError("simulated crash after append")
        return original(self, record, offset)

    path = tmp_path / "v2-crash"
    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "_record_completed", interrupt)
        with pytest.raises(RuntimeError, match="simulated crash"):
            TrainingRun(path, settings, protocol, data, worker_factory=farm).run()
    result = TrainingRun(path, settings, protocol, data, worker_factory=farm, resume=True).run()
    assert result["status"] == "complete"
    assert all(n == 1 for n in Counter(farm.calls).values())
    assert result["costs"]["solve_jobs"] == len(farm.calls)


def test_partial_journal_tail_is_preserved_and_appendable(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"task": 1}) + '\n{"task":')
    journal = ResultJournal(path)
    assert [r for _, r in journal.trailing(0)] == [{"task": 1}]
    assert path.with_suffix(".partial").read_text() == '{"task":'
    offset = journal.append({"task": 2})
    assert journal.read(offset) == {"task": 2}
    journal.close()
