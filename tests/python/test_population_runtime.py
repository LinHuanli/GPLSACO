"""全种群事务恢复和完整 FE 账目；模拟 worker 不提供性能结论。"""

import os
import platform
from collections import Counter
from concurrent.futures import Future
from dataclasses import replace

import pytest
from gp_faco.checkpoint import save_checkpoint
from gp_faco.result_journal import ResultJournal
from gp_faco.training import TrainingRun
from test_v2_runtime import Farm, setup


class PopulationFarm(Farm):
    def __init__(self):
        super().__init__()
        self.batches = []

    def __call__(self, protocol):
        worker = super().__call__(protocol)

        def submit_population(tasks):
            self.batches.append(tuple(t.occurrence_id for t in tasks))
            outcomes = [worker.submit(task).result() for task in tasks]
            for outcome in outcomes:
                outcome["worker_seconds"] /= len(tasks)
                for key in ("actual_seconds", "elapsed_seconds"):
                    outcome["native_result"][key] /= len(tasks)
                for row in outcome["native_result"]["items"]:
                    row["completed_seconds"] /= len(tasks)
            future = Future()
            future.set_result({"members": outcomes})
            return future

        worker.submit_population = submit_population
        return worker


@pytest.mark.parametrize("failure", ["pause", "after_append", "during_score"])
def test_whole_population_survives_interruption_without_repeating_members(
    tmp_path, monkeypatch, failure
):
    settings, protocol, data = setup()
    serial_farm = Farm()
    serial = TrainingRun(
        tmp_path / "serial", settings, protocol, data, worker_factory=serial_farm
    ).run()
    parallel = replace(settings, population_batch_size=6)
    farm = PopulationFarm()
    path = tmp_path / "parallel"
    original = TrainingRun._record_completed
    crashed = False

    def interrupt(self, record, offset):
        nonlocal crashed
        if record["kind"] == "population" and not crashed:
            crashed = True
            raise RuntimeError("interrupted after population append")
        return original(self, record, offset)

    import gp_faco.training as training

    score_original = training.score_panel
    scores = 0

    def score_interrupt(*args, **kwargs):
        nonlocal scores
        scores += 1
        if scores == 3:
            raise RuntimeError("interrupted during population scoring")
        return score_original(*args, **kwargs)

    if failure == "pause":
        result = TrainingRun(path, parallel, protocol, data, worker_factory=farm).run(
            stop_after_tasks=1
        )
        assert result["status"] == "paused"
        assert len(farm.calls) == 6  # 批次不能为了主机观察阈值丢弃尚未归集的成员。
    else:
        with monkeypatch.context() as patch:
            if failure == "after_append":
                patch.setattr(TrainingRun, "_record_completed", interrupt)
            else:
                patch.setattr(training, "score_panel", score_interrupt)
            with pytest.raises(RuntimeError, match="interrupted"):
                TrainingRun(path, parallel, protocol, data, worker_factory=farm).run()
    result = TrainingRun(path, parallel, protocol, data, worker_factory=farm, resume=True).run()
    assert result["status"] == "complete"
    assert all(value == 1 for value in Counter(farm.calls).values())
    assert result["selected"]["program_id"] == serial["selected"]["program_id"]
    assert result["selected"]["fitness"] == serial["selected"]["fitness"]
    assert result["variation_counts"] == serial["variation_counts"]
    assert result["training_panels"] == serial["training_panels"]
    for name in ("solve_jobs", "failed_solves", "search_tour_evaluations", "valid_members"):
        assert result["costs"][name] == serial["costs"][name]
    assert result["costs"]["native_actual_seconds"] < serial["costs"]["native_actual_seconds"]
    assert len([name for name in farm.calls if ":generation" in name]) == 6 * 3 * 2


def test_resume_does_not_launch_over_a_live_worker_before_first_generation_snapshot(tmp_path):
    settings, protocol, data = setup()
    settings = replace(settings, population_batch_size=6)
    farm = PopulationFarm()
    directory = tmp_path / "live-worker"
    TrainingRun(directory, settings, protocol, data, worker_factory=farm).run(stop_after_tasks=1)
    before = list(farm.calls)
    save_checkpoint(
        directory / "worker_runtime.json", {"active": {"pid": os.getpid(), "host": platform.node()}}
    )
    with pytest.raises(RuntimeError, match="上一 worker"):
        TrainingRun(directory, settings, protocol, data, worker_factory=farm, resume=True).run()
    assert farm.calls == before
    save_checkpoint(directory / "worker_runtime.json", {"active": None})
    result = TrainingRun(
        directory, settings, protocol, data, worker_factory=farm, resume=True
    ).run()
    assert result["status"] == "complete"
    assert all(value == 1 for value in Counter(farm.calls).values())


def test_journal_preserves_nonfinite_failure_without_rewriting_valid_tours(tmp_path):
    journal = ResultJournal(tmp_path / "results.jsonl")
    good = {"tour": list(range(1000)), "cost": 12.3}
    first = journal.append(good)
    second = journal.append({"tour": good["tour"], "cost": float("nan")})
    assert journal.read(first) == good
    assert journal.read(second) == {"tour": good["tour"], "cost": {"invalid_float": "nan"}}
    journal.close()
