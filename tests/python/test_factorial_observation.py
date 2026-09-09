"""资源争用/未知观察及非法原生返回必须保留，不能触发质量相关重做。"""

import importlib.util
import json
import sys
from pathlib import Path

from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.checkpoint import load_checkpoint
from gp_faco.factorial_policy import FactorialPolicy
from test_factorial_training import FactorialFarm, settings_and_protocol
from test_training import Source

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "e2_observation", SCRIPTS / "train_factorial_development.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_busy_wait_and_unknown_post_preserve_all_original_returns(tmp_path, monkeypatch):
    settings, protocol = settings_and_protocol()
    farm = FactorialFarm()
    run = MODULE.ObservedFactorialRun(
        tmp_path / "busy",
        settings,
        protocol,
        Source().data(),
        factorial_policy=FactorialPolicy("M10", BaselinePolicy()),
        worker_factory=farm,
    )
    observations = iter(
        (
            {"foreign_processes": 1},
            {"foreign_processes": 0},
            {"foreign_processes": None, "error": "test unknown query"},
        )
    )
    monkeypatch.setattr(run, "_boundary", lambda: next(observations, {"foreign_processes": 0}))
    sleeps = []
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)
    run.run()
    state = load_checkpoint(run.path)
    assert sleeps == [5]
    assert state["costs"]["failed_solves"] == 0
    assert state["costs"]["solve_jobs"] == len(farm.submissions)
    assert len({t.occurrence_id for t in farm.submissions}) == len(farm.submissions)
    raws = list((run.directory / "raw-returns").glob("*.json"))
    assert len(raws) == len(state["completed"])
    unknown = 0
    for path in raws:
        raw = load_checkpoint(path)
        outcome = load_checkpoint(run.directory / "tasks" / path.name)["outcome"]
        unknown += outcome["gpu_boundary_after"]["foreign_processes"] is None
        assert raw == {k: v for k, v in outcome.items() if k != "gpu_boundary_after"}
    assert unknown == 1
    assert (
        json.loads((run.directory / "resource-waits.jsonl").read_text())["foreign_processes"] == 1
    )


def test_nonfinite_native_return_is_saved_as_failure_without_retry(tmp_path, monkeypatch):
    settings, protocol = settings_and_protocol()
    farm = FactorialFarm(invalid="nan")
    run = MODULE.ObservedFactorialRun(
        tmp_path / "invalid",
        settings,
        protocol,
        Source().data(),
        factorial_policy=FactorialPolicy("M01", BaselinePolicy()),
        worker_factory=farm,
    )
    monkeypatch.setattr(run, "_boundary", lambda: {"foreign_processes": 0})
    run.run(stop_after_tasks=5)
    state = load_checkpoint(run.path)
    assert (
        state["costs"]["solve_jobs"]
        == state["costs"]["failed_solves"]
        == len(farm.submissions)
        == 5
    )
    solves = []
    for path in (run.directory / "raw-returns").glob("*.json"):
        raw = load_checkpoint(path)
        if "native_result" in raw:
            solves.append(raw)
            assert raw["native_result"]["items"][0]["cost"] == {"invalid_float": "nan"}
    assert len(solves) == 5
