"""非法原生返回保留在追加日志，不按质量重做。"""

import json

from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.checkpoint import load_checkpoint
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.factorial_training import FactorialTrainingRun
from test_factorial_training import FactorialFarm, settings_and_protocol
from test_training import Source


def test_nonfinite_native_return_is_saved_as_failure_without_retry(tmp_path):
    settings, protocol = settings_and_protocol()
    farm = FactorialFarm(invalid="nan")
    run = FactorialTrainingRun(
        tmp_path / "invalid",
        settings,
        protocol,
        Source().data(),
        factorial_policy=FactorialPolicy("M01", BaselinePolicy()),
        worker_factory=farm,
    )
    run.run(stop_after_tasks=5)
    state = load_checkpoint(run.path)
    assert (
        state["costs"]["solve_jobs"]
        == state["costs"]["failed_solves"]
        == len(farm.submissions)
        == 5
    )
    records = [
        json.loads(line) for line in (run.directory / "results.jsonl").read_text().splitlines()
    ]
    solves = [r for r in records if r["kind"] == "solve"]
    assert len(solves) == 5
    assert all(
        r["outcome"]["native_result"]["items"][0]["cost"] == {"invalid_float": "nan"}
        for r in solves
    )
