"""基线搜索返回后的崩溃窗口和独立验证预算。"""

from collections import Counter
from dataclasses import replace

import pytest
from gp_faco.configuration_search import ConfigurationRun
from test_configuration_search import Source, WorkerFarm
from test_configuration_search import setup as search_setup


@pytest.fixture
def configuration_setup():
    return search_setup.__wrapped__()


def test_append_then_crash_recovers_same_jobs_with_longer_validation(
    configuration_setup, tmp_path, monkeypatch
):
    settings, protocol, policies = configuration_setup
    settings = replace(settings, validation_evaluation_limit=32)
    farm = WorkerFarm()
    path = tmp_path / "v2-search"
    run = ConfigurationRun(path, settings, protocol, policies, Source().data(), worker_factory=farm)
    original = run._record_completed
    once = False

    def fail_after_append(record, offset):
        nonlocal once
        if record["kind"] == "solve" and not once:
            once = True
            raise RuntimeError("injected crash after appended result")
        return original(record, offset)

    monkeypatch.setattr(run, "_record_completed", fail_after_append)
    with pytest.raises(RuntimeError, match="injected"):
        run.run()
    resumed = ConfigurationRun(
        path, settings, protocol, policies, Source().data(), worker_factory=farm, resume=True
    )
    result = resumed.run()
    assert result["status"] == "complete"
    assert set(Counter(t.occurrence_id for t in farm.submissions).values()) == {1}
    assert all(
        t.evaluation_limit_per_colony == 32
        for t in farm.submissions
        if ":validation:" in t.occurrence_id
    )
    assert all(
        t.evaluation_limit_per_colony == 16
        for t in farm.submissions
        if ":search:" in t.occurrence_id
    )
