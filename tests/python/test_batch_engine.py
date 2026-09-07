"""并发预算入口的结果、类型和缺失incumbent边界。"""

import os

import numpy as np
import pytest
from gp_faco.data import Instance, tour_cost

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="仅在已分配GPU上运行"
)


@pytest.fixture
def panel():
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(31, 3, settings)
    points = {
        11: np.random.default_rng(41).random((31, 2)),
        19: np.random.default_rng(73).random((31, 2)),
    }
    fees = {key: engine.register_problem(key, xy) for key, xy in points.items()}
    keys = np.array([11, 11, 19], dtype=np.uint64)
    seeds = np.array([17, 29, 17], dtype=np.uint64)
    return engine, points, fees, keys, seeds


@pytest.mark.parametrize("mode", ["cached_charged", "end_to_end"])
def test_completed_results_and_budget_accounting(panel, mode):
    engine, points, fees, keys, seeds = panel
    result = engine.evaluate(keys, seeds, 0.12, 8, mode)
    assert len(result["items"]) == 3
    for key, item in zip(keys, result["items"], strict=True):
        assert item["has_incumbent"] and item["completed_seconds"] <= 0.12
        instance = Instance(str(key), tuple(map(tuple, points[int(key)])))
        assert tour_cost(instance, item["tour"]) == pytest.approx(item["cost"], abs=1e-10)
    assert result["launched_batches"] == result["completed_batches"] + result["discarded_batches"]
    assert result["discarded_batches"] <= 1
    assert result["elapsed_seconds"] == result["actual_seconds"] + result["charged_seconds"]
    expected_fee = (
        sum(sum(fee.values()) for fee in fees.values()) if mode == "cached_charged" else 0
    )
    assert result["charged_seconds"] == pytest.approx(expected_fee, abs=1e-12)
    assert "discarded_costs" not in result


def test_zero_budget_returns_explicit_missing_incumbents(panel):
    engine, _, _, keys, seeds = panel
    engine.evaluate(keys, seeds, 0.06, 8)
    result = engine.evaluate(keys, seeds, 0, 8)
    assert result["launched_batches"] == 0
    assert not result["preparation_completed"]
    assert all(
        not item["has_incumbent"] and item["cost"] is None and item["tour"] == []
        for item in result["items"]
    )


@pytest.mark.parametrize("kind", ["dtype", "shape", "unknown_key", "boolean", "nan", "mode"])
def test_invalid_tasks_fail_before_search(panel, kind):
    engine, _, _, keys, seeds = panel
    budget, mne, mode = 0.1, 8, "cached_charged"
    if kind == "dtype":
        keys = keys.astype(np.int64)
    elif kind == "shape":
        seeds = seeds[:2]
    elif kind == "unknown_key":
        keys = np.array([11, 11, 1234], dtype=np.uint64)
    elif kind == "boolean":
        mne = True
    elif kind == "nan":
        budget = float("nan")
    else:
        mode = "free_preparation"
    with pytest.raises((TypeError, ValueError)):
        engine.evaluate(keys, seeds, budget, mne, mode)


def test_registered_identity_is_immutable(panel):
    engine, points, fees, _, _ = panel
    assert engine.register_problem(11, points[11]) == fees[11]
    with pytest.raises(ValueError):
        engine.register_problem(11, points[11] + 0.1)
