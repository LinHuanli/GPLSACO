"""真实GP入口的类型边界、动作mask和按时状态输出；完整逐字段核对由C++诊断执行。"""

import os

import numpy as np
import pytest
from gp_faco.program_ir import Program

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="仅在已分配GPU上运行"
)


@pytest.fixture
def panel():
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(31, 2, settings)
    xy = np.zeros((31, 2), dtype=np.float64)
    xy[0, 0] = 1  # 所有排列成本为2，让不同结构确实进入档案并触发重启。
    engine.register_problem(41, xy)
    keys = np.array([41, 41], dtype=np.uint64)
    seeds = np.array([17, 29], dtype=np.uint64)
    return engine, keys, seeds, Program((0,), (4,)).to_dict()


@pytest.mark.parametrize("mode", ["cached_charged", "end_to_end"])
def test_real_program_restarts_and_completed_state(panel, mode):
    engine, keys, seeds, program = panel
    result = engine.evaluate_program(keys, seeds, 0.15, program, mode)
    assert result["completed_batches"] > 1
    assert result["completed_batches"] + result["discarded_batches"] == result["launched_batches"]
    for item, state in zip(result["items"], result["control_states"], strict=True):
        assert item["has_incumbent"] and item["cost"] == 2
        assert sorted(item["tour"]) == list(range(31))
        assert item["completed_seconds"] <= 0.15
        assert 2 <= state["archive_size"] <= 4
        assert state["restarts"] > 0 and state["epoch_batches"] == 1
        assert state["stagnant_batches"] == result["completed_batches"]
        assert 0 <= state["return_rate"] <= 1 and 0 <= state["ls_work"] <= 1
    assert "control_trace" not in result and "discarded_costs" not in result
    assert result["elapsed_seconds"] == result["actual_seconds"] + result["charged_seconds"]


def test_mask_and_zero_budget_do_not_reuse_previous_control(panel):
    engine, keys, seeds, program = panel
    engine.evaluate_program(keys, seeds, 0.06, program)
    for mask in (0xFFFF, 1 << 15):
        keep = engine.evaluate_program(keys, seeds, 0.06, program, experiment_mask=mask)
        assert all(s["restarts"] == 0 for s in keep["control_states"])
    zero = engine.evaluate_program(keys, seeds, 0, program)
    assert zero["launched_batches"] == 0 and not zero["preparation_completed"]
    assert all(s["archive_size"] == 0 and s["restarts"] == 0 for s in zero["control_states"])
    assert all(not item["has_incumbent"] and item["cost"] is None for item in zero["items"])


@pytest.mark.parametrize(
    "kind",
    [
        "dtype",
        "shape",
        "budget_bool",
        "mask_bool",
        "mask_empty",
        "restart_only",
        "mask_overflow",
        "ir",
        "terminal",
        "mode",
    ],
)
def test_invalid_program_tasks_fail_before_search(panel, kind):
    engine, keys, seeds, program = panel
    budget, mask, mode = 0.1, 0xFFFFFFFF, "cached_charged"
    if kind == "dtype":
        keys = keys.astype(np.int64)
    elif kind == "shape":
        seeds = seeds[:1]
    elif kind == "budget_bool":
        budget = True
    elif kind == "mask_bool":
        mask = True
    elif kind == "mask_empty":
        mask = 0
    elif kind == "restart_only":
        mask = 0xFFFF0000
    elif kind == "mask_overflow":
        mask = 1 << 32
    elif kind == "ir":
        program["numeric_spec_id"] = True
    elif kind == "terminal":
        program["operand"] = [12]
    else:
        mode = "uncharged_cache"
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        engine.evaluate_program(keys, seeds, budget, program, mode, mask)
