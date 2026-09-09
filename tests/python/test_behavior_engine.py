"""真实GPU公开记录接口、独立账目及多规模开关重放。"""

import os

import numpy as np
import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.behavior import summarize_behavior, validate_behavior
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.program_ir import Program
from test_factorial_engine import causal

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="需要已分配GPU"
)


@pytest.mark.parametrize("n", [31, 500, 1000])
@pytest.mark.parametrize("variant", ["M00", "M10", "M01", "M11"])
def test_counted_behavior_toggle_replay_and_complete_ledger(n, variant):
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(n, 2, settings)
    engine.register_problem(11, np.random.default_rng(835).random((n, 2)))
    keys, seeds = np.array([11, 11], dtype=np.uint64), np.array([17, 29], dtype=np.uint64)
    program = Program((0, 0, 2), (4, 5, 0), feature_spec_id=2).to_dict()
    policy = FactorialPolicy(
        variant, BaselinePolicy(region=2, restart_mode="bernoulli", restart_probability=0.5)
    )
    contract = dict(
        dimension=n,
        colonies=2,
        ants=4,
        evaluation_limit=48,
        ls_evaluation_limit=100000,
        policy=policy,
    )
    off = engine.evaluate_factorial_evaluations(keys, seeds, 48, program, policy.to_dict())
    on = engine.evaluate_factorial_evaluations(
        keys, seeds, 48, program, policy.to_dict(), record_behavior=True
    )
    repeat = engine.evaluate_factorial_evaluations(
        keys, seeds, 48, program, policy.to_dict(), "end_to_end", record_behavior=True
    )
    assert causal(off) == causal(on) == causal(repeat)
    assert "behavior" not in off
    assert on["behavior"] == repeat["behavior"]
    validate_behavior(on, **contract)
    summary = summarize_behavior(on, **contract)
    assert len(summary["solves"]) == 2
    assert sum(s["whole_solve"]["fe"] for s in summary["solves"]) == 96
    if variant in ("M00", "M11"):
        if variant == "M00":
            endpoint = engine.evaluate_baseline_evaluations(
                keys, seeds, 48, policy.baseline_policy.to_dict(), record_behavior=True
            )
        else:
            endpoint = engine.evaluate_program_evaluations(
                keys, seeds, 48, program, record_behavior=True
            )
        assert causal(on) == causal(endpoint) and on["behavior"] == endpoint["behavior"]
    zero = engine.evaluate_factorial_evaluations(
        keys, seeds, 0, program, policy.to_dict(), record_behavior=True
    )
    validate_behavior(zero, **(contract | {"evaluation_limit": 0}))
    for bad in (None, 1, "true"):
        with pytest.raises(ValueError, match="bool"):
            engine.evaluate_factorial_evaluations(
                keys, seeds, 48, program, policy.to_dict(), record_behavior=bad
            )
