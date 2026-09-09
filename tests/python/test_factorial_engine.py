"""真实原生四格接口、端点等价、共同次数与任务切换隔离。"""

import os

import numpy as np
import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.data import Instance, tour_cost
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.program_ir import Program

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="需要已分配GPU"
)


def causal(result):
    return {
        "items": [(v["tour"], v["cost"]) for v in result["items"]],
        **{
            name: result[name]
            for name in (
                "control_states",
                "completed_batches",
                "total_tour_evaluations",
                "completed_construction_steps",
                "completed_ls_evaluations",
            )
        },
    }


@pytest.mark.parametrize("n", [31, 500, 1000])
def test_native_four_variants_routes_and_original_endpoints(n):
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(n, 2, settings)
    xy = np.random.default_rng(835).random((n, 2))
    instance = Instance("development-synthetic", tuple(map(tuple, xy.tolist())))
    engine.register_problem(11, xy)
    keys, seeds = np.array([11, 11], dtype=np.uint64), np.array([17, 29], dtype=np.uint64)
    program = Program((0, 0, 2, 0, 0, 3, 0, 4, 2), (4, 8, 0, 0, 2, 0, 5, 0, 0), feature_spec_id=2)
    bases = [
        BaselinePolicy(region=2, restart_mode="bernoulli", restart_probability=0.5),
        BaselinePolicy(
            kind="rule",
            max_mne_level=3,
            region=1,
            stagnation_step=2,
            restart_stagnation=3,
            restart_cooldown=3,
        ),
    ]
    for base in bases:
        for variant in ("M00", "M10", "M01", "M11"):
            policy = FactorialPolicy(variant, base)
            result = engine.evaluate_factorial_evaluations(
                keys, seeds, 48, program.to_dict(), policy.to_dict()
            )
            assert result["factorial_policy"] == policy.to_dict()
            assert result["completed_batches"] == result["launched_batches"] == 12
            assert result["total_tour_evaluations"] == 96
            assert result["budget_seconds"] is None
            assert result["charged_seconds"] == result["overrun_seconds"] == 0
            assert result["discarded_batches"] == 0
            assert "control_trace" not in result
            for item in result["items"]:
                assert tour_cost(instance, item["tour"]) == pytest.approx(item["cost"], abs=1e-12)
            if variant == "M00":
                original = engine.evaluate_baseline_evaluations(keys, seeds, 48, base.to_dict())
                assert causal(result) == causal(original)
            if variant == "M11":
                original = engine.evaluate_program_evaluations(keys, seeds, 48, program.to_dict())
                assert causal(result) == causal(original)
            engine.evaluate_program_evaluations(keys, seeds, 16, program.to_dict())
            repeat = engine.evaluate_factorial_evaluations(
                keys, seeds, 48, program.to_dict(), policy.to_dict(), "end_to_end"
            )
            assert causal(result) == causal(repeat)


def test_binding_rejects_wrong_factor_or_budget_and_allows_valid_masks():
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(5, 1, settings)
    engine.register_problem(11, np.random.default_rng(74).random((5, 2)))
    keys, seeds = np.array([11], dtype=np.uint64), np.array([17], dtype=np.uint64)
    program = Program((0,), (4,), feature_spec_id=2)
    policy = FactorialPolicy("M01", BaselinePolicy(region=2)).to_dict()
    for updates in ({"factorial_spec_id": True}, {"variant": "Full"}, {"extra": 0}):
        with pytest.raises(ValueError):
            engine.evaluate_factorial_evaluations(
                keys, seeds, 4, program.to_dict(), policy | updates
            )
    for limit in (True, -1, 1.0, 5, 1 << 200):
        with pytest.raises(ValueError):
            engine.evaluate_factorial_evaluations(keys, seeds, limit, program.to_dict(), policy)
    for variant in ("M00", "M01", "M10", "M11"):
        p = policy | {"variant": variant}
        with pytest.raises(ValueError):
            engine.evaluate_factorial_evaluations(keys, seeds, 4, Program((0,), (4,)).to_dict(), p)
        if variant in ("M00", "M01"):
            with pytest.raises(ValueError, match="mask"):
                engine.evaluate_factorial_evaluations(
                    keys, seeds, 4, program.to_dict(), p, "cached", 1
                )
        else:
            result = engine.evaluate_factorial_evaluations(
                keys, seeds, 4, program.to_dict(), p, "cached", 1 | (1 << 31)
            )
            assert result["total_tour_evaluations"] == 4
