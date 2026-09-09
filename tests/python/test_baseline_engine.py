"""真实原生Static/Rule次数接口、规模覆盖、配置身份和非法参数拒绝。"""

import os

import numpy as np
import pytest
from gp_faco.data import Instance, tour_cost
from gp_faco.program_ir import Program

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="需要已分配GPU"
)


def policy(**updates):
    return {
        "policy_spec_id": 1,
        "kind": "static",
        "mne_level": 0,
        "max_mne_level": 0,
        "region": 0,
        "restart_mode": "none",
        "restart_period": 0,
        "restart_probability": 0.0,
        "stagnation_step": 0,
        "restart_stagnation": 0,
        "restart_cooldown": 0,
        **updates,
    }


@pytest.mark.parametrize("n", [31, 500, 1000])
def test_native_baselines_counts_routes_and_state_isolation(n):
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(n, 2, settings)
    xy = np.random.default_rng(490).random((n, 2))
    problem = Instance("p", tuple(map(tuple, xy.tolist())))
    engine.register_problem(11, xy)
    keys, seeds = np.array([11, 11], dtype=np.uint64), np.array([17, 29], dtype=np.uint64)
    configs = [
        policy(),
        policy(restart_mode="periodic", restart_period=3),
        policy(restart_mode="bernoulli", restart_probability=0.5),
        policy(
            kind="rule",
            max_mne_level=3,
            stagnation_step=2,
            restart_stagnation=3,
            restart_cooldown=3,
        ),
    ]
    for config in configs:
        result = engine.evaluate_baseline_evaluations(keys, seeds, 24, config)
        assert result["baseline_policy"] == config
        assert result["completed_batches"] == result["launched_batches"] == 6
        assert result["completed_tour_evaluations_per_colony"] == 24
        assert result["total_tour_evaluations"] == 48
        assert result["budget_seconds"] is None
        assert result["charged_seconds"] == result["overrun_seconds"] == 0
        assert "profile" not in result and "baseline_uniforms" not in result
        # 中间真实GP任务可覆盖动作/评分缓冲，随后基线仍须完全重置自身状态。
        engine.evaluate_program_evaluations(
            keys, seeds, 8, Program((0,), (4,), feature_spec_id=2).to_dict()
        )
        repeat = engine.evaluate_baseline_evaluations(keys, seeds, 24, config, "end_to_end")
        assert result["control_states"] == repeat["control_states"]
        for left, right in zip(result["items"], repeat["items"], strict=True):
            assert left["tour"] == right["tour"] and left["cost"] == right["cost"]
            assert tour_cost(problem, left["tour"]) == pytest.approx(left["cost"], abs=1e-12)


def test_baseline_binding_rejects_ambiguous_or_incompatible_configs():
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 32  # 此接口错误测试明确使用 32 FE 的工程批次。
    engine = native.FacoBatchEngine(5, 1, settings)
    engine.register_problem(11, np.random.default_rng(70).random((5, 2)))
    keys, seeds = np.array([11], dtype=np.uint64), np.array([17], dtype=np.uint64)
    for changed in (
        {"policy_spec_id": True},
        {"mne_level": -1},
        {"region": 4},
        {"region": 1 << 80},
        {"restart_probability": True},
        {"restart_probability": float("nan")},
        {"restart_mode": "periodic"},
        {"restart_mode": "unknown"},
        {"kind": "rule", "restart_stagnation": 4},
        {"stagnation_step": 1},
        {"kind": "gp"},
        {"unused": 0},
    ):
        with pytest.raises(ValueError):
            engine.evaluate_baseline_evaluations(keys, seeds, 32, policy(**changed))
    for invalid in (True, -1, 1.0, 31, 1 << 200):
        with pytest.raises(ValueError):
            engine.evaluate_baseline_evaluations(keys, seeds, invalid, policy())
    with pytest.raises(ValueError, match="mask"):
        engine.evaluate_baseline_evaluations(keys, seeds, 32, policy(), "cached", 2)
    with pytest.raises(ValueError):
        engine.evaluate_baseline_evaluations(keys, seeds, 32, policy(), "cached_charged")
