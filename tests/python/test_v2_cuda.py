"""真正 GPU 上检查曲线检查点、完整计数和无侵入的生产 event 测量。"""

import os

import numpy as np
import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.data import Instance, tour_cost
from gp_faco.experiment_v2 import representative_program

native = pytest.importorskip("gp_faco_ext")
pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_RUN_GPU_TESTS") != "1", reason="需要空闲 A5000"
)


def test_faco_curve_checkpoints_are_exact_prefixes_and_keep_full_budget():
    xy = np.random.default_rng(607).random((67, 2))
    engine = native.FacoBatchEngine(67, 4, native.FixedFacoSettings())
    engine.register_problem(1, xy)
    keys, seeds = np.ones(4, dtype=np.uint64), np.array([17, 29, 41, 53], dtype=np.uint64)
    full = engine.evaluate_faco_evaluations(keys, seeds, 64 * 3, checkpoint_iterations=[0, 1, 3])
    assert full["total_tour_evaluations"] == 64 * 3 * 4
    problem = Instance("fixture", tuple(map(tuple, xy)), numeric_id=1)
    previous = [float("inf")] * 4
    for point in full["checkpoints"]:
        single = engine.evaluate_faco_evaluations(keys, seeds, 64 * point["iterations"])
        assert point["tours"] == [row["tour"] for row in single["items"]]
        costs = [tour_cost(problem, tour) for tour in point["tours"]]
        assert all(a <= b + 1e-12 for a, b in zip(costs, previous, strict=True))
        previous = costs
    assert full["checkpoints"][-1]["tours"] == [row["tour"] for row in full["items"]]
    for points in ([1, 1], [3, 1], [4]):
        with pytest.raises(ValueError):
            engine.evaluate_faco_evaluations(keys, seeds, 64 * 3, checkpoint_iterations=points)


@pytest.mark.parametrize("kind", ["program", "baseline"])
def test_production_event_measurement_keeps_tours_and_feedback(kind):
    engine = native.FacoBatchEngine(67, 4, native.FixedFacoSettings())
    engine.register_problem(1, np.random.default_rng(19).random((67, 2)))
    keys, seeds = np.ones(4, dtype=np.uint64), np.array([17, 29, 41, 53], dtype=np.uint64)
    controller = (
        representative_program().to_dict() if kind == "program" else BaselinePolicy().to_dict()
    )
    method = getattr(engine, f"evaluate_{kind}_evaluations")
    plain = method(keys, seeds, 64 * 3, controller)
    measured = method(keys, seeds, 64 * 3, controller, profile_events_only=True)
    assert [(r["tour"], r["cost"]) for r in plain["items"]] == [
        (r["tour"], r["cost"]) for r in measured["items"]
    ]
    assert plain["control_states"] == measured["control_states"]
    assert sum(measured["production_kernel_milliseconds"].values()) > 0
