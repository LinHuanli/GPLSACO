"""少量设备内决策采样保持完整次数预算和实际动作评分。"""

import os

import numpy as np
import pytest
from gp_faco.experiment_v2 import representative_program

native = pytest.importorskip("gp_faco_ext")
pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_RUN_GPU_TESTS") != "1"
    or getattr(native, "campaign_interface_version", 0) != 1,
    reason="需要空闲 A5000 和 campaign 构建",
)


def test_sparse_decisions_match_gpu_score_and_preserve_search():
    engine = native.FacoBatchEngine(67, 2, native.FixedFacoSettings())
    engine.register_problem(19, np.random.default_rng(31).random((67, 2)))
    keys, seeds = np.array([19, 19], dtype=np.uint64), np.array([17, 29], dtype=np.uint64)
    program = representative_program().to_dict()
    ordinary = engine.evaluate_program_evaluations(keys, seeds, 64 * 20, program)
    traced = engine.evaluate_program_evaluations(
        keys, seeds, 64 * 20, program, decision_iterations=[1, 9, 20]
    )
    for name in ("control_states", "total_tour_evaluations", "completed_ls_evaluations"):
        assert traced[name] == ordinary[name]
    assert [r["tour"] for r in traced["items"]] == [r["tour"] for r in ordinary["items"]]
    assert len(traced["decisions"]) == 6
    for row in traced["decisions"]:
        scored = native.score_cuda(
            program,
            np.array(row["features"], dtype=np.float32).reshape(12, 1, 32),
            np.array([row["legal_mask"]], dtype=np.uint32),
        )
        assert int(scored["actions"][0]) == row["action"]
        assert [float(v) if np.isfinite(v) else None for v in scored["scores"][0]] == row["scores"]
        assert row["legal_mask"] & (1 << row["action"])
    for points in ([0], [21], [9, 1], [1, 1]):
        with pytest.raises(ValueError):
            engine.evaluate_program_evaluations(
                keys, seeds, 64 * 20, program, decision_iterations=points
            )
