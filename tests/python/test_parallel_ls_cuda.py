"""优化warp路径与串行前缀诊断路径在LS边界预算下应产生相同搜索结果。"""

import os

import numpy as np
import pytest
from gp_faco.experiment_v2 import representative_program

native = pytest.importorskip("gp_faco_ext")
pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_RUN_GPU_TESTS") != "1", reason="需要空闲A5000"
)


@pytest.mark.parametrize("limit", [1, 2, 19, 20, 21, 39, 40, 41, 1000])
def test_parallel_prefix_preserves_budget_stop_and_ties(limit):
    settings = native.FixedFacoSettings()
    settings.ls_evaluation_limit = limit
    engine = native.FacoBatchEngine(67, 2, settings)
    # 包含等距离和重合坐标，覆盖零增益、padding以及并列情况。
    xy = np.random.default_rng(420).integers(0, 12, (67, 2)).astype(np.float64)
    engine.register_problem(312, xy)
    keys = np.asarray([312, 312], dtype=np.uint64)
    seeds = np.asarray([17, 29], dtype=np.uint64)
    program = representative_program().to_dict()
    fast = engine.evaluate_program_evaluations(keys, seeds, settings.ants * 12, program)
    serial = engine.evaluate_program_evaluations(
        keys, seeds, settings.ants * 12, program, profile=True
    )
    for field in (
        "control_states",
        "completed_construction_steps",
        "completed_ls_evaluations",
        "total_tour_evaluations",
    ):
        assert fast[field] == serial[field]
    for a, b in zip(fast["items"], serial["items"], strict=True):
        for field in ("has_incumbent", "tour", "cost"):
            assert a[field] == b[field]
