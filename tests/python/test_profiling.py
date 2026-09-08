"""诊断计时与主fitness接口边界；两种准备模式下保持相同的实际求解结果。"""

import math
import os

import numpy as np
import pytest
from gp_faco.program_ir import Program

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="仅在已分配GPU上运行"
)


@pytest.mark.parametrize("mode", ["cached_charged", "end_to_end"])
def test_profile_phase_accounting_and_production_isolation(mode):
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(31, 2, settings)
    measured = engine.register_problem(21, np.random.default_rng(831).random((31, 2)))
    preparation = engine.preparation_profile(21)
    assert all(math.isfinite(v) and v >= 0 for v in preparation.values())
    assert math.fsum(preparation.values()) == pytest.approx(
        measured["preparation_seconds"], abs=1e-12
    )
    engine.set_preparation_charges(21, 0.001, 0.01)
    keys, seeds = np.array([21, 21], dtype=np.uint64), np.array([17, 29], dtype=np.uint64)
    program = Program((0, 0, 2), (4, 8, 0)).to_dict()
    results = [
        engine.run_program_diagnostic(keys, seeds, program, 4, 0.25, enabled, preparation_mode=mode)
        for enabled in (False, True, False)
    ]
    for result in results:
        assert result["completed_batches"] == 4 and result["discarded_batches"] == 0
        assert result["control_states"] == results[0]["control_states"]
        assert [(v["tour"], v["cost"]) for v in result["items"]] == [
            (v["tour"], v["cost"]) for v in results[0]["items"]
        ]
        assert result["charged_seconds"] == pytest.approx(0.011 if mode == "cached_charged" else 0)
    assert not results[0]["profile"]["batches"] and not results[2]["profile"]["batches"]
    profile = results[1]["profile"]
    assert profile["diagnostic_device_bytes"] == 8 * 4 * 8
    assert profile["initialization_includes_uploads"]
    for entry in profile["batches"]:
        assert entry["committed"] and entry["wall_seconds"] > 0
        assert len(entry["ant_phase_cycles"]) == 8
        assert all(v > 0 for row in entry["ant_phase_cycles"] for v in row)
        assert entry["gpu_milliseconds"]["construction_and_ls"] > 0
    production = engine.evaluate_program(keys, seeds, 0.05, program)
    assert "profile" not in production and "scope" not in production
    with pytest.raises(ValueError, match="profile必须固定"):
        engine.run_program_diagnostic(keys, seeds, program, 4, -1)
