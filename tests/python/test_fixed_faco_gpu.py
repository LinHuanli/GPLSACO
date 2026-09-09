"""固定迭代入口的标签隔离、数据形状与跨调用状态检查。"""

import os

import numpy as np
import pytest
from gp_faco.data import Instance, tour_cost

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="仅在已分配GPU上运行"
)


def backend():
    import gp_faco_ext

    assert hasattr(gp_faco_ext, "FixedFacoGpu")
    return gp_faco_ext


def test_fixed_iterations_result_and_seed_reset():
    ext = backend()
    coordinates = np.random.default_rng(701).random((51, 2))
    settings = ext.FixedFacoSettings()
    settings.ants = 8
    engine = ext.FixedFacoGpu(coordinates, 775, settings)
    result = engine.run_iterations(17, 5, 8)
    engine.run_iterations(29, 2, 2)
    repeated = engine.run_iterations(17, 5, 8)
    instance = Instance("unlabelled_fixture", tuple(map(tuple, coordinates)))
    assert tour_cost(instance, result["tour"]) == pytest.approx(result["cost"], abs=1e-10)
    assert result["cost"] <= result["initial_cost"] + 1e-10
    for field in ("tour", "cost", "construction_steps", "ls_evaluations"):
        assert repeated[field] == result[field]
    assert result["preparation_seconds"] > 0 and result["solve_seconds"] > 0
    assert result["allocated_device_bytes"] > 0


@pytest.mark.parametrize("kind", ["float32", "noncontiguous", "nan", "bad_shape"])
def test_bad_coordinate_inputs_are_rejected(kind):
    ext = backend()
    coordinates = np.random.default_rng(4).random((12, 2))
    if kind == "float32":
        coordinates = coordinates.astype(np.float32)
    elif kind == "noncontiguous":
        coordinates = coordinates[::2]
    elif kind == "nan":
        coordinates[0, 0] = np.nan
    else:
        coordinates = coordinates.reshape(-1)
    with pytest.raises((TypeError, ValueError)):
        ext.FixedFacoGpu(coordinates, 1, ext.FixedFacoSettings())


def test_invalid_configuration_and_boolean_task_ids_are_rejected():
    ext = backend()
    coordinates = np.random.default_rng(9).random((9, 2))
    settings = ext.FixedFacoSettings()
    settings.ants = 4097
    with pytest.raises(ValueError):
        ext.FixedFacoGpu(coordinates, 11, settings)
    settings.ants = 4
    with pytest.raises(ValueError):
        ext.FixedFacoGpu(coordinates, True, settings)
    engine = ext.FixedFacoGpu(coordinates, 11, settings)
    for arguments in ((True, 1, 2), (17, True, 2), (17, 1, True), (17, 1, 0)):
        with pytest.raises(ValueError):
            engine.run_iterations(*arguments)
