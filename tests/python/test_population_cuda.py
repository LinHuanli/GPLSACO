"""种群展开与单程序生产路径对照：程序/seed/面板切换、FIFO 回绕、FE 和反馈。"""

import os

import numpy as np
import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.evolution import Evolution, EvolutionSettings
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.program_ir import export_tree

native = pytest.importorskip("gp_faco_ext")
pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_RUN_GPU_TESTS") != "1", reason="需要空闲 A5000"
)


def programs():
    evolution = Evolution(EvolutionSettings(population=8, elites=2, feature_spec_id=2), 1103)
    evolution.initialize()
    result = [export_tree(individual).to_dict() for individual in evolution.population]
    result[-1] = result[0]  # 不去重，重复个体占用自己的列和 FE。
    return result


@pytest.mark.parametrize("factorial", [None, "M10", "M01"])
def test_population_matches_every_serial_member_and_reuses_geometry(factorial):
    n, replicas, iterations = 67, 4, 40
    settings = native.FixedFacoSettings()
    values = programs()
    serial = native.FacoBatchEngine(n, replicas, settings)
    expanded = native.FacoBatchEngine(n, replicas, settings, population_size=len(values))
    for key in (1, 2, 3):
        xy = np.random.default_rng(107 + key).random((n, 2))
        for engine in (serial, expanded):
            engine.register_problem(key, xy)
    policy = FactorialPolicy(factorial, BaselinePolicy()).to_dict() if factorial else None
    for panel, panel_seeds in (([1, 1, 2, 2], [17, 29, 17, 29]), ([3, 2, 3, 1], [41, 29, 53, 17])):
        keys = np.asarray(panel, dtype=np.uint64)
        seeds = np.asarray(panel_seeds, dtype=np.uint64)
        results = expanded.evaluate_population_evaluations(
            keys, seeds, iterations * 64, values, factorial_policy=policy
        )
        assert len(results) == len(values)
        for program, actual in zip(values, results, strict=True):
            expected = (
                serial.evaluate_factorial_evaluations(keys, seeds, iterations * 64, program, policy)
                if policy
                else serial.evaluate_program_evaluations(keys, seeds, iterations * 64, program)
            )
            assert [(r["tour"], r["cost"]) for r in actual["items"]] == [
                (r["tour"], r["cost"]) for r in expected["items"]
            ]
            assert actual["control_states"] == expected["control_states"]
            for field in (
                "completed_construction_steps",
                "completed_ls_evaluations",
                "completed_tour_evaluations_per_colony",
                "total_tour_evaluations",
            ):
                assert actual[field] == expected[field], field
            assert actual["actual_seconds"] * len(values) == actual["population_actual_seconds"]
            assert all(r["completed_seconds"] <= actual["actual_seconds"] for r in actual["items"])
        assert results[0]["items"] == results[-1]["items"]
        # 同一预分配引擎收缩为验证尾块，再展开；位置变化和空闲列都不得改变随机轨迹。
        for start, stop in ((3, 8), (0, 1), (0, 3), (0, 8)):
            chunk = expanded.evaluate_population_evaluations(
                keys, seeds, iterations * 64, values[start:stop], factorial_policy=policy
            )
            assert len(chunk) == stop - start
            for expected, actual in zip(results[start:stop], chunk, strict=True):
                assert [(r["tour"], r["cost"]) for r in actual["items"]] == [
                    (r["tour"], r["cost"]) for r in expected["items"]
                ]
                for field in (
                    "control_states",
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                    "total_tour_evaluations",
                    "allocated_device_bytes",
                ):
                    assert actual[field] == expected[field], field


def test_population_shape_rejection_and_single_member():
    n = 67
    settings = native.FixedFacoSettings()
    for count in (0, 129):
        with pytest.raises(ValueError):
            native.FacoBatchEngine(n, 4, settings, population_size=count)
    with pytest.raises(ValueError):
        native.FacoBatchEngine(n, 4, settings, "hard", population_size=8)
    engine = native.FacoBatchEngine(n, 4, settings)
    engine.register_problem(1, np.random.default_rng(19).random((n, 2)))
    keys = np.ones(4, dtype=np.uint64)
    seeds = np.arange(4, dtype=np.uint64)
    program = programs()[0]
    result = engine.evaluate_population_evaluations(keys, seeds, 64, [program])[0]
    expected = engine.evaluate_program_evaluations(keys, seeds, 64, program)
    assert result["control_states"] == expected["control_states"]
    assert [r["tour"] for r in result["items"]] == [r["tour"] for r in expected["items"]]
    with pytest.raises(ValueError):
        engine.evaluate_population_evaluations(keys, seeds, 64, [])
