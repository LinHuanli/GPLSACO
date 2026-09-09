"""评价次数的IR/DEAP/任务身份/外部fitness契约，及真实GPU计数入口。"""

import copy
import json
import os
from dataclasses import replace

import numpy as np
import pytest
from deap import gp
from gp_faco.data import Instance, Label, tour_cost
from gp_faco.evolution import Evolution, EvolutionSettings, individual_from_program
from gp_faco.fitness import score_panel
from gp_faco.primitives import make_primitive_set
from gp_faco.program_ir import Program, export_tree
from gp_faco.training import TrainingSettings
from gp_faco.worker import SolverSettings, SolveTask, WorkerProtocol


def test_progress_is_versioned_and_survives_variation_and_restore():
    grammar = make_primitive_set(feature_spec_id=2)
    assert "progress" in grammar.arguments and "elapsed" not in grammar.arguments
    tree = gp.PrimitiveTree.from_string("ADD(progress,restart)", grammar)
    with pytest.raises(ValueError):
        export_tree(tree)
    program = export_tree(tree, feature_spec_id=2)
    assert program.feature_spec_id == 2
    assert program.key != replace(program, feature_spec_id=1).key
    assert export_tree(individual_from_program(program, grammar)) == program
    with pytest.raises(ValueError, match="特征版本"):
        individual_from_program(program, make_primitive_set())
    run = Evolution(
        EvolutionSettings(population=6, generations=3, elites=1, feature_spec_id=2), 130
    )
    run.initialize()
    restored = Evolution.from_state_dict(json.loads(json.dumps(run.state_dict())))
    for generation in range(3):
        for evolution in (run, restored):
            evolution.begin_panel(f"panel-{generation}")
            for index, individual in enumerate(evolution.population):
                ir = export_tree(individual)
                assert ir.feature_spec_id == 2
                evolution.assign(
                    index, float(sum(ir.operand)), f"panel-{generation}", ir.identifier
                )
            evolution.finish_generation()
        assert run.state_dict() == restored.state_dict()
        assert run.advance() == restored.advance() == (generation < 2)
    assert len(run.winners) == 3
    assert all(v.feature_spec_id == 2 for v in run.shortlist())
    with pytest.raises(ValueError, match="特征版本"):
        TrainingSettings(budget_kind="search_tour_evaluations")


def counted_fixture():
    problem = Instance("square", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    protocol = WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "NVIDIA RTX A5000",
        "0",
        "test-build",
        dimensions=(4,),
        colonies=1,
        settings=SolverSettings(ants=32),
    )
    task = SolveTask(
        "counted",
        Program((0,), (0,), feature_spec_id=2),
        (problem,),
        (("square", 17),),
        preparation_mode="cached",
        evaluation_limit_per_colony=64,
    )
    label = Label((0, 1, 2, 3), 4.0)
    outcome = {
        "task_id": task.task_id(protocol),
        "protocol_id": protocol.identifier,
        "program_id": task.program.identifier,
        "dimension": 4,
        "occurrence_id": "counted",
        "status": "completed",
        "native_result": {
            "budget_kind": "search_tour_evaluations",
            "budget_seconds": None,
            "evaluation_limit_per_colony": 64,
            "completed_tour_evaluations_per_colony": 64,
            "total_tour_evaluations": 64,
            "preparation_mode": "cached",
            "preparation_completed": True,
            "actual_seconds": 1e9,
            "elapsed_seconds": 1e9,
            "charged_seconds": 0.0,
            "overrun_seconds": 0.0,
            "completed_batches": 2,
            "launched_batches": 2,
            "discarded_batches": 0,
            "items": [
                {
                    "has_incumbent": True,
                    "tour": list(label.tour),
                    "cost": 4.0,
                    "completed_seconds": 1e9,
                }
            ],
        },
    }
    return task, protocol, outcome, {"square": label}


def test_counts_identity_and_fitness_have_no_hidden_time_cutoff():
    task, protocol, outcome, labels = counted_fixture()
    assert not score_panel(task, protocol, outcome, labels).failed
    assert replace(task, evaluation_limit_per_colony=96).manifest(protocol) != task.manifest(
        protocol
    )
    with pytest.raises(ValueError, match="时间上限"):
        replace(task, budget_seconds=100.0)
    with pytest.raises(ValueError, match="整批"):
        replace(task, evaluation_limit_per_colony=65).manifest(protocol)
    for field, bad in (
        ("completed_tour_evaluations_per_colony", 63),
        ("total_tour_evaluations", 65),
        ("evaluation_limit_per_colony", 32),
        ("completed_tour_evaluations_per_colony", True),
        ("discarded_batches", 1),
        ("budget_seconds", 1e10),
    ):
        changed = copy.deepcopy(outcome)
        changed["native_result"][field] = bad
        assert score_panel(task, protocol, changed, labels).failed


@pytest.mark.skipif(os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="需要已分配GPU")
@pytest.mark.parametrize("mode", ["cached", "end_to_end"])
def test_native_exact_counts_and_old_version_rejection(mode):
    import gp_faco_ext as native

    settings = native.FixedFacoSettings()
    settings.ants = 4
    engine = native.FacoBatchEngine(31, 2, settings)
    xy = np.random.default_rng(801).random((31, 2))
    problem = Instance("p", tuple(map(tuple, xy.tolist())))
    engine.register_problem(11, xy)
    engine.set_preparation_charges(11, 1e9, 1e9)
    keys, seeds = np.array([11, 11], dtype=np.uint64), np.array([17, 29], dtype=np.uint64)
    program = Program((0, 0, 2), (0, 4, 0), feature_spec_id=2)
    results = [
        engine.evaluate_program_evaluations(keys, seeds, 24, program.to_dict(), mode)
        for _ in range(2)
    ]
    for result in results:
        assert result["budget_seconds"] is None and result["charged_seconds"] == 0
        assert (
            result["evaluation_limit_per_colony"]
            == result["completed_tour_evaluations_per_colony"]
            == 24
        )
        assert result["total_tour_evaluations"] == 48
        assert result["completed_batches"] == result["launched_batches"] == 6
        assert result["discarded_batches"] == result["overrun_seconds"] == 0
        assert "profile" not in result
        for item in result["items"]:
            assert item["has_incumbent"] and tour_cost(problem, item["tour"]) == pytest.approx(
                item["cost"], abs=1e-12
            )
        assert result["control_states"] == results[0]["control_states"]
        assert [(v["tour"], v["cost"]) for v in result["items"]] == [
            (v["tour"], v["cost"]) for v in results[0]["items"]
        ]
    zero = engine.evaluate_program_evaluations(keys, seeds, 0, program.to_dict(), mode)
    assert zero["completed_batches"] == zero["total_tour_evaluations"] == 0
    assert all(v["has_incumbent"] for v in zero["items"])
    for invalid in (True, 1.0, -1, 25, 1 << 200):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            engine.evaluate_program_evaluations(keys, seeds, invalid, program.to_dict(), mode)
    with pytest.raises(ValueError, match="特征版本"):
        engine.evaluate_program(keys, seeds, 0.1, program.to_dict())
    with pytest.raises(ValueError, match="特征版本"):
        engine.evaluate_program_evaluations(
            keys, seeds, 24, replace(program, feature_spec_id=1).to_dict(), mode
        )
