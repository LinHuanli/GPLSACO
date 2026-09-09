"""代际屏障、精英重评、DEAP算子与JSON恢复的独立可复核测试。"""

import copy
import json
import random
from dataclasses import replace

import pytest
from deap import gp
from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.evolution import (
    Evolution,
    EvolutionSettings,
    Individual,
    individual_from_program,
    isolated_deap_rng,
)
from gp_faco.primitives import make_primitive_set
from gp_faco.program_ir import Program, export_tree


def assign_all(evolution, panel, reverse=False):
    indices = list(range(len(evolution.population)))
    if reverse:
        indices.reverse()
    for index in indices:
        individual = evolution.population[index]
        if not individual.fitness.valid:
            evolution.assign(index, float(index), panel, export_tree(individual).identifier)


def serialized(evolution):
    return json.loads(json.dumps(evolution.state_dict(), allow_nan=False))


def test_all_elites_and_duplicates_reevaluated_and_final_population_evaluated():
    e = Evolution(
        EvolutionSettings(
            population=8, generations=3, elites=2, crossover_probability=0, mutation_probability=0
        ),
        31,
    )
    e.initialize()
    # 故意让所有树完全相同，仍必须提供每个实际位置的评价。
    e.population = [copy.deepcopy(e.population[0]) for _ in e.population]
    e.begin_panel("panel0")
    with pytest.raises(ValueError):
        e.finish_generation()
    assign_all(e, "panel0")
    e.finish_generation()
    assert e.advance()
    assert all(not v.fitness.valid and v.evaluated_panel is None for v in e.population)
    e.begin_panel("panel1")
    with pytest.raises(ValueError):
        e.assign(0, 0.0, "panel0", export_tree(e.population[0]).identifier)
    assign_all(e, "panel1", reverse=True)
    e.finish_generation()
    assert e.advance()
    e.begin_panel("panel2")
    assign_all(e, "panel2")
    e.finish_generation()
    before = serialized(e)
    assert not e.advance()
    assert serialized(e) == before
    assert len(e.shortlist()) == 1 and all(v.fitness.valid for v in e.population)


def test_crossover_and_mutation_are_independent_and_parents_are_cloned():
    settings = EvolutionSettings(
        population=8, generations=2, elites=2, crossover_probability=1, mutation_probability=1
    )
    e = Evolution(settings, 77)
    e.initialize()
    e.begin_panel("g0")
    assign_all(e, "g0")
    e.finish_generation()
    originals = list(e.population)
    snapshots = [(export_tree(v), v.fitness.values, v.evaluated_panel) for v in originals]
    e.advance()
    assert e.variation_counts["crossovers"] == 3
    assert e.variation_counts["mutations"] == 6
    assert [(export_tree(v), v.fitness.values, v.evaluated_panel) for v in originals] == snapshots
    assert not {id(v) for v in originals} & {id(v) for v in e.population}
    assert all(v.height <= 5 and len(v) <= 63 and not v.fitness.valid for v in e.population)


def test_joint_static_limit_uses_exact_parent_fallback():
    e = Evolution(
        EvolutionSettings(
            population=2,
            generations=2,
            elites=0,
            initial_depth_max=2,
            max_nodes=15,
            mutation_depth_min=5,
            mutation_depth_max=5,
        ),
        994,
    )
    e.initialize()
    parent = e.population[0]
    original = export_tree(parent)
    observed = 0
    with isolated_deap_rng(e.rng):
        for _ in range(32):
            before = e.variation_counts["fallbacks"]
            (child,) = e.toolbox.mutate(copy.deepcopy(parent))
            if e.variation_counts["fallbacks"] > before:
                observed += 1
                assert export_tree(child) == original
            assert child.height <= 5 and len(child) <= 15
    assert observed > 0


@pytest.mark.parametrize("no_feedback", [False, True])
def test_program_roundtrip_preserves_future_operators_and_rng(no_feedback):
    settings = EvolutionSettings(population=12, generations=3, no_feedback=no_feedback)
    global_state = random.getstate()
    e = Evolution(settings, 1103)
    e.initialize()
    assert random.getstate() == global_state
    e.begin_panel("panel0")
    e.assign(0, 0.5, "panel0", export_tree(e.population[0]).identifier)
    restored = Evolution.from_state_dict(serialized(e))
    assert serialized(restored) == serialized(e)
    for generation in range(3):
        panel = f"panel{generation}"
        for current, reverse in ((e, False), (restored, True)):
            if current.panel_id is None:
                current.begin_panel(panel)
            assign_all(current, panel, reverse)
            current.finish_generation()
            current.advance()
        assert serialized(e) == serialized(restored)
    assert random.getstate() == global_state
    assert e.shortlist() == restored.shortlist()
    if no_feedback:
        assert all(
            operand not in (1, 2, 3)
            for program in e.shortlist()
            for opcode, operand in zip(program.opcode, program.operand, strict=True)
            if opcode == 0
        )
        with pytest.raises(ValueError):
            individual_from_program(Program((0,), (2,)), e.pset)


def test_explicit_terminal_and_erc_json_reconstruction():
    pset = make_primitive_set()
    tree = Individual(gp.PrimitiveTree.from_string("SUB(MUL(return_rate, restart), -1.25)", pset))
    program = export_tree(tree)
    reconstructed = individual_from_program(Program.from_dict(program.to_dict()), pset)
    assert export_tree(reconstructed) == program and str(reconstructed) == str(tree)
    assert isinstance(reconstructed, Individual)


def test_checkpoint_atomicity_integrity_and_explicit_infinite_fitness(tmp_path, monkeypatch):
    e = Evolution(EvolutionSettings(population=4, generations=2, elites=1), 55)
    e.initialize()
    e.begin_panel("g0")
    e.assign(0, float("inf"), "g0", export_tree(e.population[0]).identifier)
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, e.state_dict())
    restored = Evolution.from_state_dict(load_checkpoint(path))
    assert restored.population[0].fitness.valid
    assert restored.population[0].fitness.values == (float("inf"),)
    before = path.read_bytes()
    with monkeypatch.context() as patch:

        def fail_replace(*args):
            raise OSError("injected before atomic rename")

        patch.setattr("gp_faco.checkpoint.os.replace", fail_replace)
        with pytest.raises(OSError):
            save_checkpoint(path, {"new": "uncommitted"})
    assert path.read_bytes() == before and not list(tmp_path.glob("*.partial"))


def test_checkpoint_rejects_stale_fitness_and_bad_grammar():
    e = Evolution(EvolutionSettings(population=4, generations=2, elites=1), 99)
    e.initialize()
    e.begin_panel("current")
    e.assign(0, 1.0, "current", export_tree(e.population[0]).identifier)
    stale = serialized(e)
    stale["population"][0]["evaluated_panel"] = "old"
    with pytest.raises(ValueError):
        Evolution.from_state_dict(stale)
    broken = serialized(e)
    broken["population"][0]["program"]["operand"][0] = 255
    with pytest.raises(ValueError):
        Evolution.from_state_dict(broken)
    with pytest.raises(ValueError):
        replace(e.settings, population=4, elites=4)
    assign_all(e, "current")
    e.finish_generation()
    broken = serialized(e)
    broken["winners"][0]["fitness"] = -1e9
    with pytest.raises(ValueError, match="当前冠军"):
        Evolution.from_state_dict(broken)


def test_checkpoint_json_key_conversion_preserves_structure_and_rejects_collisions(tmp_path):
    path = tmp_path / "keys.json"
    save_checkpoint(path, {"training": {500: ["a"], 1000: ["b"]}})
    assert load_checkpoint(path) == {"training": {"500": ["a"], "1000": ["b"]}}
    with pytest.raises(ValueError, match="JSON键转换后发生重复"):
        save_checkpoint(path, {"ambiguous": {1: "a", "1": "b"}})
    assert load_checkpoint(path) == {"training": {"500": ["a"], "1000": ["b"]}}
