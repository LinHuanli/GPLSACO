"""繁殖退化、条件动作语义与恢复的一组针对性回归。"""

import json

import numpy as np
import pytest
from deap import gp
from gp_faco.controller import ROLE_FEATURES, Controller, DecisionContexts, score_controller
from gp_faco.diverse_evolution import Candidate, DiverseEvolution, DiversitySettings
from gp_faco.evolution import Individual, isolated_deap_rng
from gp_faco.program_ir import Program, export_tree


@pytest.fixture
def contexts():
    rng = np.random.default_rng(12)
    f = rng.random((12, 64, 32), dtype=np.float32)
    f[:4] = f[:4, :, :1]
    f[4] = np.arange(32) // 16
    f[5] = (np.arange(32) % 4 + 1) / 4
    for feature in (6, 7):
        f[feature] = np.repeat(f[feature, :, [0, 16]].T, 16, axis=1)
    for feature in range(8, 12):
        f[feature] = np.repeat(f[feature, :, ::4], 4, axis=1)
    return DecisionContexts(f, np.full(64, 0xFFFFFFFF, dtype=np.uint32))


def leaf(feature):
    return Program((0,), (feature,), feature_spec_id=2)


def test_conditional_legality_and_ties(contexts):
    constant = Program((1,), (0,), (0,), feature_spec_id=2)
    c = Controller("conditional_three", (constant, constant, leaf(5)))
    masks = np.full(64, (1 << 2) | (1 << 11) | (1 << 31), dtype=np.uint32)
    result = score_controller(c, contexts.features, masks)
    assert np.all(result["actions"] == 2)
    assert [s["scores"].shape[1] for s in result["stages"]] == [2, 4, 4]
    masks[:] = 1 << 31
    assert np.all(score_controller(c, contexts.features, masks)["actions"] == 31)
    assert "scores" not in result


def test_role_and_total_node_limits():
    with pytest.raises(ValueError, match="角色树"):
        Controller("conditional_three", (leaf(5), leaf(8), leaf(5)))
    with pytest.raises(ValueError, match="角色树"):
        Controller("conditional_three", (leaf(4), leaf(5), leaf(5)))
    deep = Program((0,) * 6 + (2,) * 5, (0,) * 11, feature_spec_id=2)
    assert Controller("conditional_three", (deep, deep, deep))

    def binary(depth):
        return (0,) if depth == 0 else binary(depth - 1) + binary(depth - 1) + (2,)

    big = Program(binary(5), (0,) * 63, feature_spec_id=2)
    with pytest.raises(ValueError, match="总节点"):
        Controller("conditional_three", (big, big, big))


@pytest.mark.parametrize("representation", ["joint_single", "conditional_three"])
def test_unique_population_height_strata_and_resume(contexts, representation):
    e = DiverseEvolution(representation, 1103, contexts)
    e.initialize()
    assert [p.trees[0].height for p in e.population] == [2 + i % 3 for i in range(128)]
    assert all(sum(map(len, p.trees)) <= 63 for p in e.population)
    if representation == "conditional_three":
        assert all(
            arg in ROLE_FEATURES[role]
            for p in e.population
            for role, t in enumerate(p.trees)
            for op, arg in zip(export_tree(t).opcode, export_tree(t).operand, strict=True)
            if op == 0
        )
    e.assign([1.0] * 128)
    restored = DiverseEvolution.from_state_dict(json.loads(json.dumps(e.state_dict())), contexts)
    e.advance()
    restored.advance()
    assert [p.controller().key for p in e.population] == [
        p.controller().key for p in restored.population
    ]
    assert all(
        a.controller().key != b.controller().key
        for i, a in enumerate(e.population)
        for b in e.population[:i]
    )
    assert e.metrics()["max_behavior_group"] <= 2
    assert e.counts["crossover_structure_changed"] > 0
    assert e.counts["crossover_behavior_changed"] > 0


def test_root_crossover_and_leaf_growth(contexts):
    e = DiverseEvolution("joint_single", 1, contexts)
    left = Candidate(
        [Individual(gp.PrimitiveTree.from_string("mne_level", e.psets[0]), 2)], "joint_single"
    )
    right = Candidate(
        [Individual(gp.PrimitiveTree.from_string("restart", e.psets[0]), 2)], "joint_single"
    )
    with isolated_deap_rng(e.rng):
        assert e.crossover(left, right).controller().key == right.controller().key
        assert e.crossover(left, left).controller().key == left.controller().key
        for _ in range(20):
            assert 1 <= e.mutate(left).trees[0].height <= 3
    assert e._valid(left)  # 进化得到的单叶仍合法，不以强制深度伪造复杂性。


def test_parent_ties_do_not_prefer_small_tree(contexts):
    e = DiverseEvolution("joint_single", 1, contexts, DiversitySettings(tournament=128))
    for expression in ("mne_level", "ADD(mne_level, mne_level)"):
        p = Candidate(
            [Individual(gp.PrimitiveTree.from_string(expression, e.psets[0]), 2)], "joint_single"
        )
        p.fitness = 1.0
        e.population.append(p)
    with isolated_deap_rng(e.rng):
        counts = [0, 0]
        for _ in range(1000):
            counts[int(len(e.tournament().trees[0]) > 1)] += 1
    assert min(counts) > 400


def test_capacity_failure_is_bounded(contexts):
    f = np.zeros_like(contexts.features)
    e = DiverseEvolution("joint_single", 2, DecisionContexts(f, contexts.masks))
    with pytest.raises(RuntimeError, match="不放宽约束"):
        e.initialize()
    assert len(e.population) == 2


def test_pause_preserves_published_work(tmp_path):
    from gp_faco.representation_pilot import PilotPaused, collect_evaluation

    (tmp_path / "pause.json").write_text("{}")
    with pytest.raises(PilotPaused):
        collect_evaluation(tmp_path, ["example"], [])


def test_external_fitness_uses_route_not_accumulated_search_cost(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from gp_faco import representation_pilot as pilot
    from gp_faco.data import Instance

    problem = Instance("square", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

    @contextmanager
    def dataset():
        yield SimpleNamespace(
            load_instance=lambda _: problem, load_label=lambda _: SimpleNamespace(cost=4.0)
        )

    monkeypatch.setattr(pilot, "source", dataset)
    native = {
        "total_tour_evaluations": 128,
        "items": [{"tour": [0, 1, 2, 3], "has_incumbent": True, "cost": 4.00000000000001}],
    }
    session = SimpleNamespace(
        set_colonies=lambda _: None, device={}, solve=lambda *_: (native, {"solve_seconds": 0.1})
    )
    result = pilot.solve_gpu_job(
        tmp_path,
        {
            "kind": "baseline_gpu",
            "pairs": [("square", 17)],
            "iterations": 1,
            "method": pilot.FIXED16,
        },
        session,
    )
    assert result["rows"][0]["cost"] == 4.0
    assert result["rows"][0]["gap_percent"] == 0.0
