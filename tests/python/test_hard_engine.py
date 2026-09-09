"""图约束公共绑定的身份、共同初解、非法输入与注册事务。"""

import copy
import os

import numpy as np
import pytest
from gp_faco.data import Instance
from gp_faco.graph_matching import GraphSettings, engine_graph_spec, match_graphs
from gp_faco.program_ir import Program

pytestmark = pytest.mark.skipif(
    os.environ.get("GP_FACO_REQUIRE_CUDA") != "1", reason="需要已分配GPU"
)


def fixture():
    import gp_faco_ext as native

    xy = np.random.default_rng(691).random((7, 2))
    problem = Instance("seven", tuple(map(tuple, xy.tolist())))
    settings = native.FixedFacoSettings()
    settings.ants, settings.primary_width, settings.backup_width, settings.ls_width = 4, 2, 3, 4
    common = native.prepare_common_initial(xy, settings)
    priors = {
        kind: {
            "prior_spec_id": 1,
            "dimension": 7,
            "instance_id": problem.instance_id,
            "settings": {"kind": kind},
            "rows": [[{"to": (i + 1) % 7}] for i in range(7)],
        }
        for kind in ("ALPHA", "POPMUSIC")
    }
    graph = match_graphs(
        problem,
        tuple(common["tour"]),
        priors,
        GraphSettings(primary_width=2, backup_width=3, ls_width=4, uniform_backup_slots=2),
    )["ALPHA"]
    return native, xy, settings, common, graph


@pytest.mark.parametrize("constraint_mode", ["hard", "escape"])
def test_graph_public_binding_rejects_ambiguous_nodes_and_failed_registration_is_atomic(
    constraint_mode,
):
    native, xy, settings, common, cached = fixture()
    engine = native.FacoBatchEngine(7, 1, settings, constraint_mode)
    graph = engine_graph_spec(cached)
    wrong = []
    for field, value in (
        ("graph_spec_id", True),
        ("graph_spec_id", 2),
        ("common_initial_tour", common["tour"][1:] + common["tour"][:1]),
        ("edges", graph["edges"] + [graph["edges"][0]]),
        ("edges", [[True, 1]]),
        ("edges", [[0, 1 << 80]]),
        ("primary", [[False, 1]] * 7),
        ("primary", [[7, 1]] * 7),
        ("unused", 0),
    ):
        wrong.append({**graph, field: value})
    wrong.append({k: v for k, v in graph.items() if k != "ls"})
    for invalid in wrong:
        with pytest.raises(ValueError):
            engine.register_graph_problem(11, xy, invalid)
    for key in (True, -1, 1.0, 1 << 80):
        with pytest.raises(ValueError):
            engine.register_graph_problem(key, xy, graph)
    with pytest.raises(ValueError):
        engine.register_problem(11, xy)
    engine.register_graph_problem(11, xy, graph)
    engine.register_graph_problem(11, xy, copy.deepcopy(graph))
    for changed_xy, changed_graph in ((xy + 0.01, graph), (xy, wrong[2])):
        with pytest.raises(ValueError):
            engine.register_graph_problem(11, changed_xy, changed_graph)
    keys, seeds = np.array([11], np.uint64), np.array([17], np.uint64)
    program = Program((0,), (4,), feature_spec_id=2).to_dict()
    zero = engine.evaluate_program_evaluations(keys, seeds, 0, program)
    assert zero["items"][0]["tour"] == common["tour"]
    assert zero["items"][0]["cost"] == common["cost"]
    assert zero["constraint_mode"] == constraint_mode
    assert zero["graph_edges_per_colony"] == [len(graph["edges"])]
    assert zero["total_tour_evaluations"] == 0 and zero["budget_seconds"] is None
    with pytest.raises(ValueError):
        engine.evaluate_program(keys, seeds, 1.0, Program((0,), (4,)).to_dict())


def test_invalid_graph_nodes_and_unsupported_modes():
    native, xy, settings, _, graph = fixture()
    changed = copy.deepcopy(graph)
    changed["backup"][0][0] = 8
    hard = native.FacoBatchEngine(7, 1, settings, "hard")
    with pytest.raises(ValueError):
        hard.register_graph_problem(11, xy, engine_graph_spec(changed))
    with pytest.raises(ValueError):
        native.FacoBatchEngine(7, 1, settings, "escape_v0")
    engine = native.FacoBatchEngine(7, 1, settings)
    with pytest.raises(ValueError):
        engine.register_graph_problem(11, xy, engine_graph_spec(graph))
    for bad in (xy.astype(np.float32), xy.reshape(-1), np.full((7, 2), np.nan)):
        with pytest.raises((ValueError, TypeError)):
            native.prepare_common_initial(bad, settings)


def test_escape_public_counts_resources_and_gp_baseline_reuse():
    from gp_faco.baseline_policy import BaselinePolicy
    from gp_faco.data import tour_cost

    native, xy, settings, common, graph = fixture()
    engine = native.FacoBatchEngine(7, 2, settings, "escape")
    hard = native.FacoBatchEngine(7, 2, settings, "hard")
    for target in (engine, hard):
        target.register_graph_problem(11, xy, engine_graph_spec(graph))
    keys, seeds = np.array([11, 11], np.uint64), np.array([17, 29], np.uint64)
    policy = BaselinePolicy(
        mne_level=3, max_mne_level=3, restart_mode="bernoulli", restart_probability=0.5
    ).to_dict()
    result = engine.evaluate_baseline_evaluations(keys, seeds, 64, policy)
    assert result["constraint_mode"] == "escape" and result["escape_spec_id"] == 1
    assert result["escape_edge_capacity_per_ant"] == 64
    assert result["total_tour_evaluations"] == 128 and result["completed_batches"] == 16
    assert result["budget_seconds"] is None
    assert (
        result["charged_seconds"] == result["discarded_batches"] == result["overrun_seconds"] == 0
    )
    counters = result["escape_counters"]
    assert all(type(value) is int and value >= 0 for value in counters.values())
    assert counters["new_edges"] <= 64 * 128
    assert counters["construction_gates"] <= counters["construction_opportunities"]
    assert (
        result["completed_construction_steps"]
        <= counters["construction_opportunities"]
        <= result["completed_construction_steps"] + 128
    )
    assert counters["ls_replaced_slots"] <= 5 * counters["ls_anchor_nodes"]
    program = Program((0,), (4,), feature_spec_id=2).to_dict()
    engine.evaluate_program_evaluations(keys, seeds, 8, program)
    repeated = engine.evaluate_baseline_evaluations(keys, seeds, 64, policy, "end_to_end")
    assert (
        repeated["escape_counters"] == counters
        and repeated["control_states"] == result["control_states"]
    )
    problem = Instance("seven", tuple(map(tuple, xy.tolist())))
    for a, b in zip(result["items"], repeated["items"], strict=True):
        assert a["tour"] == b["tour"] and a["cost"] == b["cost"]
        assert tour_cost(problem, a["tour"]) == pytest.approx(a["cost"], abs=1e-12)
    zero = engine.evaluate_program_evaluations(keys, seeds, 0, program)
    assert not any(zero["escape_counters"].values())
    assert all(item["tour"] == common["tour"] for item in zero["items"])
    reference = hard.evaluate_program_evaluations(keys, seeds, 0, program)
    assert result["allocated_device_bytes"] == reference["allocated_device_bytes"]
    assert result["reserved_escape_device_bytes"] == reference["reserved_escape_device_bytes"] > 0
    settings.primary_width = 17
    with pytest.raises(ValueError):
        native.FacoBatchEngine(31, 1, settings, "escape")
