"""Hard公共绑定的身份、共同初解、非法输入与注册事务。"""

import copy
import os

import numpy as np
import pytest
from gp_faco.data import Instance
from gp_faco.graph_matching import GraphSettings, engine_graph_spec, match_graphs
from gp_faco.program_ir import Program
from gp_faco.worker import coordinate_hash

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
            "coordinate_sha256": coordinate_hash(problem),
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


def test_graph_public_binding_rejects_ambiguous_nodes_and_failed_registration_is_atomic():
    native, xy, settings, common, cached = fixture()
    engine = native.FacoBatchEngine(7, 1, settings, "hard")
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
    assert zero["constraint_mode"] == "hard"
    assert zero["graph_edges_per_colony"] == [len(graph["edges"])]
    assert zero["total_tour_evaluations"] == 0 and zero["budget_seconds"] is None
    with pytest.raises(ValueError):
        engine.evaluate_program(keys, seeds, 1.0, Program((0,), (4,)).to_dict())


def test_graph_cache_integrity_and_unsupported_modes():
    native, xy, settings, _, graph = fixture()
    changed = copy.deepcopy(graph)
    changed["backup"][0][0] = 7
    with pytest.raises(ValueError):
        engine_graph_spec(changed)
    with pytest.raises(ValueError):
        native.FacoBatchEngine(7, 1, settings, "escape")
    engine = native.FacoBatchEngine(7, 1, settings)
    with pytest.raises(ValueError):
        engine.register_graph_problem(11, xy, engine_graph_spec(graph))
    for bad in (xy.astype(np.float32), xy.reshape(-1), np.full((7, 2), np.nan)):
        with pytest.raises((ValueError, TypeError)):
            native.prepare_common_initial(bad, settings)
