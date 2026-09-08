"""以手工可枚举的图验证实际边预算、图与槽位分离及无标签匹配。"""

import copy

import pytest
from gp_faco.data import Instance
from gp_faco.graph_matching import GraphSettings, match_graphs
from gp_faco.worker import coordinate_hash


def fixture():
    p = Instance(
        "seven",
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.5), (3.0, 1.0), (2.0, 2.0), (0.0, 2.0)),
    )
    priors = {}
    for kind in ("ALPHA", "POPMUSIC"):
        rows = []
        for i in range(7):
            if kind == "ALPHA":
                ids = [(i + 2) % 7, (i + 1) % 7]
                ids += [j for j in range(7) if j != i and j not in ids]
            else:
                ids = [(i - 1) % 7, (i + 1) % 7]
                if i == 0:
                    ids = [2, 1]
                if i == 2:
                    ids = [0, 1]
            rows.append([{"to": j} for j in ids])
        priors[kind] = {
            "prior_spec_id": 1,
            "dimension": 7,
            "coordinate_sha256": coordinate_hash(p),
            "settings": {"kind": kind},
            "rows": rows,
        }
    return p, priors


def test_exact_graph_budget_manual_priority_and_fixed_slots():
    p, priors = fixture()
    settings = GraphSettings(primary_width=2, backup_width=2, ls_width=3, uniform_backup_slots=1)
    result = match_graphs(p, tuple(range(7)), priors, settings)
    expected = {tuple(sorted((i, (i + 1) % 7))) for i in range(7)} | {(0, 2)}
    for kind, graph in result.items():
        assert set(map(tuple, graph["edges"])) == expected
        assert graph["extra_edge_budget"] == 1
        assert graph["discarded_extra_edges"] == (6 if kind == "ALPHA" else 0)
        assert graph["primary"][0] == [2, 1]
        assert graph["ls"][0] == [1, 2, 6]  # 图边0–6不在宽度2的主行，但不能从E0丢弃。
        assert graph["degrees"][0] == 3
        assert sum(graph["degrees"]) == 2 * len(expected)
        for i in range(7):
            assert len(graph["primary"][i]) == len(graph["backup"][i]) == 2
            assert len(graph["ls"][i]) == 3
            assert i not in graph["primary"][i] and i not in graph["backup"][i]
            assert not (set(graph["primary"][i]) & set(graph["backup"][i]) - {7})
    assert match_graphs(p, tuple(range(7)), priors, settings) == result
    changed = match_graphs(
        p,
        tuple(range(7)),
        priors,
        GraphSettings(
            primary_width=2,
            backup_width=2,
            ls_width=3,
            uniform_backup_slots=1,
            preparation_seed=811,
        ),
    )
    assert all(changed[k]["edges"] == result[k]["edges"] for k in result)
    assert all(changed[k]["sha256"] != result[k]["sha256"] for k in result)


def test_matching_rejects_wrong_identity_duplicate_nodes_and_initial_tour():
    p, priors = fixture()
    for kind in ("ALPHA", "POPMUSIC"):
        wrong = copy.deepcopy(priors)
        wrong[kind]["coordinate_sha256"] = "0" * 64
        with pytest.raises(ValueError):
            match_graphs(p, tuple(range(7)), wrong)
        wrong = copy.deepcopy(priors)
        wrong[kind]["rows"][0].append(wrong[kind]["rows"][0][0])
        with pytest.raises(ValueError):
            match_graphs(p, tuple(range(7)), wrong)
    with pytest.raises(ValueError):
        match_graphs(p, (0, 0, 2, 3, 4, 5, 6), priors)
    with pytest.raises(ValueError):
        match_graphs(p, tuple(range(7)), {"ALPHA": priors["ALPHA"]})
