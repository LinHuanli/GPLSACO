"""以手工可枚举的图验证实际边预算、图与槽位分离及无标签匹配。"""

import copy
import math
import random

import pytest
from gp_faco.data import Instance
from gp_faco.graph_matching import GraphSettings, match_graphs


def fixture():
    p = Instance(
        "seven",
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.5), (3.0, 1.0), (2.0, 2.0), (0.0, 2.0)),
        numeric_id=1,
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
            "instance_id": p.instance_id,
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
    assert all(changed[k]["settings"] != result[k]["settings"] for k in result)


def test_matching_rejects_wrong_identity_duplicate_nodes_and_initial_tour():
    p, priors = fixture()
    for kind in ("ALPHA", "POPMUSIC"):
        wrong = copy.deepcopy(priors)
        wrong[kind]["instance_id"] = "foreign"
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


def test_actual_slots_match_per_node_with_unequal_degrees_and_prior_tails():
    n = 15
    problem = Instance("unequal", tuple((float(i * i), float(i % 4)) for i in range(n)))
    rng, priors = random.Random(919), {}
    for kind in ("ALPHA", "POPMUSIC"):
        rows = []
        for i in range(n):
            values = rng.sample([j for j in range(n) if j != i], n - 1)
            if kind == "POPMUSIC":
                values = values[: 2 + i % 8]
            rows.append([{"to": j} for j in values])
        priors[kind] = {
            "prior_spec_id": 1,
            "dimension": n,
            "instance_id": problem.instance_id,
            "settings": {"kind": kind},
            "rows": rows,
        }
    settings = GraphSettings(primary_width=4, backup_width=6, ls_width=3, uniform_backup_slots=2)
    result = match_graphs(problem, tuple(range(n)), priors, settings)
    a, b = result["ALPHA"], result["POPMUSIC"]
    assert len(a["edges"]) == len(b["edges"])
    assert a["degrees"] != b["degrees"]
    assert a["matched_actual_slots"] == b["matched_actual_slots"]
    assert any(v == 0 for v in a["matched_actual_slots"]["native_backup"])
    assert any(v > 0 for v in a["matched_actual_slots"]["native_backup"])
    for i in range(n):
        primary_count = min(4, a["degrees"][i], b["degrees"][i])
        ls_count = min(3, a["degrees"][i], b["degrees"][i])
        actual_tail = []
        for kind, graph in result.items():
            main = [j for j in graph["primary"][i] if j != n]
            actual_tail.append(
                [e["to"] for e in priors[kind]["rows"][i][4:] if e["to"] not in {i, *main}]
            )
        tail_count = min(4, *(len(v) for v in actual_tail))
        for graph in result.values():
            assert graph["matching_spec_id"] == 2
            assert sum(j != n for j in graph["primary"][i]) == primary_count
            assert sum(j != n for j in graph["ls"][i]) == ls_count
            assert sum(j != n for j in graph["backup"][i]) == tail_count + 2
            assert len(set(graph["backup"][i]) - {n}) == tail_count + 2
            assert not (set(graph["primary"][i]) & set(graph["backup"][i]) - {n})
            neighbors = [y if x == i else x for x, y in graph["edges"] if i in (x, y)]

            def distance(j, i=i):
                dx = problem.coordinates[i][0] - problem.coordinates[j][0]
                dy = problem.coordinates[i][1] - problem.coordinates[j][1]
                return math.sqrt(dx * dx + dy * dy), j

            assert graph["ls"][i][:ls_count] == sorted(neighbors, key=distance)[:ls_count]
    assert match_graphs(problem, tuple(range(n)), priors, settings) == result
