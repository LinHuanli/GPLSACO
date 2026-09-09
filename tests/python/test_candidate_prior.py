"""先验成员与真实距离分离、外部工具实际重放及畸形输出拒绝。"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
from gp_faco.candidate_prior import PriorSettings, parse_candidates, prepare_candidates
from gp_faco.data import Instance
from gp_faco.worker import PROJECT


def triangle():
    problem = Instance("triangle", ((0.0, 0.0), (0.3, 0.4), (0.0, 1.0)))
    settings = PriorSettings(maximum_candidates=2, distance_scale=10, seed=11)
    raw = {
        "candidate_export_version": 1,
        "dimension": 3,
        "kind": "ALPHA",
        "scale": 10,
        "precision": 1,
        "seed": 11,
        "maximum_candidates": 2,
        "native_preparation_cpu_seconds": 0.1,
        "nodes": [
            {"id": 1, "pi": -4, "dad": 0, "count": 2, "edges": [[3, 0, 9, 10], [2, 1, 3, 5]]},
            {"id": 2, "pi": 2, "dad": 1, "count": 2, "edges": [[1, 0, 3, 5], [3, 0, 12, 7]]},
            {"id": 3, "pi": 3, "dad": 2, "count": 2, "edges": [[1, 0, 9, 10], [2, 0, 12, 7]]},
        ],
    }
    return problem, settings, raw


def test_prior_order_and_fp64_distance_views_have_same_members():
    problem, settings, raw = triangle()
    prior = parse_candidates(problem, settings, raw)
    assert [edge["to"] for edge in prior["rows"][0]] == [2, 1]
    assert prior["distance_rows"][0] == [1, 2]
    assert prior["directed_slots"] == 6 and prior["undirected_edges"] == 3
    assert prior["rows"][1][1]["distance"] == pytest.approx(np.sqrt(0.45))
    assert prior["rows"][1][1]["quantized_distance"] == 7
    assert prior["initial_tour_exported"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "identity",
        "dimension",
        "node_id",
        "duplicate",
        "quantized",
        "transformed",
        "count",
        "float_edge",
        "extra",
    ],
)
def test_bad_native_candidate_output_is_rejected(mutation):
    problem, settings, original = triangle()
    raw = copy.deepcopy(original)
    if mutation == "identity":
        raw["seed"] = 12
    elif mutation == "dimension":
        raw["nodes"].pop()
    elif mutation == "node_id":
        raw["nodes"][1]["id"] = 3
    elif mutation == "duplicate":
        raw["nodes"][0]["edges"][1] = raw["nodes"][0]["edges"][0]
    elif mutation == "quantized":
        raw["nodes"][0]["edges"][0][3] = 11
    elif mutation == "transformed":
        raw["nodes"][0]["edges"][0][2] = 8
    elif mutation == "count":
        raw["nodes"][0]["count"] = 1
    elif mutation == "float_edge":
        raw["nodes"][0]["edges"][0][0] = 3.0
    else:
        raw["optimal_tour"] = [1, 2, 3]
    with pytest.raises(ValueError):
        parse_candidates(problem, settings, raw)


@pytest.mark.skipif(os.environ.get("GP_FACO_REQUIRE_LKH_PRIOR") != "1", reason="需要本地候选适配器")
@pytest.mark.parametrize("kind", ["ALPHA", "POPMUSIC"])
@pytest.mark.parametrize("n", [7, 31, 500, 1000])
def test_native_candidate_member_distance_and_seed_replay(tmp_path, kind, n):
    binary = Path(
        os.environ.get(
            "GP_FACO_LKH_PRIOR_BINARY",
            str(PROJECT / ".deps/lkh-candidates-v1/GPLSACO-LKH-Candidates"),
        )
    )
    xy = np.random.default_rng(n + 133).random((n, 2))
    problem = Instance(f"synthetic-{n}", tuple(map(tuple, xy.tolist())))
    settings = PriorSettings(kind=kind, maximum_candidates=min(80, n - 1), seed=73001)
    first = prepare_candidates(problem, settings, binary, tmp_path / "first")
    second = prepare_candidates(problem, settings, binary, tmp_path / "second")
    for key in (
        "rows",
        "distance_rows",
        "node_pi",
        "directed_slots",
        "undirected_edges",
        "max_distance_quantization_error",
    ):
        assert first[key] == second[key]
    for row, distance_row in zip(first["rows"], first["distance_rows"], strict=True):
        assert set(distance_row) == {v["to"] for v in row}
        assert len(row) <= n - 1
    assert first["degree_min"] >= 1
    assert first["max_distance_quantization_error"] <= 0.5e-6 + 1e-15
