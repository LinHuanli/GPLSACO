"""原始距离与文件语义的反例；矩阵金标准不调用解析器的布局展开。"""

from dataclasses import asdict

import pytest
from gp_faco.distance import ATT, CEIL_2D, EUC_2D, GEO, coordinate_distance
from gp_faco.tsplib import OriginalProblem, UnsupportedTSPLIB, parse_problem, parse_tours


def coordinate_file(metric="EUC_2D", nodes="1 0 0\n2 0.5 0\n3 2.5 0", extra=""):
    return (
        "NAME : tiny\nTYPE : TSP\nDIMENSION : 3\n"
        "COMMENT : best solution 123456789 must remain outside solver\n"
        f"EDGE_WEIGHT_TYPE : {metric}\n{extra}NODE_COORD_SECTION\n{nodes}\nEOF\n"
    )


def test_integer_rounding_and_coordinate_identity():
    p = parse_problem(coordinate_file(nodes="3 2.5 0\n1 0 0\n2 0.5 0"))
    assert p.coordinates == ((0.0, 0.0), (0.5, 0.0), (2.5, 0.0))
    assert p.distance_spec == EUC_2D
    assert p.distance(0, 1) == 1 and p.distance(0, 2) == 3
    assert p.tour_cost((0, 1, 2)) == 6
    assert "123456789" not in str(asdict(p))
    with pytest.raises(ValueError):
        p.distance(-1, 0)
    with pytest.raises(ValueError):
        p.distance(True, 0)


def test_duplicate_nodes_remain_distinct_and_nonself_zero_is_valid():
    p = parse_problem(coordinate_file(nodes="1 0 0\n2 0 0\n3 2 0"))
    assert p.dimension == 3 and p.distance(0, 1) == 0
    assert p.tour_cost((0, 1, 2)) == 4
    with pytest.raises(ValueError):
        p.tour_cost((0, 0, 2))


def test_ceil_att_and_geo_hand_boundaries():
    xy = ((0.0, 0.0), (2.1, 0.0), (3.0, 1.0))
    assert coordinate_distance(xy, 0, 1, EUC_2D) == 2
    assert coordinate_distance(xy, 0, 1, CEIL_2D) == 3
    assert coordinate_distance(xy, 0, 2, ATT) == 1
    assert coordinate_distance(((0.0, 0.0), (3.0, 4.0)), 0, 1, ATT) == 2
    geo = OriginalProblem("geographic", 3, GEO, ((0.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    assert geo.distance(0, 1) == 112
    assert geo.distance(0, 2) == 1 and geo.distance(0, 0) == 0
    with pytest.raises(ValueError):
        coordinate_distance(((0.0, 0.0), (1e200, 0.0)), 0, 1, EUC_2D)


MATRICES = {
    "FULL_MATRIX": "0 11 12 13 11 0 23 24 12 23 0 34 13 24 34 0",
    "UPPER_ROW": "11 12 13 23 24 34",
    "LOWER_ROW": "11 12 23 13 24 34",
    "UPPER_DIAG_ROW": "0 11 12 13 0 23 24 0 34 0",
    "LOWER_DIAG_ROW": "0 11 0 12 23 0 13 24 34 0",
    "UPPER_COL": "11 12 23 13 24 34",
    "LOWER_COL": "11 12 13 23 24 34",
    "UPPER_DIAG_COL": "0 11 0 12 23 0 13 24 34 0",
    "LOWER_DIAG_COL": "0 11 12 13 0 23 24 0 34 0",
}
GOLDEN = ((0, 11, 12, 13), (11, 0, 23, 24), (12, 23, 0, 34), (13, 24, 34, 0))


def explicit_file(layout, values, display=""):
    return (
        "NAME: explicit\nTYPE: TSP\nDIMENSION: 4\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
        f"EDGE_WEIGHT_FORMAT: {layout}\nEDGE_WEIGHT_SECTION\n{values}\n{display}EOF\n"
    )


@pytest.mark.parametrize("layout", list(MATRICES))
def test_all_nine_explicit_layouts_and_display_independence(layout):
    display = "DISPLAY_DATA_SECTION\n1 0 0\n2 999 10000\n3 -5 3\n4 7 -9\n"
    p = parse_problem(explicit_file(layout, MATRICES[layout], display))
    assert p.coordinates == ()
    assert tuple(tuple(p.distance(a, b) for b in range(4)) for a in range(4)) == GOLDEN
    assert p.tour_cost((0, 1, 2, 3)) == 81


@pytest.mark.parametrize(
    "change",
    [
        ("TYPE : TSP", "TYPE : ATSP"),
        ("DIMENSION : 3", "DIMENSION : 3.0"),
        ("DIMENSION : 3", "DIMENSION : 4"),
        ("EDGE_WEIGHT_TYPE : EUC_2D", "EDGE_WEIGHT_TYPE : UNKNOWN"),
        ("NODE_COORD_SECTION", "DISPLAY_DATA_SECTION"),
        ("3 2.5 0", "2 2.5 0"),
        ("3 2.5 0", "3 nan 0"),
        ("3 2.5 0", "3 2.5 0 2"),
        ("EOF", "FIXED_EDGES_SECTION\n1 2\n-1\nEOF"),
        ("EOF", "EOF\nextra"),
        ("DIMENSION : 3", "DIMENSION : 3\nDIMENSION : 3"),
    ],
)
def test_malformed_or_changed_problem_is_not_silently_reinterpreted(change):
    with pytest.raises(ValueError):
        parse_problem(coordinate_file().replace(*change))


@pytest.mark.parametrize(
    "values",
    [
        "0 11 12 13 10 0 23 24 12 23 0 34 13 24 34 0",  # 非对称
        "1 11 12 13 11 0 23 24 12 23 0 34 13 24 34 0",  # 非零对角
        "0 -1 12 13 -1 0 23 24 12 23 0 34 13 24 34 0",
        "0 11 12 13 11 0 23 24 12 23 0 34 13 24 34",  # 缺失
        "0 11.0 12 13 11 0 23 24 12 23 0 34 13 24 34 0",
    ],
)
def test_explicit_matrix_integrity(values):
    with pytest.raises(ValueError):
        parse_problem(explicit_file("FULL_MATRIX", values))


def test_conflicting_coordinate_weight_definition_is_rejected():
    with pytest.raises(ValueError):
        parse_problem(coordinate_file(extra="EDGE_WEIGHT_FORMAT: FULL_MATRIX\n"))
    with pytest.raises(ValueError):
        parse_problem(coordinate_file(extra="NODE_COORD_TYPE: NO_COORDS\n"))
    with pytest.raises(UnsupportedTSPLIB):
        parse_problem(explicit_file("FUNCTION", "1 2 3"))


def test_tours_are_separate_complete_permutations():
    header = "NAME: independent\nTYPE: TOUR\nDIMENSION: 3\nTOUR_SECTION\n"
    assert parse_tours(header + "1 2 3 -1\nEOF", 3) == ((0, 1, 2),)
    assert parse_tours(header + "1 2 3 -1 3 2 1 -1 -1\nEOF", 3) == ((0, 1, 2), (2, 1, 0))
    for value in ["1 1 3 -1", "1 2 -1", "1 2 3", "1 2 4 -1", "1 2 3 -1 -1 2", "-1"]:
        with pytest.raises(ValueError):
            parse_tours(header + value + "\nEOF", 3)
