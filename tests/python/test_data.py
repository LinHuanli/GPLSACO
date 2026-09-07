import math

import pytest
from gp_faco.data import Instance, parse_record, point_set_hash, tour_cost, write_explicit_tsplib


def test_closed_tour_and_label_separation():
    instance, label = parse_record("0 0 1 0 1 1 0 1 output 1 2 3 4 1", "square")
    assert label.tour == (0, 1, 2, 3)
    assert label.cost == 4.0
    assert not hasattr(instance, "tour") and not hasattr(instance, "cost")
    assert label.certificate_status == "not_independently_verified"


@pytest.mark.parametrize(
    "text",
    [
        "0 0 1 0 0 1 output 1 2 2 1",
        "0 0 1 0 0 1 output 1 2 3 2",
        "0 0 1 0 0 1 output 1 2 4 1",
        "nan 0 1 0 0 1 output 1 2 3 1",
        "0 0 1 0 0 output 1 2 3 1",
        "0 0 1 0 0 1 output 1 2 3 output",
        "0 0 1 0 output 1 2 1",
        "0 0 0 0 0 0 output 1 2 3 1",
    ],
)
def test_malformed_record_rejected(text):
    with pytest.raises(ValueError):
        parse_record(text, "bad")


def test_original_continuous_metric_and_point_permutation(tmp_path):
    instance = Instance("triangle", ((0.0, 0.0), (0.6, 0.0), (0.0, 0.8)))
    assert math.isclose(tour_cost(instance, (0, 1, 2)), 2.4)
    permuted = Instance("permuted", tuple(reversed(instance.coordinates)))
    assert point_set_hash(instance) == point_set_hash(permuted)
    output = tmp_path / "triangle.tsp"
    write_explicit_tsplib(instance, output)
    assert "EXPLICIT" in output.read_text()
    matrix = output.read_text().split("EDGE_WEIGHT_SECTION\n")[1].split("EOF")[0]
    assert list(map(float, matrix.split())) == [0.0, 0.6, 0.8, 0.0, 1.0, 0.0]


def test_no_silent_tsplib_distance_guess():
    with pytest.raises(ValueError):
        Instance("unsupported", ((0, 0), (1, 0), (0, 1)), "EUC_2D")
