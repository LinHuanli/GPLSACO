"""离线任务身份、标签边界、失败保留和宏平均；不初始化CUDA。"""

from dataclasses import replace

import pytest
from gp_faco.data import Instance, Label, tour_cost
from gp_faco.fitness import PanelFitness, aggregate_panels, score_panel
from gp_faco.program_ir import Program
from gp_faco.worker import SolverSettings, SolveTask, WorkerProtocol


@pytest.fixture
def protocol():
    return WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "unit-test-device",
        "0",
        "0" * 64,
        dimensions=(3, 4),
        colonies=4,
    )


def instance(name, n=3, shift=0):
    return Instance(name, tuple((float(i) + shift, float(i % 2)) for i in range(n)))


def task(name="generation0:individual0:scale3", n=3):
    a, b = instance(f"a{n}", n), instance(f"b{n}", n, 3)
    return SolveTask(
        name,
        Program((0,), (4,)),
        (a, b),
        ((a.instance_id, 17), (b.instance_id, 17), (b.instance_id, 29), (b.instance_id, 41)),
        1.0,
    )


def valid_outcome(t, p):
    problems = {v.instance_id: v for v in t.problems}
    items = []
    labels = {}
    for name, value in problems.items():
        tour = tuple(range(value.dimension))
        labels[name] = Label(tour, tour_cost(value, tour))
    for name, _ in t.replicas:
        labels_entry = labels[name]
        items.append(
            {
                "has_incumbent": True,
                "tour": list(labels_entry.tour),
                "cost": labels_entry.cost,
                "completed_seconds": 0.5,
            }
        )
    return {
        "task_id": t.task_id(p),
        "protocol_sha256": p.sha256,
        "program_sha256": t.program.sha256,
        "occurrence_id": t.occurrence_id,
        "dimension": t.dimension,
        "status": "completed",
        "native_result": {
            "items": items,
            "budget_seconds": t.budget_seconds,
            "preparation_mode": t.preparation_mode,
            "actual_seconds": 0.6,
            "charged_seconds": 0.1,
            "elapsed_seconds": 0.7,
            "overrun_seconds": 0,
            "launched_batches": 2,
            "completed_batches": 2,
            "discarded_batches": 0,
        },
    }, labels


def test_identity_covers_occurrence_hardware_and_order(protocol):
    t = task()
    original = t.task_id(protocol)
    assert replace(t, problems=tuple(reversed(t.problems))).task_id(protocol) == original
    variations = (
        replace(t, occurrence_id="generation0:individual1:scale3"),
        replace(t, program=Program((0,), (5,))),
        replace(t, budget_seconds=2.0),
        replace(t, replicas=tuple(reversed(t.replicas))),
        replace(t, preparation_mode="end_to_end"),
        replace(t, experiment_mask=0xFFFF),
    )
    assert all(v.task_id(protocol) != original for v in variations)
    assert t.task_id(replace(protocol, driver_version="1")) != original
    assert t.task_id(replace(protocol, settings=SolverSettings(ants=4))) != original
    assert "label" not in t.manifest(protocol) and "coordinates" not in t.manifest(protocol)


def test_deep_snapshot_and_rejected_label_payload(protocol):
    coordinates = [[0, 0], [1, 0], [0, 1]]
    a = Instance("mutable", coordinates)
    t = SolveTask("copy", Program((0,), (4,)), (a,), tuple(("mutable", i) for i in range(4)), 1)
    before = t.task_id(protocol)
    coordinates[0][0] = 100
    assert t.task_id(protocol) == before and t.problems[0].coordinates[0] == (0, 0)
    with pytest.raises(TypeError):
        replace(t, problems=(Label((0, 1, 2), 3),))
    with pytest.raises(ValueError):
        replace(t, replicas=(("mutable", 0),) * 4)
    with pytest.raises(ValueError):
        replace(t, replicas=(("missing", 0),))
    with pytest.raises(ValueError):
        replace(t, budget_seconds=True)
    with pytest.raises(ValueError):
        replace(t, experiment_mask=0xFFFF0000)
    with pytest.raises(ValueError):
        t.manifest(replace(protocol, colonies=3))


@pytest.mark.parametrize("kind", ["identity", "missing", "late", "tour", "cost", "counts", "fee"])
def test_external_failure_keeps_whole_panel(protocol, kind):
    t = task()
    outcome, labels = valid_outcome(t, protocol)
    assert not score_panel(t, protocol, outcome, labels).failed
    result = outcome["native_result"]
    if kind == "identity":
        outcome["task_id"] = "other"
    elif kind == "missing":
        result["items"][0]["has_incumbent"] = False
    elif kind == "late":
        result["items"][0]["completed_seconds"] = 1.0001
    elif kind == "tour":
        result["items"][0]["tour"] = [0, 0, 1]
    elif kind == "cost":
        result["items"][0]["cost"] += 1
    elif kind == "counts":
        result["completed_batches"] = 3
    else:
        result["elapsed_seconds"] = 0.8
    scored = score_panel(t, protocol, outcome, labels)
    assert scored.failed and scored.members == ()
    assert aggregate_panels((t,), protocol, (scored,), (3,)) == float("inf")


def test_macro_average_and_out_of_order_identity(protocol):
    a, b = task(), task("generation0:individual0:scale4", 4)
    # n=3中a的一个seed为0，b的三个seed均30，实例平均15；n=4实例平均50。
    first = PanelFitness(
        a.task_id(protocol),
        protocol.sha256,
        a.program.sha256,
        3,
        tuple((name, seed, 0.0 if name == "a3" else 30.0) for name, seed in a.replicas),
    )
    second = PanelFitness(
        b.task_id(protocol),
        protocol.sha256,
        b.program.sha256,
        4,
        tuple((name, seed, 50.0) for name, seed in b.replicas),
    )
    assert aggregate_panels((a, b), protocol, (second, first), (3, 4)) == 32.5
    failed = replace(first, members=(), error="worker task failed")
    assert aggregate_panels((a, b), protocol, (second, failed), (3, 4)) == float("inf")
    with pytest.raises(ValueError):
        aggregate_panels((a, b), protocol, (second,), (3, 4))
    with pytest.raises(ValueError):
        aggregate_panels((a, b), protocol, (second, first, first), (3, 4))
    with pytest.raises(ValueError):
        aggregate_panels(
            (a, b), protocol, (second, replace(first, protocol_sha256="foreign")), (3, 4)
        )
    repeated = replace(a, occurrence_id="another-actual-evaluation")
    repeated_result = replace(first, task_id=repeated.task_id(protocol))
    with pytest.raises(ValueError):
        aggregate_panels((a, repeated), protocol, (first, repeated_result), (3,))


def test_bad_labels_abort_instead_of_random_training(protocol):
    t = task()
    outcome, labels = valid_outcome(t, protocol)
    with pytest.raises(ValueError):
        score_panel(t, protocol, outcome, {})
    key = t.problems[0].instance_id
    labels[key] = replace(labels[key], cost=100)
    with pytest.raises(ValueError):
        score_panel(t, protocol, outcome, labels)


def test_frozen_fee_table_is_complete_and_part_of_task_identity(protocol):
    original = task()
    charges = tuple((p.instance_id, 0.001, 0.02) for p in original.problems)
    frozen = replace(original, preparation_charges=charges)
    assert frozen.task_id(protocol) != original.task_id(protocol)
    assert replace(frozen, preparation_charges=tuple(reversed(charges))).task_id(
        protocol
    ) == frozen.task_id(protocol)
    for invalid in (
        charges[:1],
        charges + charges[:1],
        ((original.problems[0].instance_id, True, 0.2), charges[1]),
        ((original.problems[0].instance_id, 0.1, float("nan")), charges[1]),
    ):
        with pytest.raises(ValueError):
            replace(original, preparation_charges=invalid)
    with pytest.raises(ValueError):
        replace(frozen, preparation_mode="end_to_end")


def test_external_evaluator_checks_declared_fee_total(protocol):
    original = task()
    frozen = replace(
        original, preparation_charges=tuple((p.instance_id, 0.001, 0.02) for p in original.problems)
    )
    outcome, labels = valid_outcome(frozen, protocol)
    outcome["native_result"]["preparation_completed"] = True
    assert score_panel(frozen, protocol, outcome, labels).failed
    result = outcome["native_result"]
    result["charged_seconds"], result["elapsed_seconds"] = 0.042, 0.642
    assert not score_panel(frozen, protocol, outcome, labels).failed
