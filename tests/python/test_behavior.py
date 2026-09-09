"""固定事件窗口、零分母、原始轨迹篡改拒绝与任务身份隔离。"""

import copy
from dataclasses import replace

import numpy as np
import pytest
from gp_faco.baseline_policy import BaselinePolicy
from gp_faco.behavior import COUNTERS, summarize_behavior, validate_behavior, window_metrics
from gp_faco.data import Instance
from gp_faco.factorial_policy import FactorialPolicy
from gp_faco.program_ir import Program
from gp_faco.worker import FactorialTask, WorkerProtocol


@pytest.fixture
def example():
    policy = FactorialPolicy(
        "M00", BaselinePolicy(region=1, restart_mode="periodic", restart_period=3)
    )
    states = [
        {
            "return_rate": 0.0,
            "ls_work": 0.0,
            "stagnant_batches": 0,
            "epoch_batches": 0,
            "restarts": 0,
        }
        for _ in range(2)
    ]
    rows = []
    for batch in range(13):
        for colony in range(2):
            before = states[colony]
            restart = batch > 0 and batch % 3 == 0
            q = np.float32(0 if restart else before["return_rate"])
            q += np.float32(np.float32(1 - q) * np.float32(0.0625))
            after = {
                "return_rate": float(q),
                "ls_work": 0.0,
                "stagnant_batches": batch + 1,
                "epoch_batches": (0 if restart else before["epoch_batches"]) + 1,
                "restarts": before["restarts"] + restart,
            }
            rows.append(
                {
                    "batch": batch,
                    "colony": colony,
                    "ants": 4,
                    "dimension": 5,
                    "alternative": 0 if batch else 4,
                    "action": 20 if restart else 4,
                    "baseline_requested_action": 20 if restart else 4,
                    "legal_mask": 0xFFFFFFFF if batch else 0xFFFF,
                    "action_mask": 0xFFFFFFFF if batch else 0xFFFF,
                    "global_before": 10.0,
                    "reference_before": 10.0,
                    "reference_used": 10.0,
                    "global_after": 10.0,
                    "iteration_best_cost": 10.0,
                    "feedback_before": before,
                    "feedback_after": after,
                    **dict.fromkeys(COUNTERS, 0),
                    "exact_returns": 4,
                    "fingerprint_returns": 4,
                }
            )
            states[colony] = after
    return {
        "behavior": {
            "behavior_spec_id": 1,
            "device_bytes": 1024,
            "host_row_bytes": len(rows) * 256,
            "rows": rows,
        },
        "completed_batches": 13,
        "launched_batches": 13,
        "discarded_batches": 0,
        "total_tour_evaluations": 104,
        "completed_construction_steps": 0,
        "completed_ls_evaluations": 0,
        "control_states": states,
        "items": [{"cost": 10.0}, {"cost": 10.0}],
    }, {
        "dimension": 5,
        "colonies": 2,
        "ants": 4,
        "evaluation_limit": 52,
        "ls_evaluation_limit": 100000,
        "policy": policy,
    }


def test_windows_are_fixed_complete_and_overlapping_events_retained(example):
    result, contract = example
    summary = summarize_behavior(result, **contract)
    for solve in summary["solves"]:
        assert solve["complete_events"] == solve["boundary_events"] == 2
        assert [event["batch"] for event in solve["restart_events"]] == [3, 6, 9, 12]
        assert [event["complete"] for event in solve["restart_events"]] == [
            False,
            True,
            True,
            False,
        ]
        for event in solve["restart_events"][1:3]:
            assert event["before"]["fe"] == event["after"]["fe"] == 16
            assert event["additional_restarts_after"] == 1
        assert solve["whole_solve"]["restart_per_fe"] == 4 / 52
        assert solve["whole_solve"]["per_ant"]["exact_returns"] == 1
        assert set(solve["paired_windows"]["mean_after_minus_before"].values()) == {0}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "order",
        "mask",
        "action",
        "request",
        "bool",
        "counter",
        "return",
        "feedback",
        "global",
        "nan",
        "work",
        "final",
        "version",
        "unknown",
    ],
)
def test_tampered_observations_are_rejected(example, mutation):
    original, contract = example
    result = copy.deepcopy(original)
    rows = result["behavior"]["rows"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[3] = rows[2]
    elif mutation == "order":
        rows[2], rows[3] = rows[3], rows[2]
    elif mutation == "mask":
        rows[0]["legal_mask"] = 0xFFFFFFFF
    elif mutation == "action":
        rows[0]["action"] = 20
    elif mutation == "request":
        rows[0]["baseline_requested_action"] = 5
    elif mutation == "bool":
        rows[0]["batch"] = False
    elif mutation == "counter":
        rows[0]["ls_move_evaluations"] = 400001
    elif mutation == "return":
        rows[0]["final_new_edges"] = 1
    elif mutation == "feedback":
        rows[0]["feedback_after"]["return_rate"] = 0.9
    elif mutation == "global":
        rows[0]["global_after"] = 11.0
    elif mutation == "nan":
        rows[0]["iteration_best_cost"] = float("nan")
    elif mutation == "work":
        result["completed_ls_evaluations"] = 1
    elif mutation == "final":
        result["items"][0]["cost"] = 9.0
    elif mutation == "version":
        result["behavior"]["behavior_spec_id"] = True
    else:
        rows[0]["surprise"] = 0
    with pytest.raises(ValueError, match="行为"):
        validate_behavior(result, **contract)


def test_empty_solve_and_no_event_have_missing_window_values(example):
    result, contract = example
    result["behavior"]["rows"] = []
    result["behavior"]["host_row_bytes"] = 0
    result.update(completed_batches=0, launched_batches=0, total_tour_evaluations=0)
    contract["evaluation_limit"] = 0
    for solve in summarize_behavior(result, **contract)["solves"]:
        assert solve["paired_windows"] is None
        assert solve["whole_solve"]["restart_per_fe"] is None
        assert solve["restart_events"] == []
    assert window_metrics([])["mne_action_frequencies"] == [None] * 4


def test_observation_is_task_identity_and_preserves_controller_identity():
    problem = Instance("a", ((0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)))
    task = FactorialTask(
        "observe",
        Program((0,), (4,), feature_spec_id=2),
        (problem,),
        (("a", 17),),
        evaluation_limit_per_colony=32,
        preparation_mode="cached",
        factorial_policy=FactorialPolicy("M10", BaselinePolicy()),
    )
    protocol = WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "test",
        "0",
        "0" * 64,
        dimensions=(5,),
        colonies=1,
    )
    observed = replace(task, record_behavior=True)
    assert task.controller_sha256 == observed.controller_sha256
    assert task.task_id(protocol) != observed.task_id(protocol)
    assert "behavior_spec_id" not in task.manifest(protocol)
    assert observed.manifest(protocol)["behavior_spec_id"] == 1
    for wrong in (1, None, "yes"):
        with pytest.raises(TypeError):
            replace(task, record_behavior=wrong)
