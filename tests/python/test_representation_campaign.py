"""Controller全链路选模身份、预算完整性与配对统计。"""

import copy

import pytest
from gp_faco.controller import Controller
from gp_faco.program_ir import Program
from gp_faco.representation_campaign import RUN_IDS, candidate_controllers, rank_results
from gp_faco.representation_pilot import BASELINES
from gp_faco.representation_statistics import comparison_statistics, holm


def leaf(identifier):
    tree = Program((0,), (5,), feature_spec_id=2)
    return Controller("joint_single", (tree,), identifier).to_dict()


def test_selection_deduplicates_ir_but_preserves_representation():
    a, b = leaf("first"), leaf("same")
    c = Controller(
        "conditional_three",
        (
            Program((0,), (0,), feature_spec_id=2),
            Program((0,), (0,), feature_spec_id=2),
            Program((0,), (5,), feature_spec_id=2),
        ),
        "three",
    ).to_dict()
    summary = {"history": [{"winner": a}, {"winner": b}], "final_population": [b, c]}
    assert [c["controller_id"] for c in candidate_controllers(summary)] == ["first", "three"]


def test_selection_requires_complete_pairs_and_keeps_three_tree_bundle():
    c = leaf("one")
    panel = {"dimension": 500, "ids": ["a", "b"], "seeds": [17]}
    values = [
        {"instance_id": name, "dimension": 500, "seed": 17, "iterations": 1000, "gap_percent": gap}
        for name, gap in [("a", 0.2), ("b", 0.4)]
    ]
    result = {"members": [{"controller_id": "one", "rows": values}]}
    scores, order = rank_results([result], [c], panel, 1000)
    assert order == ["one"] and scores["one"]["gap_percent"] == pytest.approx(0.3)
    bad = copy.deepcopy(result)
    bad["members"][0]["rows"][1]["instance_id"] = "a"
    with pytest.raises(ValueError, match="收齐"):
        rank_results([bad], [c], panel, 1000)
    with pytest.raises(ValueError, match="收齐"):
        rank_results([result], [c], panel, 5000)


def test_all_methods_required_and_holm_family_includes_fixed_baselines():
    panels = [{"dimension": n, "ids": ["a", "b"], "seeds": [17, 29]} for n in (500, 1000)]
    rows = [
        {
            "method": method,
            "dimension": panel["dimension"],
            "instance_id": name,
            "seed": seed,
            "iterations": 5000,
            "gap_percent": float(method in BASELINES),
            "cost": 100 + (method in BASELINES),
        }
        for method in [*RUN_IDS, *BASELINES]
        for panel in panels
        for name in panel["ids"]
        for seed in panel["seeds"]
    ]
    config = {"test_iterations": 5000, "statistics_seed": 91001, "bootstrap_replicates": 10}
    result = comparison_statistics(rows, config, panels)
    assert len(result["comparisons"]) == 18
    assert sum("holm_p" in r for r in result["comparisons"]) == 9
    assert result["comparisons"][0]["wins_ties_losses"] == [2, 0, 0]
    assert result["comparisons"][0]["relative_route_improvement_percent"] == pytest.approx(
        100 / 101
    )
    for broken in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError):
            comparison_statistics(broken, config, panels)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_result_published_during_liveness_query_is_not_retried(tmp_path, monkeypatch):
    from gp_faco import campaign_scheduler as module
    from gp_faco.checkpoint import atomic_json

    path = tmp_path / "jobs" / "baseline_native-race"
    path.mkdir(parents=True)
    atomic_json(path / "runtime.json", {"status": "running", "host": "example", "pid": 1})
    scheduler = module.CampaignScheduler.__new__(module.CampaignScheduler)
    scheduler.directory = tmp_path
    scheduler.active = {path.name: {"gpu_uuid": None, "attempt": 1, "host": "example"}}
    scheduler.states = {path.name: "running"}
    scheduler._next_liveness = {}
    scheduler.next_discovery = 0

    def finished_during_query(_):
        atomic_json(path / "result.json", {"status": "completed"})
        return False

    monkeypatch.setattr(module, "process_alive", finished_during_query)
    assert scheduler.reconcile()
    assert scheduler.states[path.name] == "completed"
    assert not scheduler.active
    assert not (path / "failures.json").exists()


def test_compact_scores_keep_streamed_rows_and_separate_timing(tmp_path):
    from gp_faco.campaign import read
    from gp_faco.checkpoint import atomic_json
    from gp_faco.representation_campaign import RepresentationScheduler

    scheduler = RepresentationScheduler.__new__(RepresentationScheduler)
    scheduler.directory = tmp_path
    scheduler.jobs = {}
    for i in range(2):
        key = f"population-test-{i}"
        scheduler.jobs[key] = {
            "controllers": [{"controller_id": "one"}],
            "dimension": 500,
            "pairs": [(f"instance-{i}", 17)],
        }
        path = tmp_path / "jobs" / key
        path.mkdir(parents=True)
        atomic_json(
            path / "result.json",
            {
                "status": "completed",
                "solve_seconds": 2.0,
                "members": [{"rows": [{"instance_id": f"instance-{i}", "tour": [0, 1, 2]}]}],
            },
        )
    scheduler.save_result_rows(list(scheduler.jobs), "test_scores.json")
    result = read(tmp_path / "test_scores.json")
    assert result["rows"] == [{"instance_id": "instance-0"}, {"instance_id": "instance-1"}]
    assert sum(t["computed_pairs"] for t in result["timings"]) == 2
    assert sum(t["solve_seconds"] for t in result["timings"]) == 4


def test_persistent_worker_assignment_publishes_process_before_request(tmp_path):
    from gp_faco.campaign import read
    from gp_faco.campaign_pool import PopulationScheduler
    from gp_faco.checkpoint import atomic_json

    scheduler = PopulationScheduler.__new__(PopulationScheduler)
    scheduler.directory = tmp_path
    scheduler.active, scheduler.states = {}, {}
    (tmp_path / "jobs" / "job").mkdir(parents=True)
    worker = tmp_path / "workers" / "gpu"
    worker.mkdir(parents=True)
    atomic_json(
        worker / "runtime.json",
        {
            "host": "example",
            "pid": 17,
            "start_ticks": 41,
            "status": "idle",
        },
    )
    scheduler.submit_pool("job", {"host": "example", "uuid": "gpu"})
    runtime = read(tmp_path / "jobs/job/runtime.json")
    request = read(worker / "request.json")
    assert runtime["status"] == "assigned"
    assert (runtime["pid"], runtime["start_ticks"], runtime["job"]) == (17, 41, "job")
    assert request["worker_pid"] == runtime["pid"]


def test_missing_remote_job_receipt_uses_persistent_worker_lease(tmp_path, monkeypatch):
    from gp_faco import campaign_scheduler as module

    path = tmp_path / "jobs/job"
    path.mkdir(parents=True)
    scheduler = module.CampaignScheduler.__new__(module.CampaignScheduler)
    scheduler.directory = tmp_path
    scheduler.active = {
        "job": {
            "host": "example",
            "gpu_uuid": "gpu",
            "attempt": 1,
            "persistent_worker": True,
            "launched_unix": 0,
        }
    }
    scheduler.states = {"job": "running"}
    scheduler._next_liveness = {}
    queried = []
    monkeypatch.setattr(
        module, "job_lock_active", lambda host, directory: queried.append(directory) or True
    )
    assert not scheduler.reconcile()
    assert queried == [tmp_path / "workers/gpu"]
    assert scheduler.states["job"] == "running"
    assert not (path / "failures.json").exists()


def test_campaign_read_opens_file_without_stale_existence_probe():
    from gp_faco.campaign import read

    class RemotePublishedFile:
        def exists(self):
            return False

        def read_text(self):
            return '{"status": "completed"}'

    assert read(RemotePublishedFile()) == {"status": "completed"}
