"""正式入口的数据隔离、完整配置、面板预登记及观察失败后的真实返回保留。"""

import copy
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from gp_faco.checkpoint import load_checkpoint
from gp_faco.training import TrainingRun
from test_training import WorkerFarm
from test_training import setup as shared_setup

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("research_e1", PROJECT / "scripts/research_e1.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


@pytest.fixture
def core_setup():
    return shared_setup.__wrapped__()


def test_formal_settings_keep_all_five_seeds_and_full_evaluation_counts():
    config = json.loads((PROJECT / "configs/e1_protocol_v1.json").read_text())
    for condition in ("GP-Full", "GP-NoFeedback"):
        for seed in (1103, 2207, 3313, 4409, 5519):
            settings = research.training_settings(config, condition, seed)
            assert settings.evolution.population * settings.evolution.generations == 6400
            assert settings.evolution.elites == 4 and settings.evolution.feature_spec_id == 2
            assert settings.budgets == ((500, 4096), (1000, 4096))
            assert settings.evolution.no_feedback == (condition == "GP-NoFeedback")
            assert settings.budget_kind == "search_tour_evaluations"
    for condition, seed in (("GP-NoFeedback", 7), ("escape", 1103)):
        with pytest.raises(ValueError):
            research.training_settings(config, condition, seed)


def test_freeze_member_enumeration_never_reads_coordinates_or_labels():
    class IDsOnly:
        def record_ids(self, role, n):
            return [f"{n}-{role}-{i}" for i in range(4)]

        def load_instance(self, identity):
            raise AssertionError("freeze不应读取坐标")

        def load_label(self, identity):
            raise AssertionError("freeze不应读取标签")

    config = {
        "dimensions": [5, 7],
        "data": {
            "expected_counts": {
                str(n): {role: 4 for role in ("train", "validation", "test")} for n in (5, 7)
            }
        },
    }
    result = research.data_members(IDsOnly(), config)
    assert len(result["5"]["test"]) == 4

    class Overlap(IDsOnly):
        def record_ids(self, role, n):
            return [f"shared-{i}" for i in range(4)]

    with pytest.raises(ValueError, match="重叠"):
        research.data_members(Overlap(), config)
    config["data"]["expected_counts"]["5"]["test"] = 3
    with pytest.raises(ValueError, match="成员数"):
        research.data_members(IDsOnly(), config)


def counted_setup(values):
    settings, protocol, source = values
    settings = replace(
        settings,
        evolution=replace(settings.evolution, feature_spec_id=2),
        budgets=((5, 8), (7, 8)),
        budget_kind="search_tour_evaluations",
        preparation_mode="cached",
    )
    config = {
        "training": asdict(settings),
        "evolution": asdict(settings.evolution),
        "dimensions": [5, 7],
    }
    members = {str(n): {"train": source.training[n]} for n in (5, 7)}
    return settings, protocol, source, research.training_panels(config, members)


def test_registered_observations_preserve_plain_evolution_and_original_futures(
    tmp_path, core_setup
):
    settings, protocol, source, panels = counted_setup(core_setup)
    ordinary_farm, observed_farm = WorkerFarm(), WorkerFarm(timeout=True)
    ordinary = TrainingRun(
        tmp_path / "ordinary", settings, protocol, source.data(), worker_factory=ordinary_farm
    )
    expected = ordinary.run()
    calls = []

    def boundary(protocol, pid):
        calls.append(pid)
        if len(calls) % 2 == 0:
            raise OSError("injected post-return resource query failure")
        return {"foreign_processes": 0}

    observed = research.RegisteredTrainingRun(
        tmp_path / "observed",
        settings,
        protocol,
        source.data(),
        worker_factory=observed_farm,
        expected_panels=panels,
        boundary=boundary,
    )
    actual = observed.run()
    assert actual["selected"] == expected["selected"]
    assert len(observed_farm.submissions) == len(ordinary_farm.submissions)
    assert all(v.observations == 2 for v in observed_farm.futures)
    state = load_checkpoint(tmp_path / "observed/checkpoint.json")
    assert len(state["admission"]) == len(observed_farm.submissions) + len(
        observed_farm.preparations
    )
    assert [v["panels"] for v in state["training_panels"]] == panels
    for path in (tmp_path / "observed/tasks").glob("*.json"):
        record = load_checkpoint(path)
        assert record["outcome"]["status"] == "completed"
        assert record["outcome"]["gpu_boundary_after"]["foreign_processes"] is None
        if record["kind"] == "solve":
            assert record["checked"]["error"] is None


@pytest.mark.parametrize("cause", ["wrong_panel", "busy_gpu"])
def test_no_submission_if_registered_panel_or_device_admission_fails(tmp_path, core_setup, cause):
    settings, protocol, source, panels = counted_setup(core_setup)
    panels = copy.deepcopy(panels)
    if cause == "wrong_panel":
        panels[0][0]["seeds"][0] ^= 1
    farm = WorkerFarm()
    run = research.RegisteredTrainingRun(
        tmp_path / "rejected",
        settings,
        protocol,
        source.data(),
        worker_factory=farm,
        expected_panels=panels,
        boundary=lambda p, pid: {"foreign_processes": 1},
    )
    with pytest.raises(ValueError):
        run.run()
    assert not farm.submissions
