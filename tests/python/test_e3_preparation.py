"""正式范围、无标签准备和中断事务：成功先验不可重跑，未知终态不可猜测。"""

import copy
import importlib.util
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from gp_faco import e3_preparation as prep
from gp_faco.checkpoint import atomic_json
from gp_faco.data import tour_cost
from gp_faco.e3_protocol import preparation_jobs, static_policies, training_panels, validate_config
from gp_faco.worker import PROJECT, content_hash, coordinate_hash, file_hash
from test_candidate_prior import triangle
from test_graph_matching import fixture as graph_fixture

_spec = importlib.util.spec_from_file_location(
    "audit_e3_graphs", PROJECT / "scripts/audit_e3_graphs.py"
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def formal_config():
    return json.loads((PROJECT / "configs/e3_protocol_v1.json").read_text())


def members_for(config):
    return {
        str(n): {
            role: [f"{n}-{role}-{i:06d}" for i in range(count)]
            for role, count in config["data"]["expected_counts"][str(n)].items()
        }
        for n in config["dimensions"]
    }


def test_formal_design_full_twenty_runs_and_exact_graph_population():
    config = formal_config()
    validate_config(config)
    assert len(static_policies(config)) == 160
    members = members_for(config)
    panels = training_panels(config, members)
    development = {str(n): [f"{n}-development-{i}" for i in range(128)] for n in (500, 1000)}
    jobs = preparation_jobs(config, members, panels, development)
    assert len(panels) == 50 and len(jobs) == 2296
    assert [sum(j["dimension"] == n for j in jobs) for n in (500, 1000)] == [1146, 1150]
    assert sum("gp_training_sampled" in j["roles"] for j in jobs) == 1592
    assert sum("gp_validation" in j["roles"] for j in jobs) == 512
    assert all("test" not in j["instance_id"] for j in jobs)
    assert len(config["conditions"]) * len(config["evolution_seeds"]) == 20


def test_independent_population_audit_precedes_coordinate_loading(tmp_path):
    config = formal_config()
    members = members_for(config)
    development = {str(n): [f"{n}-development-{i:03d}" for i in range(128)] for n in (500, 1000)}
    panels = training_panels(config, members)
    jobs = preparation_jobs(config, members, panels, development)
    source = SimpleNamespace(
        record_ids=lambda role, n: (
            development[str(n)] if role == "development" else members[str(n)][role]
        )
    )
    values = {
        "members.json": members,
        "development.json": development,
        "training_panels.json": panels,
    }
    for name, value in values.items():
        atomic_json(tmp_path / name, value)
    plan = {"config": config, "files": {name: file_hash(tmp_path / name) for name in values}}
    _audit.verify_members(source, tmp_path, plan, jobs)
    wrong = [dict(row) for row in jobs]
    wrong[0] = {**wrong[0], "instance_id": members["500"]["test"][0]}
    with pytest.raises(AssertionError, match="完整并集"):
        _audit.verify_members(source, tmp_path, plan, wrong)


@pytest.mark.parametrize(
    "change", ["wall", "boolean_version", "short_training", "prior_seed", "duplicate_grid"]
)
def test_formal_design_rejects_scope_drift(change):
    config = formal_config()
    if change == "wall":
        config["wall_clock_limit"] = 3600
    elif change == "boolean_version":
        config["graph_spec_id"] = True
    elif change == "short_training":
        config["evolution"]["generations"] = 3
    elif change == "prior_seed":
        config["priors"]["ALPHA"]["seed"] += 1
    else:
        config["static_search"]["policy_grid"]["regions"] = [0, 1, 2, 2]
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "change", ["test_in_development", "duplicate_development", "cross_dimension"]
)
def test_preparation_refuses_overlap_and_test_members(change):
    config = formal_config()
    members = members_for(config)
    panels = training_panels(config, members)
    dev = {str(n): [f"{n}-development-{i}" for i in range(128)] for n in (500, 1000)}
    if change == "test_in_development":
        dev["500"][0] = members["500"]["test"][0]
    elif change == "duplicate_development":
        dev["500"][1] = dev["500"][0]
    else:
        dev["1000"][0] = dev["500"][0]
    with pytest.raises(ValueError):
        preparation_jobs(config, members, panels, dev)


@pytest.fixture
def transaction(tmp_path, monkeypatch):
    problem, settings, raw = triangle()
    binary = tmp_path / "binary"
    binary.write_text("synthetic candidate executable identity")
    atomic_json(
        tmp_path / "build-manifest.json", {"exit_code": 0, "binary_sha256": file_hash(binary)}
    )
    calls = {"common": 0, "ALPHA": 0, "POPMUSIC": 0, "instance": 0}

    def fake_prepare(problem, settings, binary, directory):
        calls[settings.kind] += 1
        directory.mkdir()
        for name in ("native.log", "problem.tsp", "parameters.par"):
            (directory / name).write_text(name)
        manifest = {
            "instance_id": problem.instance_id,
            "coordinate_sha256": coordinate_hash(problem),
            "dimension": problem.dimension,
            "settings": asdict(settings),
            "binary_sha256": file_hash(binary),
            "build_manifest_sha256": file_hash(binary.parent / "build-manifest.json"),
            "module_sha256": file_hash(PROJECT / "python/gp_faco/candidate_prior.py"),
            "wall_clock_limit": None,
            "problem_file_sha256": file_hash(directory / "problem.tsp"),
            "parameter_file_sha256": file_hash(directory / "parameters.par"),
        }
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(
            directory / "resources.json",
            {"exit_code": 0, "log_sha256": file_hash(directory / "native.log")},
        )
        atomic_json(directory / "native-candidates.json", {**raw, "kind": settings.kind})

    def common(xy, _settings):
        calls["common"] += 1
        assert xy.tolist() == [list(p) for p in problem.coordinates]
        return {
            "tour": [0, 1, 2],
            "cost": tour_cost(problem, [0, 1, 2]),
            "cheap_seconds": 0.01,
            "preparation_seconds": 0.02,
        }

    def load_instance(name):
        calls["instance"] += 1
        assert name == problem.instance_id
        return problem

    def load_label(*_args):
        pytest.fail("图准备不应读取标签")

    plan = {
        "sha256": "a" * 64,
        "candidate_binary_path": str(binary.relative_to(PROJECT)),
        "config": {
            "priors": {kind: {**asdict(settings), "kind": kind} for kind in prep.KINDS},
            "graph": asdict(prep.GraphSettings()),
        },
    }
    source = SimpleNamespace(load_instance=load_instance, load_label=load_label)
    output = tmp_path / "output"
    monkeypatch.setattr(prep, "prepare_candidates", fake_prepare)
    monkeypatch.setattr(prep, "reusable_entry", lambda *_args: None)
    monkeypatch.setattr(
        prep,
        "_context",
        (plan, source, output, SimpleNamespace(prepare_common_initial=common), None),
        raising=False,
    )
    job = {
        "index": 0,
        "dimension": 3,
        "instance_id": problem.instance_id,
        "roles": ["gp_training_sampled"],
    }
    return SimpleNamespace(
        problem=problem,
        settings=settings,
        binary=binary,
        calls=calls,
        output=output,
        plan=plan,
        job=job,
        directory=output / "instances/00000",
    )


def test_complete_resume_checks_bytes_and_never_repeats_native(transaction):
    t = transaction
    first = prep.run_job(t.job)
    assert not first["resumed"]
    second = prep.run_job(t.job)
    assert second["resumed"] and first["receipt_sha256"] == second["receipt_sha256"]
    assert t.calls == {"common": 1, "ALPHA": 1, "POPMUSIC": 1, "instance": 1}
    (t.directory / "ALPHA/native.log").write_text("changed")
    with pytest.raises(ValueError, match="已完成实例文件改变"):
        prep.run_job(t.job)
    assert t.calls["ALPHA"] == 1


def test_resume_between_priors_keeps_common_and_successful_native(transaction, monkeypatch):
    t = transaction
    original = prep.ensure_prior

    def fail_before_second(problem, settings, binary, directory):
        if settings.kind == "POPMUSIC":
            raise RuntimeError("simulated coordinator interruption")
        return original(problem, settings, binary, directory)

    monkeypatch.setattr(prep, "ensure_prior", fail_before_second)
    with pytest.raises(RuntimeError, match="simulated"):
        prep.run_job(t.job)
    assert t.calls["ALPHA"] == 1 and t.calls["POPMUSIC"] == 0
    monkeypatch.setattr(prep, "ensure_prior", original)
    prep.run_job(t.job)
    assert t.calls["common"] == t.calls["ALPHA"] == t.calls["POPMUSIC"] == 1
    assert len(list((t.directory / "attempts").glob("*.json"))) == 2


def test_resume_after_graph_write_without_repeating_native(transaction, monkeypatch):
    t = transaction
    original = prep.atomic_json

    def fail_commit(path, value):
        if path.name == "complete.json":
            raise RuntimeError("simulated commit interruption")
        return original(path, value)

    monkeypatch.setattr(prep, "atomic_json", fail_commit)
    with pytest.raises(RuntimeError, match="simulated"):
        prep.run_job(t.job)
    graph_hash = file_hash(t.directory / "graphs.json")
    monkeypatch.setattr(prep, "atomic_json", original)
    prep.run_job(t.job)
    assert file_hash(t.directory / "graphs.json") == graph_hash
    assert t.calls["common"] == t.calls["ALPHA"] == t.calls["POPMUSIC"] == 1


@pytest.mark.parametrize("status", [None, 17])
def test_unknown_or_failed_native_attempt_never_restarts(transaction, status):
    t = transaction
    directory = t.output / "incomplete"
    directory.mkdir(parents=True)
    if status is not None:
        atomic_json(directory / "resources.json", {"exit_code": status})
    with pytest.raises(ValueError):
        prep.ensure_prior(t.problem, t.settings, t.binary, directory)
    assert t.calls["ALPHA"] == 0


def test_prior_parser_recovery_needs_confirmed_zero_exit(transaction):
    t = transaction
    directory = t.output / "parse-only"
    directory.parent.mkdir()
    prep.prepare_candidates(t.problem, t.settings, t.binary, directory)
    assert not (directory / "prior.json").exists()
    prior, _ = prep.ensure_prior(t.problem, t.settings, t.binary, directory)
    assert prior["dimension"] == 3 and t.calls["ALPHA"] == 1
    value = json.loads((directory / "native-candidates.json").read_text())
    value["nodes"][0]["edges"][0][3] += 1
    atomic_json(directory / "native-candidates.json", value)
    with pytest.raises(ValueError):
        prep.ensure_prior(t.problem, t.settings, t.binary, directory)
    assert t.calls["ALPHA"] == 1


def test_completed_identity_collision_rejected(transaction):
    t = transaction
    prep.run_job(t.job)
    other = {**t.job, "instance_id": "different"}
    with pytest.raises(ValueError, match="身份改变"):
        prep.run_job(other)
    assert t.calls["instance"] == 1


def test_concurrent_instance_lock_is_not_a_second_attempt(transaction):
    t = transaction
    with prep.exclusive_lock(t.directory / ".lock"):
        with pytest.raises(BlockingIOError):
            prep.run_job(t.job)
    assert sum(t.calls.values()) == 0


def test_common_cost_corruption_is_rejected(transaction):
    t = transaction
    with pytest.raises(ValueError, match="共同初解成本"):
        prep.check_common(
            t.problem,
            {"tour": [0, 1, 2], "cost": 2.0, "cheap_seconds": 0.0, "preparation_seconds": 0.0},
        )


@pytest.mark.parametrize("change", [None, "edges", "primary", "ls", "backup", "quota", "prior"])
def test_independent_graph_audit_checks_semantics_after_rehash(change):
    problem, priors = graph_fixture()
    settings = prep.GraphSettings(
        primary_width=2, backup_width=2, ls_width=3, uniform_backup_slots=1
    )
    common = {"tour": list(range(7))}
    graphs = prep.match_graphs(problem, tuple(common["tour"]), priors, settings)
    graphs = copy.deepcopy(graphs)
    graph = graphs["ALPHA"]
    if change == "edges":
        graph["edges"][0] = [0, 3]
    elif change == "primary":
        graph["primary"][0].reverse()
    elif change == "ls":
        graph["ls"][0].reverse()
    elif change == "backup":
        row = graph["backup"][0]
        row[-1] = next(j for j in range(7) if j not in {0, *row, *graph["primary"][0]})
    elif change == "quota":
        graph["matched_actual_slots"]["ls"][0] -= 1
    elif change == "prior":
        graph["prior_sha256"] = "b" * 64
    graph["sha256"] = content_hash({k: v for k, v in graph.items() if k != "sha256"})
    if change is None:
        assert (
            _audit.verify_graphs(problem, common, priors, graphs, asdict(settings))[
                "edges_per_graph"
            ]
            == 8
        )
    else:
        with pytest.raises(AssertionError):
            _audit.verify_graphs(problem, common, priors, graphs, asdict(settings))
