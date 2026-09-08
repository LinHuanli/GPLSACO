"""图目录身份、无标签任务边界、Hard外部可行性及缓存键；不初始化CUDA。"""

import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from gp_faco import worker as module
from gp_faco.data import Label, tour_cost
from gp_faco.fitness import score_panel
from gp_faco.graph_catalog import GraphCatalog
from gp_faco.graph_matching import GraphSettings, match_graphs
from gp_faco.program_ir import Program
from gp_faco.worker import SolveTask, WorkerProtocol, content_hash, coordinate_hash, file_hash
from test_graph_matching import fixture as graph_fixture


@pytest.fixture
def catalog_fixture(tmp_path):
    problem, priors = graph_fixture()
    settings = GraphSettings(primary_width=2, backup_width=2, ls_width=3, uniform_backup_slots=1)
    graphs = match_graphs(problem, tuple(range(problem.dimension)), priors, settings)
    source = tmp_path / "source.json"
    source.write_text('{"scope":"synthetic engineering"}\n')
    matched = tmp_path / "matched.json"
    matched.write_text(json.dumps({"graphs": graphs}))
    catalog = {
        "graph_catalog_version": 1,
        "graph_spec_id": 1,
        "matching_spec_id": 2,
        "settings": asdict(settings),
        "source_manifest": "source.json",
        "source_manifest_sha256": file_hash(source),
        "entries": [
            {
                "instance_id": problem.instance_id,
                "dimension": problem.dimension,
                "coordinate_sha256": coordinate_hash(problem),
                "path": "matched.json",
                "file_sha256": file_hash(matched),
                "graphs": {
                    kind: {"sha256": g["sha256"], "edges": len(g["edges"])}
                    for kind, g in graphs.items()
                },
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    return problem, settings, graphs, catalog, path


def protocol_for(fixture, monkeypatch, mode="hard", kind="ALPHA"):
    problem, settings, _, _, path = fixture
    monkeypatch.setattr(module, "PROJECT", path.parent)
    monkeypatch.setattr(module, "implementation_hash", lambda: "0" * 64)
    return WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "unit-test-device",
        "0",
        "0" * 64,
        dimensions=(problem.dimension,),
        colonies=4,
        settings=module.SolverSettings(
            ants=4,
            primary_width=settings.primary_width,
            backup_width=settings.backup_width,
            ls_width=settings.ls_width,
        ),
        constraint_mode=mode,
        graph_prior_kind=kind,
        graph_catalog_path=path.name,
        graph_catalog_sha256=file_hash(path),
    )


def make_task(problem, limit=8):
    return SolveTask(
        "graph-worker:test",
        Program((0,), (4,), feature_spec_id=2),
        (problem,),
        tuple((problem.instance_id, seed) for seed in (17, 29, 41, 53)),
        preparation_mode="cached",
        evaluation_limit_per_colony=limit,
    )


def outcome_for(task, protocol):
    problem = task.problems[0]
    tour = list(range(problem.dimension))
    limit = task.evaluation_limit_per_colony
    native = {
        "constraint_mode": protocol.constraint_mode,
        "graph_spec_id": 1,
        "graph_edges_per_colony": [8] * 4,
        "budget_seconds": None,
        "preparation_mode": "cached",
        "actual_seconds": 0.5,
        "elapsed_seconds": 0.5,
        "charged_seconds": 0,
        "overrun_seconds": 0,
        "launched_batches": limit // 4,
        "completed_batches": limit // 4,
        "discarded_batches": 0,
        "budget_kind": "search_tour_evaluations",
        "preparation_completed": True,
        "evaluation_limit_per_colony": limit,
        "completed_tour_evaluations_per_colony": limit,
        "total_tour_evaluations": limit * 4,
        "items": [
            {
                "has_incumbent": True,
                "tour": list(tour),
                "cost": tour_cost(problem, tour),
                "completed_seconds": 0.3,
            }
            for _ in range(4)
        ],
    }
    return {
        "task_id": task.task_id(protocol),
        "protocol_sha256": protocol.sha256,
        **task.controller_identity(),
        **protocol.graph_identity(task.problems),
        "occurrence_id": task.occurrence_id,
        "dimension": problem.dimension,
        "status": "completed",
        "native_result": native,
    }, {problem.instance_id: Label(tuple(tour), tour_cost(problem, tour))}


def test_graph_identity_covers_prior_constraint_and_catalog_without_changing_coordinate_key(
    catalog_fixture,
    monkeypatch,
):
    problem = catalog_fixture[0]
    protocol = protocol_for(catalog_fixture, monkeypatch)
    task = make_task(problem)
    identities = {
        task.task_id(replace(protocol, constraint_mode=mode, graph_prior_kind=kind))
        for mode in ("hard", "escape")
        for kind in ("ALPHA", "POPMUSIC")
    }
    assert len(identities) == 4
    description = task.manifest(protocol)
    assert description["graph_inputs"][0]["graph_sha256"] == catalog_fixture[2]["ALPHA"]["sha256"]
    assert description["problems"][0][1] == coordinate_hash(problem)
    assert "coordinates" not in description and "label" not in json.dumps(description)
    with pytest.raises(ValueError, match="evaluation-count"):
        replace(
            task,
            program=Program((0,), (4,)),
            budget_seconds=1.0,
            evaluation_limit_per_colony=None,
            preparation_mode="end_to_end",
        ).manifest(protocol)
    with pytest.raises(ValueError, match="坐标改变"):
        replace(
            task,
            problems=(
                replace(problem, coordinates=tuple((x + 1, y) for x, y in problem.coordinates)),
            ),
        ).manifest(protocol)
    with pytest.raises(ValueError, match="普通worker"):
        replace(protocol, constraint_mode="unrestricted")


@pytest.mark.parametrize(
    "change", ["file", "source", "version", "duplicate", "escape_path", "label"]
)
def test_catalog_rejects_modified_inputs_and_invalid_identity(catalog_fixture, change):
    _, _, _, catalog, path = catalog_fixture
    if change == "file":
        path.write_text(path.read_text() + " ")
        with pytest.raises(ValueError, match="SHA256"):
            GraphCatalog(path.parent, path.name, "0" * 64)
        return
    if change == "source":
        (path.parent / "source.json").write_text("{}")
    elif change == "version":
        catalog["matching_spec_id"] = True
    elif change == "duplicate":
        catalog["entries"].append(copy.deepcopy(catalog["entries"][0]))
    elif change == "escape_path":
        catalog["entries"][0]["path"] = "../foreign.json"
    else:
        catalog["label_path"] = "labels.json"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        GraphCatalog(path.parent, path.name, file_hash(path))


def test_graph_payload_change_and_full_digest_not_just_declared_hash(catalog_fixture):
    problem, _, graphs, catalog, path = catalog_fixture
    source = GraphCatalog(path.parent, path.name, file_hash(path))
    assert source.load_graph(problem, "ALPHA")["edges"] == graphs["ALPHA"]["edges"]
    matched = path.parent / "matched.json"
    payload = json.loads(matched.read_text())
    payload["graphs"]["ALPHA"]["backup"][0][0] = 7
    matched.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="缓存文件被修改"):
        source.load_graph(problem, "ALPHA")
    catalog["entries"][0]["file_sha256"] = file_hash(matched)
    path.write_text(json.dumps(catalog))
    source = GraphCatalog(path.parent, path.name, file_hash(path))
    with pytest.raises(ValueError, match="冻结身份"):
        source.load_graph(problem, "ALPHA")


@pytest.mark.parametrize("change", ["mode", "graph", "edges", "tour", "zero", "escape_version"])
def test_external_graph_contract_failure_keeps_whole_panel(catalog_fixture, monkeypatch, change):
    protocol = protocol_for(catalog_fixture, monkeypatch)
    task = make_task(catalog_fixture[0], limit=0 if change == "zero" else 8)
    outcome, labels = outcome_for(task, protocol)
    assert not score_panel(task, protocol, outcome, labels).failed
    result = outcome["native_result"]
    if change == "mode":
        result["constraint_mode"] = "unrestricted"
    elif change == "graph":
        outcome["graph_inputs"][0]["graph_sha256"] = "0" * 64
    elif change == "edges":
        result["graph_edges_per_colony"][0] = True
    elif change in ("tour", "zero"):
        tour = list(range(7))
        if change == "tour":
            tour[1], tour[3] = tour[3], tour[1]
        else:
            tour = tour[1:] + tour[:1]
        result["items"][0].update(tour=tour, cost=tour_cost(task.problems[0], tour))
    else:
        protocol = replace(protocol, constraint_mode="escape")
        outcome, labels = outcome_for(task, protocol)
        outcome["native_result"].update(escape_spec_id=1, escape_edge_capacity_per_ant=64)
        assert not score_panel(task, protocol, outcome, labels).failed
        outcome["native_result"]["escape_spec_id"] = True
    assert score_panel(task, protocol, outcome, labels).failed


def test_registration_preserves_coordinate_rng_key_and_reuses_only_verified_graph(
    catalog_fixture,
    monkeypatch,
):
    protocol = protocol_for(catalog_fixture, monkeypatch)
    problem = catalog_fixture[0]
    calls = []

    class Engine:
        def register_graph_problem(self, key, xy, graph):
            calls.append((key, xy.tolist(), graph))
            return {"cheap_seconds": 0.1, "preparation_seconds": 0.2}

    n = problem.dimension
    monkeypatch.setattr(module, "_protocol", protocol)
    monkeypatch.setattr(module, "_engines", {n: Engine()})
    monkeypatch.setattr(module, "_registered", {n: {}})
    monkeypatch.setattr(module, "_registration_fees", {n: {}})
    first = module._register((problem,))
    assert module._register((problem,)) == first and len(calls) == 1
    key = int(coordinate_hash(problem)[:16], 16)
    assert calls[0][0] == key and first[0] == {problem.instance_id: key}
    assert set(calls[0][2]) == {
        "graph_spec_id",
        "common_initial_tour",
        "edges",
        "primary",
        "backup",
        "ls",
    }
    module._registered[n][key] = content_hash("different-full-graph-identity")
    with pytest.raises(ValueError, match="碰撞"):
        module._register((problem,))


@pytest.mark.parametrize("foreign", [False, True])
def test_idle_wait_rechecks_foreign_processes_without_solver_deadline(
    catalog_fixture,
    monkeypatch,
    foreign,
):
    protocol = protocol_for(catalog_fixture, monkeypatch)
    replies = iter(
        [
            "",
            f"{protocol.gpu_uuid}, unit-test-device, 2, 97, 0",
            f"{protocol.gpu_uuid}, 1234" if foreign else "",
            f"{protocol.gpu_uuid}, unit-test-device, 2, 0, 0",
        ]
    )
    sleeps = []
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    if foreign:
        with pytest.raises(RuntimeError, match="已有计算进程"):
            module._wait_for_idle(protocol)
    else:
        _, observations = module._wait_for_idle(protocol)
        assert len(observations) == 2
    assert sleeps == [1]


@pytest.mark.parametrize("mode", ["hard", "escape"])
def test_graph_cache_capacity_rebuild_preserves_mode_and_clears_old_preparation(
    catalog_fixture,
    monkeypatch,
    mode,
):
    protocol = replace(
        protocol_for(catalog_fixture, monkeypatch, mode), maximum_registered_per_dimension=4
    )
    problem, calls = catalog_fixture[0], []
    n = problem.dimension

    class ReplacementEngine:
        def __init__(self, dimension, colonies, settings, constraint):
            calls.append((dimension, colonies, settings.ants, constraint))

        def register_graph_problem(self, key, xy, graph):
            assert graph["common_initial_tour"] == list(range(n))
            return {"cheap_seconds": 0.25, "preparation_seconds": 0.75}

    monkeypatch.setattr(
        module,
        "_native",
        SimpleNamespace(FacoBatchEngine=ReplacementEngine, FixedFacoSettings=SimpleNamespace),
    )
    monkeypatch.setattr(module, "_protocol", protocol)
    monkeypatch.setattr(module, "_engines", {n: object()})
    monkeypatch.setattr(module, "_registered", {n: {key: str(key) for key in range(4)}})
    monkeypatch.setattr(module, "_registration_fees", {n: {key: {"old": True} for key in range(4)}})
    monkeypatch.setattr(module, "_assigned_charges", {n: {0: (9.0, 9.0)}})
    monkeypatch.setattr(module, "_engine_generations", {n: 7})
    keys, fees = module._register((problem,))
    assert calls == [(n, 4, 4, mode)] and module._engine_generations[n] == 8
    assert set(module._registered[n]) == set(module._registration_fees[n]) == set(keys.values())
    assert module._assigned_charges[n] == {}
    assert fees == {problem.instance_id: {"cheap_seconds": 0.25, "preparation_seconds": 0.75}}
