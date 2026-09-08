#!/usr/bin/env python3
"""独立重建E3完整E0/访问槽/共同初解；全部实例通过后才发布不可变图目录。"""

import argparse
import importlib
import json
import math
import multiprocessing
import os
import random
import socket
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.candidate_prior import PriorSettings, parse_candidates  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import tour_cost, validate_tour  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.graph_catalog import GraphCatalog, project_file  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402

KINDS = ("ALPHA", "POPMUSIC")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path, digest=None):
    if digest is not None:
        require(file_hash(path) == digest, f"独立摘要检查失败: {path}")
    return json.loads(path.read_text())


def verify_graphs(problem, common, priors, graphs, settings):
    """从两个原始有向排名重建完整无向图，不调用match_graphs或准备端验收。"""
    n = problem.dimension
    fingerprint = coordinate_hash(problem)
    width = min(settings["primary_width"], n - 1)
    backup_width = min(settings["backup_width"], n - 1 - width)
    ls_width = min(settings["ls_width"], n - 1)
    validate_tour(common["tour"], n)
    initial = {tuple(sorted((common["tour"][i - 1], common["tour"][i]))) for i in range(n)}
    ranks, extras, expected, neighbors = {}, {}, {}, {}
    for kind in KINDS:
        ranks[kind] = [
            {edge["to"]: rank for rank, edge in enumerate(row)} for row in priors[kind]["rows"]
        ]
        directed = {
            (i, edge["to"]) for i, row in enumerate(priors[kind]["rows"]) for edge in row[:width]
        }
        extras[kind] = {tuple(sorted(edge)) for edge in directed} - initial
    budget = min(map(len, extras.values()))
    for kind in KINDS:
        rank = ranks[kind]

        def order(edge, rank=rank):
            a, b = edge
            pair = sorted((rank[a].get(b, n), rank[b].get(a, n)))
            return *pair, a, b

        expected[kind] = sorted(initial | set(sorted(extras[kind], key=order)[:budget]))
        graph = graphs[kind]
        require(
            graph["sha256"] == content_hash({k: v for k, v in graph.items() if k != "sha256"}),
            "图内容摘要",
        )
        require(graph["edges"] == [list(e) for e in expected[kind]], "完整E0优先级或成员改变")
        require(
            graph["extra_edge_budget"] == budget
            and graph["discarded_extra_edges"] == len(extras[kind]) - budget,
            "完整E0匹配预算",
        )
        require(
            graph["common_initial_tour"] == common["tour"]
            and graph["coordinate_sha256"] == fingerprint
            and graph["dimension"] == n
            and graph["prior_kind"] == kind
            and graph["settings"] == settings
            and graph["prior_sha256"] == content_hash(priors[kind]),
            "图实例/先验/准备身份",
        )
        require(
            type(graph["graph_spec_id"]) is int
            and graph["graph_spec_id"] == 1
            and type(graph["matching_spec_id"]) is int
            and graph["matching_spec_id"] == 2,
            "图版本",
        )
        rows = [set() for _ in range(n)]
        for a, b in expected[kind]:
            rows[a].add(b)
            rows[b].add(a)
        neighbors[kind] = rows
        require(graph["degrees"] == [len(row) for row in rows], "E0度数")
        for field, size in (("primary", width), ("backup", backup_width), ("ls", ls_width)):
            require(len(graph[field]) == n, "枚举行数")
            for i, row in enumerate(graph[field]):
                valid = [v for v in row if v != n]
                require(
                    len(row) == size
                    and len(valid) == len(set(valid))
                    and all(type(v) is int and 0 <= v < n and v != i for v in valid)
                    and row == valid + [n] * (size - len(valid)),
                    "槽宽/节点合法性/哨兵",
                )
    primary_quota, ls_quota, tails_quota = [], [], []
    uniforms = min(settings["uniform_backup_slots"], backup_width)
    for i in range(n):
        p = min(width, *(len(neighbors[k][i]) for k in KINDS))
        ls_count = min(ls_width, *(len(neighbors[k][i]) for k in KINDS))
        primary_quota.append(p)
        ls_quota.append(ls_count)
        main = {
            kind: sorted(
                neighbors[kind][i], key=lambda j, kind=kind: (ranks[kind][i].get(j, n), j)
            )[:p]
            for kind in KINDS
        }
        tails = {
            kind: [
                edge["to"]
                for edge in priors[kind]["rows"][i][width:]
                if edge["to"] not in {i, *main[kind]}
            ]
            for kind in KINDS
        }
        t = min(backup_width - uniforms, *(len(row) for row in tails.values()))
        tails_quota.append(t)
        for kind in KINDS:
            graph = graphs[kind]
            require(graph["primary"][i] == main[kind] + [n] * (width - p), "主行优先级或成对配额")

            def distance(j, i=i):
                dx = problem.coordinates[i][0] - problem.coordinates[j][0]
                dy = problem.coordinates[i][1] - problem.coordinates[j][1]
                return math.sqrt(dx * dx + dy * dy), j

            ls = sorted(neighbors[kind][i], key=distance)[:ls_count]
            require(graph["ls"][i] == ls + [n] * (ls_width - ls_count), "真实FP64距离LS视图")
            tail = tails[kind][:t]
            forbidden = {i, *main[kind], *tail}
            pool = [j for j in range(n) if j not in forbidden]
            seed = int(
                content_hash(
                    {
                        "coordinates": fingerprint,
                        "seed": settings["preparation_seed"],
                        "node": i,
                    }
                )[:16],
                16,
            )
            row = tail + random.Random(seed).sample(pool, uniforms)
            require(
                graph["backup"][i] == row + [n] * (backup_width - len(row)), "备用尾部/隔离均匀流"
            )
    for graph in graphs.values():
        require(
            graph["matched_actual_slots"]
            == {
                "primary": primary_quota,
                "ls": ls_quota,
                "native_backup": tails_quota,
                "uniform_backup": [uniforms] * n,
            },
            "有效槽计数",
        )
    return {
        "nodes": n,
        "edges_per_graph": n + budget,
        "primary_slots": sum(primary_quota),
        "ls_slots": sum(ls_quota),
        "native_backup_slots": sum(tails_quota),
        "uniform_slots": n * uniforms,
    }


def verify_prior(problem, settings, directory, plan):
    """原始输出逐边重新解析，并独立核对实际LKH参数、坐标文件和零退出收据。"""
    manifest = read(directory / "manifest.json")
    resources = read(directory / "resources.json")
    require(type(resources["exit_code"]) is int and resources["exit_code"] == 0, "原生准备失败")
    require(
        manifest["binary_sha256"] == plan["candidate_binary_sha256"]
        and manifest["build_manifest_sha256"] == plan["candidate_build_manifest_sha256"]
        and manifest["module_sha256"]
        == plan["preparation_sources"]["python/gp_faco/candidate_prior.py"]
        and manifest["settings"] == settings
        and manifest["instance_id"] == problem.instance_id
        and manifest["coordinate_sha256"] == coordinate_hash(problem)
        and manifest["dimension"] == problem.dimension
        and manifest["wall_clock_limit"] is None,
        "原生准备依赖或输入身份",
    )
    require(
        file_hash(directory / "native.log") == resources["log_sha256"]
        and file_hash(directory / "parameters.par") == manifest["parameter_file_sha256"]
        and file_hash(directory / "problem.tsp") == manifest["problem_file_sha256"],
        "原生输入/日志摘要",
    )
    parameters = {}
    for line in (directory / "parameters.par").read_text().splitlines():
        key, value = (v.strip() for v in line.split("=", 1))
        require(key not in parameters, "重复LKH参数")
        parameters[key] = value
    expected = {
        "PROBLEM_FILE": "problem.tsp",
        "CANDIDATE_SET_TYPE": settings["kind"],
        "MAX_CANDIDATES": str(settings["maximum_candidates"]),
        "ASCENT_CANDIDATES": str(max(50, settings["maximum_candidates"])),
        "SEED": str(settings["seed"]),
        "SCALE": str(settings["distance_scale"]),
        "PRECISION": "1",
        "EXCESS": "1",
        "MAX_TRIALS": "1",
        "RUNS": "1",
        "INITIAL_TOUR_ALGORITHM": "WALK",
        "POPMUSIC_INITIAL_TOUR": "NO",
        "TRACE_LEVEL": "1",
        **{
            "POPMUSIC_" + key.upper(): str(settings["popmusic_" + key])
            for key in ("solutions", "sample_size", "max_neighbors", "trials")
        },
    }
    require(parameters == expected, "实际LKH参数超出冻结协议")
    lines = (directory / "problem.tsp").read_text().splitlines()
    require(
        lines[:5]
        == [
            "NAME: gpfaco_prior",
            "TYPE: TSP",
            f"DIMENSION: {problem.dimension}",
            "EDGE_WEIGHT_TYPE: EUC_2D",
            "NODE_COORD_SECTION",
        ]
        and lines[-1] == "EOF",
        "TSP头信息",
    )
    require(len(lines) == problem.dimension + 6, "TSP行数")
    for i, line in enumerate(lines[5:-1]):
        node, x, y = line.split()
        require(
            int(node) == i + 1 and (float(x), float(y)) == problem.coordinates[i], "LKH实际输入坐标"
        )
    raw = read(directory / "native-candidates.json")
    prior = parse_candidates(problem, PriorSettings(**settings), raw)
    prior.update(
        manifest_sha256=content_hash(manifest),
        native_output_sha256=file_hash(directory / "native-candidates.json"),
    )
    require(prior == read(directory / "prior.json"), "原始候选解析重放")
    return prior, resources


def audit_instance(problem, payload, plan, native, native_settings):
    require(
        payload["coordinate_sha256"] == coordinate_hash(problem)
        and payload["coordinates"] == [list(xy) for xy in problem.coordinates]
        and payload["instance_id"] == problem.instance_id
        and payload["dimension"] == problem.dimension
        and payload["distance_spec"] == problem.distance_spec
        and payload["plan_sha256"] == plan["sha256"],
        "图准备未使用数据库原始坐标",
    )
    common = payload["common"]
    replay = native.prepare_common_initial(
        np.array(problem.coordinates, np.float64), native_settings
    )
    require(
        common["tour"] == replay["tour"] and common["cost"] == replay["cost"], "共同初解原生重放"
    )
    error = abs(tour_cost(problem, common["tour"]) - common["cost"])
    require(error <= 1e-10 + 1e-12 * abs(common["cost"]), "初解独立成本重算")
    priors, costs = {}, {}
    for kind in KINDS:
        evidence = payload["prior_receipts"][kind]
        for name, digest in evidence["files"].items():
            read_path = project_file(PROJECT, name)
            require(file_hash(read_path) == digest, "原始候选收据文件变化")
        priors[kind], costs[kind] = verify_prior(
            problem,
            plan["config"]["priors"][kind],
            project_file(PROJECT, evidence["directory"]),
            plan,
        )
        require(costs[kind] == evidence["resources"], "重复资源字段不一致")
    graph_result = verify_graphs(
        problem, common, priors, payload["graphs"], plan["config"]["graph"]
    )
    return {**graph_result, "common_cost_absolute_error": error, "original_prior_resources": costs}


def verify_members(source, directory, plan, jobs):
    """独立重放MT面板和角色并集；在第一次加载坐标之前排除全部TEST成员。"""
    values = {name: read(directory / name, digest) for name, digest in plan["files"].items()}
    config, members, development = (
        plan["config"],
        values["members.json"],
        values["development.json"],
    )
    seen = set()
    for n in (500, 1000):
        for role in ("train", "validation", "test"):
            ids = source.record_ids(role, n)
            require(
                ids == members[str(n)][role]
                and len(ids) == config["data"]["expected_counts"][str(n)][role]
                and not seen.intersection(ids),
                "数据库实际split成员不同或重复",
            )
            seen.update(ids)
        ids = source.record_ids("development", n)
        require(ids == development[str(n)] and not seen.intersection(ids), "实际开发成员不同或重叠")
        seen.update(ids)
    rng, panels = random.Random(config["training"]["panel_seed"]), []
    for _ in range(50):
        row = []
        for n in (500, 1000):
            ids = sorted(rng.sample(members[str(n)]["train"], 16))
            seeds = []
            while len(seeds) != 2:
                candidate = rng.getrandbits(64)
                if candidate not in seeds:
                    seeds.append(candidate)
            row.append({"dimension": n, "ids": ids, "seeds": seeds})
        panels.append(row)
    require(panels == values["training_panels.json"], "独立50代面板重放不同")
    expected = []
    for n in (500, 1000):
        sampled = {
            identity
            for generation in panels
            for panel in generation
            if panel["dimension"] == n
            for identity in panel["ids"]
        }
        roles = {identity: ["gp_training_sampled"] for identity in sampled}
        for identity in members[str(n)]["validation"]:
            require(identity not in roles, "GP验证与训练采样重叠")
            roles[identity] = ["gp_validation"]
        for role, begin, end in (("search", 0, 64), ("validation", 64, 96)):
            for identity in development[str(n)][begin:end]:
                require(identity not in roles, "Static开发与GP主split重叠")
                roles[identity] = ["static_" + role]
        require(not set(roles).intersection(members[str(n)]["test"]), "准备坐标包含TEST")
        offset = len(expected)
        expected.extend(
            {"index": offset + i, "dimension": n, "instance_id": identity, "roles": role}
            for i, (identity, role) in enumerate(sorted(roles.items()))
        )
    require(expected == jobs, "准备任务不是固定训练采样/验证/Static开发的完整并集")


def _initialize_auditor(plan, database, dataset_root):
    global _audit_context
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "审计子进程须禁用GPU")
    native_path = project_file(PROJECT, plan["native_binary_path"])
    require(file_hash(native_path) == plan["native_binary_sha256"], "审计原生依赖改变")
    sys.path.insert(0, str(native_path.parent))
    native = importlib.import_module("gp_faco_ext")
    require(Path(native.__file__).resolve() == native_path, "审计原生模块路径")
    settings = native.FixedFacoSettings()
    for name, value in plan["config"]["solver"].items():
        setattr(settings, name, value)
    source, labels = IndexedDataset(Path(database), Path(dataset_root)), []

    def authorize(action, table, _column, _db, _trigger):
        if action == sqlite3.SQLITE_READ and table == "labels":
            labels.append(table)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    source.connection.set_authorizer(authorize)
    _audit_context = plan, source, native, settings, labels


def _audit_row(row):
    plan, source, native, settings, labels = _audit_context
    receipt = read(project_file(PROJECT, row["receipt_path"]), row["receipt_sha256"])
    require(
        receipt["sha256"] == content_hash({k: v for k, v in receipt.items() if k != "sha256"})
        and receipt["job"] == row["job"]
        and receipt["plan_sha256"] == plan["sha256"]
        and receipt["status"] == "complete"
        and receipt["mode"] == row["mode"],
        "实例完成收据",
    )
    for name, digest in receipt["files"].items():
        require(file_hash(project_file(PROJECT, name)) == digest, "实例完成文件摘要")
    payload = read(project_file(PROJECT, receipt["graph_path"]))
    require(payload["job"] == row["job"] and payload["mode"] == row["mode"], "图文件任务身份")
    problem = source.load_instance(row["job"]["instance_id"])
    result = audit_instance(problem, payload, plan, native, settings)
    require(
        receipt["graphs"]
        == {
            kind: {"sha256": graph["sha256"], "edges": len(graph["edges"])}
            for kind, graph in payload["graphs"].items()
        },
        "目录图身份字段",
    )
    entry = {
        "instance_id": problem.instance_id,
        "dimension": problem.dimension,
        "coordinate_sha256": coordinate_hash(problem),
        "path": receipt["graph_path"],
        "file_sha256": receipt["files"][receipt["graph_path"]],
        "graphs": receipt["graphs"],
    }
    return (
        entry,
        {"job": row["job"], **result},
        {"host": socket.gethostname(), "pid": os.getpid()},
        len(labels),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "inputs", "database", "dataset-root", "output", "catalog"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    for name in ("freeze", "inputs", "output", "catalog"):
        value = getattr(args, name).resolve()
        require(value.is_relative_to(PROJECT), "审计文件须在工作树")
        setattr(args, name, value)
    require(not args.output.exists() and not args.catalog.exists(), "审计和封存输出必须全新")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU独立审计须明确禁用GPU")
    require(1 <= args.processes <= len(os.sched_getaffinity(0)), "审计CPU进程数越界")
    plan = read(args.freeze / "plan.json")
    require(
        plan["sha256"] == content_hash({k: v for k, v in plan.items() if k != "sha256"}),
        "准备协议摘要",
    )
    require(file_hash(args.database) == plan["database_sha256"], "数据库摘要")
    for name, digest in plan["preparation_sources"].items():
        require(file_hash(project_file(PROJECT, name)) == digest, "准备源码改变")
    jobs = read(args.freeze / "preparation_jobs.json", plan["files"]["preparation_jobs.json"])
    manifest_path = args.inputs / "manifest.json"
    manifest = read(manifest_path)
    require(
        manifest["status"] == "complete_pending_independent_audit"
        and manifest["plan_sha256"] == plan["sha256"]
        and manifest["instances"] == len(jobs) == 2296
        and [r["job"] for r in manifest["rows"]] == jobs
        and not manifest["formal_test_released"],
        "完整准备终态",
    )
    for name, digest in manifest["sessions"].items():
        read(project_file(PROJECT, name), digest)
    native_path = project_file(PROJECT, plan["native_binary_path"])
    require(file_hash(native_path) == plan["native_binary_sha256"], "共同初解二进制摘要")
    started, entries, results, modes = time.perf_counter(), [], [], {}
    label_queries = []
    with IndexedDataset(args.database, args.dataset_root) as source:

        def authorize(action, table, _column, _db, _trigger):
            if action == sqlite3.SQLITE_READ and table == "labels":
                label_queries.append(table)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        source.connection.set_authorizer(authorize)
        verify_members(source, args.freeze, plan, jobs)
    workers = {}
    with ProcessPoolExecutor(
        max_workers=args.processes,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_auditor,
        initargs=(plan, str(args.database), str(args.dataset_root)),
    ) as pool:
        # 每组最多32个独立只读审计，不把全部实例提前排队，也不重启超时观察。
        for begin in range(0, len(jobs), 32):
            rows = manifest["rows"][begin : begin + 32]
            for row, (entry, result, worker, labels) in zip(
                rows, pool.map(_audit_row, rows), strict=True
            ):
                require(labels == 0, "审计子进程访问标签")
                entries.append(entry)
                results.append(result)
                workers[worker["pid"]] = worker
                modes[row["mode"]] = modes.get(row["mode"], 0) + 1
            print(json.dumps({"audited": len(results), "expected": len(jobs)}), flush=True)
    require(
        modes == {"fresh_paired_preparation": 2264, "reuse_verified_development_cache": 32},
        "完整缓存/新准备计数",
    )
    catalog = {
        "graph_catalog_version": 1,
        "graph_spec_id": 1,
        "matching_spec_id": 2,
        "settings": plan["config"]["graph"],
        "entries": sorted(entries, key=lambda row: row["instance_id"]),
        "source_manifest": str(manifest_path.relative_to(PROJECT)),
        "source_manifest_sha256": file_hash(manifest_path),
    }
    # GraphCatalog构造也需通过后才保留正式名字，失败不会留下可误用的已发布目录。
    temporary = args.catalog.with_name(args.catalog.name + ".audit-candidate")
    require(not temporary.exists(), "保留的目录审计尝试不能覆写")
    atomic_json(temporary, catalog)
    checked = GraphCatalog(PROJECT, str(temporary.relative_to(PROJECT)), file_hash(temporary))
    require(len(checked._entries) == len(jobs), "worker目录完整性")
    os.rename(temporary, args.catalog)
    report = {
        "status": "passed",
        "scope": "complete non-test graph preparation audit; formal training gate pending",
        "plan_sha256": plan["sha256"],
        "input_manifest_sha256": file_hash(manifest_path),
        "catalog_path": str(args.catalog.relative_to(PROJECT)),
        "catalog_sha256": file_hash(args.catalog),
        "audit_source_sha256": file_hash(Path(__file__)),
        "instances": len(results),
        "graphs": 2 * len(results),
        "modes": modes,
        "max_common_cost_error": max(v["common_cost_absolute_error"] for v in results),
        "wall_seconds": time.perf_counter() - started,
        "rows": results,
        "workers": sorted(workers.values(), key=lambda row: row["pid"]),
        "formal_test_released": False,
        "label_queries": len(label_queries),
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "instances", "graphs", "modes", "catalog_sha256")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
