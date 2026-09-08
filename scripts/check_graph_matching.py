#!/usr/bin/env python3
"""复用已归档先验和共同初解，独立核验真实开发实例的逐节点有效槽位匹配。"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import Instance  # noqa: E402
from gp_faco.graph_matching import engine_graph_spec, match_graphs  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(problem, graphs, priors, previous):
    n, counts, neighbors = problem.dimension, {}, {}
    for kind, graph in graphs.items():
        require(graph["edges"] == previous[kind]["edges"], "图匹配改变了完整E0边集")
        require(
            graph["common_initial_tour"] == previous[kind]["common_initial_tour"], "共同初解改变"
        )
        require(graph["coordinate_sha256"] == coordinate_hash(problem), "坐标身份不同")
        require(
            graph["matching_spec_id"] == 2 and engine_graph_spec(graph)["graph_spec_id"] == 1,
            "匹配版本或六字段原生格式不符",
        )
        neighbors[kind] = [[] for _ in range(n)]
        for a, b in graph["edges"]:
            neighbors[kind][a].append(b)
            neighbors[kind][b].append(a)
        counts[kind] = {}
        for field, width in (("primary", 16), ("ls", 20), ("backup", 64)):
            counts[kind][field] = []
            require(len(graph[field]) == n, "枚举行数错误")
            for i, row in enumerate(graph[field]):
                valid = [j for j in row if j != n]
                require(
                    len(row) == width and row == valid + [n] * (width - len(valid)),
                    "物理宽度或右侧哨兵错误",
                )
                require(
                    len(valid) == len(set(valid))
                    and all(type(j) is int and 0 <= j < n and j != i for j in valid),
                    "枚举含重复、无效节点或自环",
                )
                if field != "backup":
                    require(set(valid) <= set(neighbors[kind][i]), "主/LS视图越过E0")
                counts[kind][field].append(len(valid))
    a, b = counts["ALPHA"], counts["POPMUSIC"]
    require(a == b, "相同节点的实际主/LS/备用槽位数不同")
    native_tail_counts = []
    for i in range(n):
        degree = min(len(rows[i]) for rows in neighbors.values())
        require(
            a["primary"][i] == min(16, degree) and a["ls"][i] == min(20, degree),
            "有效槽位未采用成对较小值",
        )
        tails = {
            kind: [
                e["to"]
                for e in priors[kind]["rows"][i][16:]
                if e["to"] not in {i, *graphs[kind]["primary"][i]}
            ]
            for kind in graphs
        }
        tail_count = min(48, *(len(v) for v in tails.values()))
        native_tail_counts.append(tail_count)
        require(a["backup"][i] == tail_count + 16, "原生尾部或均匀备用数量不符")
        for kind, graph in graphs.items():
            require(graph["backup"][i][:tail_count] == tails[kind][:tail_count], "后续先验排名改变")
            require(
                not (set(graph["backup"][i]) & set(graph["primary"][i]) - {n}), "备用与主行重复"
            )

            def distance(j, i=i):
                dx = problem.coordinates[i][0] - problem.coordinates[j][0]
                dy = problem.coordinates[i][1] - problem.coordinates[j][1]
                return math.sqrt(dx * dx + dy * dy), j

            count = a["ls"][i]
            require(
                graph["ls"][i][:count] == sorted(neighbors[kind][i], key=distance)[:count],
                "LS距离视图截断或排序不符",
            )
    return {
        "nodes": n,
        "edges_per_graph": len(graphs["ALPHA"]["edges"]),
        "actual_slot_totals_per_graph": {k: sum(v) for k, v in a.items()},
        "native_backup_range": [min(native_tail_counts), max(native_tail_counts)],
        "primary_range": [min(a["primary"]), max(a["primary"])],
        "ls_range": [min(a["ls"]), max(a["ls"])],
        "uniform_backup_per_node": 16,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    require(
        args.inputs.is_relative_to(PROJECT) and args.output.is_relative_to(PROJECT),
        "产物必须位于当前GPLSACO工作树",
    )
    old = json.loads((args.inputs / "manifest.json").read_text())
    require(old["status"] == "complete" and len(old["files"]) == 32, "原始完整32实例准备尚未结束")
    require(
        {(v["dimension"], v["index"]) for v in old["files"]}
        == {(n, i) for n in (500, 1000) for i in range(16)},
        "原始面板规模/索引不完整",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rows, started = [], time.perf_counter()
    for entry in old["files"]:
        source_path = args.inputs / entry["path"]
        require(file_hash(source_path) == entry["sha256"], "已归档坐标/初解/图文件改变")
        previous = json.loads(source_path.read_text())
        problem = Instance(
            previous["instance_id"],
            tuple(map(tuple, previous["coordinates"])),
            previous["distance_spec"],
        )
        n, i = entry["dimension"], entry["index"]
        require(problem.dimension == n, "原始实例与面板规模不符")
        priors, evidence = {}, {}
        for kind in ("ALPHA", "POPMUSIC"):
            p = args.inputs / f"prior-{n}-{i}-{kind}" / "prior.json"
            priors[kind] = json.loads(p.read_text())
            require(
                content_hash(priors[kind]) == previous["graphs"][kind]["prior_sha256"],
                "已归档先验内容改变",
            )
            evidence[str(p.relative_to(PROJECT))] = file_hash(p)
        begin = time.perf_counter()
        graphs = match_graphs(problem, tuple(previous["common"]["tour"]), priors)
        elapsed = time.perf_counter() - begin
        checked = verify(problem, graphs, priors, previous["graphs"])
        path = args.output / entry["path"]
        atomic_json(
            path,
            {
                "source_instance": str(source_path.relative_to(PROJECT)),
                "source_instance_sha256": entry["sha256"],
                "source_priors_sha256": evidence,
                "graphs": graphs,
                "matching_seconds": elapsed,
                "checked": checked,
            },
        )
        rows.append(
            {
                "dimension": n,
                "index": i,
                "path": path.name,
                "sha256": file_hash(path),
                "matching_seconds": elapsed,
                **checked,
            }
        )
        print(json.dumps({"dimension": n, "index": i, "status": "matched"}), flush=True)
    report = {
        "status": "passed",
        "scope": "cached-prior graph matching engineering; no search or labels",
        "matching_spec_id": 2,
        "native_graph_format_id": 1,
        "source_input_manifest": str((args.inputs / "manifest.json").relative_to(PROJECT)),
        "source_input_manifest_sha256": file_hash(args.inputs / "manifest.json"),
        "sources": {
            str(p.relative_to(PROJECT)): file_hash(p)
            for p in (Path(__file__), PROJECT / "python/gp_faco/graph_matching.py")
        },
        "instances": len(rows),
        "matched_nodes": sum(r["nodes"] for r in rows),
        "rows": rows,
        "matching_seconds": sum(r["matching_seconds"] for r in rows),
        "matching_and_verification_wall_seconds": time.perf_counter() - started,
        "preparation_scope": "reuse original priors and common initial tour; "
        "measured matching only; "
        "original preparation receipts retained separately, not a new full preparation measurement",
        "formal_E3_complete": False,
        "escape_implemented": False,
    }
    atomic_json(args.output / "manifest.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "instances": len(rows),
                "matched_nodes": report["matched_nodes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
