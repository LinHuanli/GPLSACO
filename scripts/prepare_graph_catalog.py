#!/usr/bin/env python3
"""从已审计匹配v2缓存建立无标签worker目录，不重做LKH或选择性能结果。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from check_matched_graph_engine import load_inputs  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import Instance  # noqa: E402
from gp_faco.graph_catalog import GraphCatalog  # noqa: E402
from gp_faco.worker import coordinate_hash, file_hash  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    if not all(p.is_relative_to(PROJECT) for p in (inputs, output)) or output.exists():
        parser.error("需要工作树内输入和全新输出文件")
    manifest, _, records = load_inputs(inputs)
    entries, problems, settings = [], [], None
    rows = {(row["dimension"], row["index"]): row for row in manifest["rows"]}
    for key, value in sorted(records.items()):
        problem = Instance(value["original"]["instance_id"], value["original"]["coordinates"])
        graphs = value["matched"]["graphs"]
        for graph in graphs.values():
            if settings is None:
                settings = graph["settings"]
            if settings != graph["settings"]:
                raise ValueError("缓存中图设置不一致")
        row = rows[key]
        entries.append(
            {
                "instance_id": problem.instance_id,
                "dimension": problem.dimension,
                "coordinate_sha256": coordinate_hash(problem),
                "path": str((inputs / row["path"]).relative_to(PROJECT)),
                "file_sha256": row["sha256"],
                "graphs": {
                    kind: {"sha256": g["sha256"], "edges": len(g["edges"])}
                    for kind, g in graphs.items()
                },
            }
        )
        problems.append(problem)
    payload = {
        "graph_catalog_version": 1,
        "graph_spec_id": 1,
        "matching_spec_id": 2,
        "settings": settings,
        "entries": sorted(entries, key=lambda row: row["instance_id"]),
        "source_manifest": str((inputs / "manifest.json").relative_to(PROJECT)),
        "source_manifest_sha256": file_hash(inputs / "manifest.json"),
    }
    atomic_json(output, payload)
    catalog = GraphCatalog(PROJECT, str(output.relative_to(PROJECT)), file_hash(output))
    for problem in problems:
        for kind in ("ALPHA", "POPMUSIC"):
            catalog.load_graph(problem, kind)
    print(
        json.dumps(
            {
                "status": "passed",
                "instances": len(problems),
                "graphs": 2 * len(problems),
                "catalog_sha256": file_hash(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
