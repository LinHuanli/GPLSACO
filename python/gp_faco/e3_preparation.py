"""E3 离线候选准备；普通实例编号与显式参数，线上仅首次注册时读取。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from gp_faco.candidate_prior import PriorSettings, prepare_candidates
from gp_faco.checkpoint import atomic_json
from gp_faco.graph_matching import GraphSettings, match_graphs
from gp_faco.worker import PROJECT


def prepare_catalog(
    problems, directory: Path, binary: Path, common_initial, settings: GraphSettings | None = None
):
    """common_initial 是无标签初始化函数；既有候选结果直接读取，不重跑 LKH。"""
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT):
        raise ValueError("候选目录必须在项目内")
    directory.mkdir(parents=True, exist_ok=True)
    settings = settings or GraphSettings()
    entries = []
    for problem in problems:
        location = directory / problem.instance_id
        location.mkdir(exist_ok=True)
        path = location / "matched.json"
        if path.exists():
            graphs = json.loads(path.read_text())["graphs"]
            if any(graph["settings"] != asdict(settings) for graph in graphs.values()):
                raise ValueError("已有图匹配参数不同；请使用新的普通实验目录")
        else:
            priors = {}
            for kind in ("ALPHA", "POPMUSIC"):
                prior_file = location / kind / "prior.json"
                priors[kind] = (
                    json.loads(prior_file.read_text())
                    if prior_file.exists()
                    else prepare_candidates(
                        problem, PriorSettings(kind=kind), binary, location / kind
                    )
                )
            initial = common_initial(problem)
            graphs = match_graphs(problem, tuple(initial), priors, settings)
            atomic_json(path, {"instance_id": problem.instance_id, "graphs": graphs})
        entries.append(
            {
                "instance_id": problem.instance_id,
                "dimension": problem.dimension,
                "path": str(path.relative_to(PROJECT)),
                "graphs": {
                    kind: {
                        "graph_id": f"{problem.instance_id}-{kind}",
                        "edges": len(graphs[kind]["edges"]),
                    }
                    for kind in graphs
                },
            }
        )
    catalog = {
        "graph_catalog_version": 2,
        "graph_spec_id": 1,
        "matching_spec_id": 2,
        "settings": asdict(settings),
        "entries": entries,
    }
    atomic_json(directory / "catalog.json", catalog)
    return catalog
