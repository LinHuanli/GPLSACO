"""E3 候选结果复用，不重复导出；输出仍包含原有图约束所需字段。"""

import json
from pathlib import Path

from gp_faco import e3_preparation as module
from gp_faco.graph_matching import GraphSettings
from test_graph_matching import fixture


def test_prepare_catalog_uses_existing_priors_and_matching_result(tmp_path, monkeypatch):
    problem, priors = fixture()
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    for kind, prior in priors.items():
        p = tmp_path / "graphs" / problem.instance_id / kind
        p.mkdir(parents=True)
        (p / "prior.json").write_text(json.dumps(prior))

    def forbidden(*args):
        raise AssertionError("不应再次运行候选导出")

    monkeypatch.setattr(module, "prepare_candidates", forbidden)
    settings = GraphSettings(primary_width=2, backup_width=2, ls_width=3, uniform_backup_slots=1)
    first = module.prepare_catalog(
        [problem], tmp_path / "graphs", Path("unused"), lambda _: range(7), settings
    )
    second = module.prepare_catalog(
        [problem], tmp_path / "graphs", Path("unused"), forbidden, settings
    )
    assert first == second
    assert first["entries"][0]["graphs"]["ALPHA"]["edges"] == 8
