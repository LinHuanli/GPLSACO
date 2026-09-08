"""派生协议字段重建后仍严格校验，审计修正不需要重跑已完成GPU任务。"""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from gp_faco.worker import SolverSettings, WorkerProtocol

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "behavior_worker_check", PROJECT / "scripts/check_behavior_worker.py"
)


def test_saved_protocol_derivation_remains_part_of_identity(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    protocol = WorkerProtocol(
        "GPU-056fae3f-b504-efe0-2d9d-b1186860e643",
        "test",
        "0",
        "0" * 64,
        dimensions=(500, 1000),
        colonies=32,
        settings=SolverSettings(),
    )
    saved = json.loads(json.dumps(asdict(protocol)))
    assert module.protocol_from_manifest(saved).sha256 == protocol.sha256
    for altered in (saved | {"implementation_sha256": "1" * 64}, saved | {"extra": 1}):
        with pytest.raises(ValueError):
            module.protocol_from_manifest(altered)
