"""已返回原结果不可重跑；启动竞争接续必须有完整暂停点/日志/审计证明。"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from gp_faco.checkpoint import save_checkpoint
from gp_faco.worker import file_hash

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "startup_wait", PROJECT / "scripts/wait_e3_startup_resource.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_startup_pause_requires_no_new_worker_attempt_or_cost(tmp_path):
    directory = tmp_path / "run"
    (directory / "tasks").mkdir(parents=True)
    prefix = {
        "run_id": "r",
        "completed": {"done": "hash"},
        "costs": {"FE": 128},
        "worker_history": [{"pid": 1}],
        "active_worker": None,
        "pending": None,
    }
    save_checkpoint(tmp_path / "prefix-snapshot/checkpoint.json", prefix)
    state = copy.deepcopy(prefix)
    state["pending"] = {"key": "pending", "attempts": []}
    log, resources, audit = tmp_path / "log", tmp_path / "resources", tmp_path / "audit"
    log.write_text("RuntimeError: 指定GPU已有计算进程，worker不启动")
    resources.write_text(json.dumps({"exit_code": 1}))
    audit.write_text(json.dumps({"status": "snapshot_passed", "run_id": "r"}))
    incident = {
        "run_id": "r",
        "new_native_submissions": 0,
        "checkpoint_sha256": "checkpoint",
        "pending_key": "pending",
        "log": str(log.relative_to(PROJECT)),
        "log_sha256": file_hash(log),
        "resources": str(resources.relative_to(PROJECT)),
        "resources_sha256": file_hash(resources),
    }
    incident_path = tmp_path / "incident.json"
    incident_path.write_text(json.dumps(incident))
    request = {
        "run_id": "r",
        "startup_incident": str(incident_path.relative_to(PROJECT)),
        "startup_incident_sha256": file_hash(incident_path),
        "paused_checkpoint_sha256": "checkpoint",
        "startup_audit": str(audit.relative_to(PROJECT)),
        "startup_audit_sha256": file_hash(audit),
    }
    MODULE.verify_startup_pause(state, directory, request)
    for mutation in ("attempt", "worker", "ledger", "done", "history", "wrong_run"):
        bad = copy.deepcopy(state)
        if mutation == "attempt":
            bad["pending"]["attempts"] = [{"status": "returned"}]
        elif mutation == "worker":
            bad["active_worker"] = {"pid": 2}
        elif mutation == "ledger":
            bad["costs"]["FE"] += 128
        elif mutation == "done":
            bad["completed"]["extra"] = "hash"
        elif mutation == "history":
            bad["worker_history"].append({"pid": 2})
        else:
            bad["run_id"] = "different"
        with pytest.raises(ValueError):
            MODULE.verify_startup_pause(bad, directory, request)
    (directory / "tasks/pending.json").write_text("actual returned outcome")
    with pytest.raises(ValueError):
        MODULE.verify_startup_pause(state, directory, request)
    (directory / "tasks/pending.json").unlink()
    log.write_text("unexplained ordinary failure")
    with pytest.raises(ValueError):
        MODULE.verify_startup_pause(state, directory, request)
