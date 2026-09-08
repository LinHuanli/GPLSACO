"""资源等待不提交GPU任务；只自动接续有明确边界证据的空attempt待办。"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from gp_faco.checkpoint import load_checkpoint, save_checkpoint
from gp_faco.worker import file_hash

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "e3_unsubmitted_wait", ROOT / "scripts/wait_e3_unsubmitted_resource.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def paused():
    return {
        "phase": "training",
        "run_id": "f" * 64,
        "pending": {"key": "a" * 64, "attempts": []},
        "admission": [{"task_key": "a" * 64, "foreign_processes": 1}],
        "completed": {"old": "unchanged"},
        "active_worker": {"pid": 123},
    }


def test_only_unsubmitted_resource_pause_is_eligible(tmp_path):
    state = paused()
    assert MODULE.unsubmitted_resource_pause(state, tmp_path)
    for mutation in ("returned", "running", "unknown", "zero", "wrong_task", "failed", "missing"):
        bad = copy.deepcopy(state)
        if mutation in ("returned", "running"):
            bad["pending"]["attempts"] = [{"status": mutation}]
        elif mutation in ("unknown", "zero"):
            bad["admission"][-1]["foreign_processes"] = None if mutation == "unknown" else 0
        elif mutation == "wrong_task":
            bad["admission"][-1]["task_key"] = "wrong"
        elif mutation == "failed":
            bad["phase"] = "failed"
        else:
            bad["pending"] = None
        assert not MODULE.unsubmitted_resource_pause(bad, tmp_path)
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / ("a" * 64 + ".json")).write_text("actual return")
    assert not MODULE.unsubmitted_resource_pause(state, tmp_path)


def test_idle_requires_all_strict_known_boundaries():
    free = {"processes": 0, "memory_used_mib": 4, "utilization_percent": 0}
    assert MODULE.is_idle(free)
    for update in (
        {"processes": 1},
        {"processes": None},
        {"processes": False},
        {"memory_used_mib": 1025},
        {"utilization_percent": 6},
    ):
        assert not MODULE.is_idle(free | update)


@pytest.mark.parametrize("final_phase", ["complete", "failed"])
def test_wait_and_repeat_only_explicit_pre_submission_pause(tmp_path, monkeypatch, final_phase):
    directory, output = tmp_path / "run", tmp_path / "output"
    directory.mkdir()
    output.mkdir()
    initial = paused()
    save_checkpoint(directory / "checkpoint.json", initial)
    hardware = {
        "gpu_uuid": "GPU-test",
        "execution_host": "host",
        "gpu_model": "model",
        "driver_version": "1",
    }
    save_checkpoint(directory / "manifest.json", {"worker_protocol": hardware})
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    request = {
        "request_spec_id": 1,
        "no_wall_clock_limit": True,
        "host": "host",
        "gpu_uuid": "GPU-test",
        "gpu_model": "model",
        "driver_version": "1",
        "run_id": initial["run_id"],
        "input": str(directory.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "paused_checkpoint_sha256": file_hash(directory / "checkpoint.json"),
        "prefix_audit": str(audit.relative_to(ROOT)),
        "prefix_audit_sha256": file_hash(audit),
        "execution": "artifacts/e3/execution-v1",
        "database": "database",
        "dataset_root": "dataset",
        "condition": "GP-Full",
        "evolution_seed": 1103,
        "sources": {
            "scripts/wait_e3_unsubmitted_resource.py": file_hash(
                ROOT / "scripts/wait_e3_unsubmitted_resource.py"
            )
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    observations = iter(
        [
            {"processes": 1},
            {"processes": 0, "memory_used_mib": 4, "utilization_percent": 0},
            {"processes": 0, "memory_used_mib": 4, "utilization_percent": 0},
        ]
    )
    waits, calls = [], []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setattr(MODULE.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(MODULE, "observe", lambda uuid: next(observations))
    monkeypatch.setattr(MODULE.time, "sleep", waits.append)

    class Child:
        pid = 456

        def __init__(self, command, *, cwd, stdout, stderr):
            assert waits == [5]
            calls.append(command)
            assert command[-1] == "--resume"
            assert "--stop-after-tasks" not in command
            assert str(ROOT / "scripts/research_e3.py") in command
            assert command[command.index("--execution") + 1] == str(
                ROOT / "artifacts/e3/execution-v1"
            )
            self.stdout = stdout

        def wait(self):
            if len(calls) == 1:
                # 使用原正式CLI实际返回的错误文本，避免模拟器重复等待器的拼写错误。
                self.stdout.write("ValueError: 目标GPU占用未知或存在外来进程，尚未提交当前任务\n")
                self.stdout.flush()
                return 1
            state = load_checkpoint(directory / "checkpoint.json")
            state.update(phase=final_phase, pending=None, active_worker=None)
            save_checkpoint(directory / "checkpoint.json", state)
            return 0 if final_phase == "complete" else 1

    monkeypatch.setattr(MODULE.subprocess, "Popen", Child)
    assert MODULE.run(request_path) == (0 if final_phase == "complete" else 1)
    assert len(calls) == 2
    receipt = json.loads((output / "waiting-receipt.json").read_text())
    assert receipt["native_submissions_while_waiting"] == 0
    assert len(receipt["cli_attempts"]) == 2 and receipt["waiting_samples"] == 1
    assert load_checkpoint(directory / "checkpoint.json")["completed"] == initial["completed"]
