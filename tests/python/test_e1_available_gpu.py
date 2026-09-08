"""硬件补充只能改变设备身份，不能缩减共同搜索配置或跳过兼容记录核对。"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import research_e1_available_gpu as available  # noqa: E402
from gp_faco.worker import SolverSettings, WorkerProtocol, content_hash, file_hash  # noqa: E402


@pytest.fixture
def hardware_fixture(tmp_path):
    config = json.loads((PROJECT / "configs/e1_protocol_v1.json").read_text())
    plan = {"sha256": "1" * 64, "native_binary_sha256": "0" * 64, "config": config}
    report = {
        "status": "passed",
        "gpu_model": "NVIDIA RTX PRO 5000 Blackwell",
        "driver_version": "610.43.02",
        "native_binary_sha256": plan["native_binary_sha256"],
    }
    path = tmp_path / "compatibility.json"
    path.write_text(json.dumps(report))
    amendment = {
        "base_e1_plan_sha256": plan["sha256"],
        "native_binary_sha256": plan["native_binary_sha256"],
        "entrypoint_sha256": file_hash(Path(available.__file__)),
        "compatibility_report": str(path),
        "compatibility_report_sha256": file_hash(path),
    }
    amendment["sha256"] = content_hash(amendment)
    return plan, amendment


def test_hardware_change_preserves_every_solver_and_batch_setting(hardware_fixture):
    plan, amendment = hardware_fixture
    uuid = "GPU-a86d9468-9819-de61-46d7-b6b328c0f9bd"
    result = available.available_protocol(
        plan, amendment, uuid, "NVIDIA RTX PRO 5000 Blackwell, 610.43.02\n"
    )
    config = plan["config"]
    original = WorkerProtocol(
        uuid,
        config["gpu_model"],
        config["driver_version"],
        plan["native_binary_sha256"],
        dimensions=tuple(config["dimensions"]),
        colonies=config["colonies"],
        settings=SolverSettings(**config["solver"]),
        maximum_registered_per_dimension=config["maximum_registered_per_dimension"],
    )
    changed, expected = asdict(result), asdict(original)
    assert changed.pop("gpu_model") == "NVIDIA RTX PRO 5000 Blackwell"
    expected.pop("gpu_model")
    assert changed == expected


def test_unverified_hardware_or_modified_report_is_rejected(hardware_fixture):
    plan, amendment = hardware_fixture
    uuid = "GPU-a86d9468-9819-de61-46d7-b6b328c0f9bd"
    for device in ("NVIDIA RTX A6000, 610.43.02", "NVIDIA RTX PRO 5000 Blackwell, 609"):
        with pytest.raises(ValueError, match="兼容验收"):
            available.available_protocol(plan, amendment, uuid, device)
    Path(amendment["compatibility_report"]).write_text("{}")
    with pytest.raises(ValueError, match="记录改变"):
        available.available_protocol(
            plan, amendment, uuid, "NVIDIA RTX PRO 5000 Blackwell, 610.43.02"
        )


def test_amendment_binds_original_plan_binary_and_entrypoint(hardware_fixture, tmp_path):
    plan, amendment = hardware_fixture
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment))
    assert available.load_amendment(path, plan) == amendment
    for field in ("base_e1_plan_sha256", "native_binary_sha256", "entrypoint_sha256"):
        changed = {k: v for k, v in amendment.items() if k != "sha256"}
        changed[field] = "9" * 64
        changed["sha256"] = content_hash(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            available.load_amendment(path, plan)
