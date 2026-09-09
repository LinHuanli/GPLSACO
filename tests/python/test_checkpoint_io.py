"""检查点只做普通JSON读写，覆盖实际中断窗口和旧状态接续。"""

import json

import pytest
from gp_faco import checkpoint


def test_roundtrip_has_only_version_and_state(tmp_path):
    path = tmp_path / "checkpoint.json"
    state = {"generation": 12, "rng": [1, 2, 3], "pending": None, "costs": {"FE": 4096}}
    checkpoint.save_checkpoint(path, state)
    assert json.loads(path.read_text()) == {"checkpoint_version": 2, "state": state}
    assert checkpoint.load_checkpoint(path) == state


def test_old_envelope_loads_state_without_metadata_checks(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {"checkpoint_version": 1, "state": {"FE": 8192}, "historical_metadata": "ignored"}
        )
    )
    assert checkpoint.load_checkpoint(path) == {"FE": 8192}


def test_failed_replace_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    checkpoint.save_checkpoint(path, {"FE": 4096})

    def interrupted(*_args):
        raise OSError("模拟写入完成但尚未替换的中断")

    monkeypatch.setattr(checkpoint.os, "replace", interrupted)
    with pytest.raises(OSError, match="模拟"):
        checkpoint.save_checkpoint(path, {"FE": 8192})
    assert checkpoint.load_checkpoint(path) == {"FE": 4096}
    assert list(tmp_path.glob("*.partial")) == []


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"checkpoint_version":true,"state":{}}',
        '{"checkpoint_version":2,"state":{},"state":{"FE":1}}',
        '{"checkpoint_version":99,"state":{}}',
        '{"checkpoint_version":2,"state":[]}',
    ],
)
def test_invalid_json_structure_is_rejected(tmp_path, text):
    path = tmp_path / "checkpoint.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(path)


def test_json_key_conversion_must_not_discard_state(tmp_path):
    with pytest.raises(ValueError, match="重复"):
        checkpoint.save_checkpoint(tmp_path / "checkpoint.json", {1: "first", "1": "second"})
