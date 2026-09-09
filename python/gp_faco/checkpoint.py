"""项目内JSON检查点：记录状态并原子替换，不做内容摘要计算或校验。"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def atomic_json(path: Path, value: dict) -> None:
    target = path.resolve()
    if not target.is_relative_to(PROJECT):
        raise ValueError("checkpoint与任务产物必须在GPLSACO内")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{os.urandom(8).hex()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, state: dict) -> None:
    if type(state) is not dict:
        raise ValueError("checkpoint根状态必须为对象")
    # 统一JSON键表示，仍拒绝整数键和字符串键相撞后悄悄丢失状态。
    normalized = json.loads(json.dumps(state, allow_nan=False), object_pairs_hook=_unique_pairs)
    atomic_json(path, {"checkpoint_version": 2, "state": normalized})


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON键转换后发生重复，不能丢弃状态字段")
        value[key] = item
    return value


def load_checkpoint(path: Path) -> dict:
    target = path.resolve()
    if not target.is_relative_to(PROJECT):
        raise ValueError("仅从GPLSACO内恢复checkpoint")
    envelope = json.loads(target.read_text(), object_pairs_hook=_unique_pairs)
    if type(envelope) is not dict:
        raise ValueError("checkpoint根结构必须为对象")
    version = envelope.get("checkpoint_version")
    if (
        type(version) is not int
        or version not in (1, 2)
        or type(envelope.get("state")) is not dict
        or (version == 2 and set(envelope) != {"checkpoint_version", "state"})
    ):
        raise ValueError("checkpoint版本或结构不符")
    # 既有v1记录直接读取state，历史附加元数据不再参与恢复计算。
    return envelope["state"]
