"""项目内JSON事务：版本、内容摘要与同目录原子替换，不反序列化pickle。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from gp_faco.worker import PROJECT, content_hash


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
    # JSON把整数键转换为字符串；先往返到实际存储结构再计算摘要，否则排序顺序可能改变。
    normalized = json.loads(json.dumps(state, allow_nan=False), object_pairs_hook=_unique_pairs)
    atomic_json(
        path, {"checkpoint_version": 1, "sha256": content_hash(normalized), "state": normalized}
    )


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
    if (
        set(envelope) != {"checkpoint_version", "sha256", "state"}
        or type(envelope["checkpoint_version"]) is not int
        or envelope["checkpoint_version"] != 1
        or type(envelope["state"]) is not dict
        or content_hash(envelope["state"]) != envelope["sha256"]
    ):
        raise ValueError("checkpoint版本、结构或内容摘要不符")
    return envelope["state"]
