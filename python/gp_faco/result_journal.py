"""任务结果追加日志；按字节位置恢复，不计算内容摘要。"""

from __future__ import annotations

import json
import math
from pathlib import Path


def portable_outcome(value):
    """只在出现非有限返回时遍历诊断数据，保存明确的非法浮点标记。"""
    if type(value) is float and not math.isfinite(value):
        return {"invalid_float": repr(value)}
    if isinstance(value, dict):
        return {key: portable_outcome(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_outcome(item) for item in value]
    return value


class ResultJournal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a+b")

    def append(self, record: dict) -> int:
        try:
            payload = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except ValueError:
            # 正常种群直接使用 C 编码器，不先递归复制数百万个路线节点。
            # 非有限诊断仍保留，不能因序列化异常丢失已经完成的 GPU 结果。
            payload = json.dumps(
                portable_outcome(record), ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        self.stream.seek(0, 2)
        offset = self.stream.tell()
        self.stream.write((payload + "\n").encode())
        self.stream.flush()
        return offset

    def read(self, offset: int) -> dict:
        self.stream.seek(offset)
        return json.loads(self.stream.readline())

    def trailing(self, offset: int):
        """只恢复完整行。崩溃留下的末行另存，下一次追加从完整边界开始。"""
        self.stream.seek(offset)
        while True:
            start = self.stream.tell()
            line = self.stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                self.path.with_suffix(".partial").write_bytes(line)
                self.stream.truncate(start)
                self.stream.flush()
                break
            yield start, json.loads(line)

    @property
    def position(self) -> int:
        self.stream.seek(0, 2)
        return self.stream.tell()

    def close(self):
        self.stream.close()
