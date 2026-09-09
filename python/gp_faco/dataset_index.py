"""按已登记偏移读取数据；已有实验成员编号保持原样。"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

from gp_faco.data import Instance, Label, parse_record


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def source_role(path):
    return {"train_dataset": "train", "val_dataset": "validation", "test_dataset": "test"}[
        path.split("/", 1)[0]
    ]


def select_main_files(inventory, scales):
    return sorted(
        (
            item
            for item in inventory["files"]
            if item.get("dimension") in scales
            and "/tsplib/" not in item["path"]
            and " copy" not in item["path"]
            and (
                "uniform" in Path(item["path"]).name
                or (item["path"].startswith("test_dataset/") and "concorde" in item["path"])
            )
        ),
        key=lambda item: item["path"],
    )


def build_index(root, files, output, policy, workers=1):
    """新数据仅一次顺序扫描，显式文件/行编号；已有 v1 正式划分不调用此函数。"""
    output.mkdir(parents=True, exist_ok=True)
    database = output / "instances.sqlite"
    if database.exists():
        raise FileExistsError(database)
    connection = sqlite3.connect(database)
    connection.executescript("""
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE records(record_id TEXT PRIMARY KEY,path TEXT,row INTEGER,byte_offset INTEGER,
                           byte_length INTEGER,dimension INTEGER);
      CREATE TABLE labels(record_id TEXT PRIMARY KEY,cost REAL);
      CREATE TABLE assignments(representative_id TEXT PRIMARY KEY,split TEXT,dimension INTEGER);
    """)
    members = {}
    with connection:
        for file_number, item in enumerate(sorted(files, key=lambda f: f["path"]), 1):
            offset = 0
            with (root / item["path"]).open("rb") as stream:
                for row, line in enumerate(stream):
                    if line.strip():
                        name = f"file{file_number:03d}-row{row + 1:07d}"
                        problem, label = parse_record(line.decode(), name)
                        connection.execute(
                            "INSERT INTO records VALUES(?,?,?,?,?,?)",
                            (name, item["path"], row, offset, len(line), problem.dimension),
                        )
                        connection.execute("INSERT INTO labels VALUES(?,?)", (name, label.cost))
                        members.setdefault(problem.dimension, []).append(
                            (name, source_role(item["path"]), item["path"], row)
                        )
                    offset += len(line)
        rng = random.Random(policy["split_seed"])
        anchors = {(a["path"], a["row"]) for a in policy.get("development_anchors", [])}
        for n, rows in members.items():
            training = [r[0] for r in rows if r[1] == "train" and (r[2], r[3]) not in anchors]
            rng.shuffle(training)
            development = [r[0] for r in rows if (r[2], r[3]) in anchors]
            needed = policy["development_groups_per_scale"] - len(development)
            development += training[:needed]
            training = training[needed:]
            validation = [r[0] for r in rows if r[1] == "validation" and r[0] not in development]
            needed = policy["validation_groups_per_scale"] - len(validation)
            if needed < 0 or needed > len(training):
                raise ValueError("新数据划分数量不符合预登记设置")
            validation += training[:needed]
            training = training[needed:]
            testing = [r[0] for r in rows if r[1] == "test" and r[0] not in development]
            for split, ids in (
                ("train", training),
                ("development", development),
                ("validation", validation),
                ("test", testing),
            ):
                connection.executemany(
                    "INSERT INTO assignments VALUES(?,?,?)", ((name, split, n) for name in ids)
                )
        connection.execute("INSERT INTO metadata VALUES('status','complete')")
    connection.close()
    result = {
        "schema_version": 2,
        "policy": policy,
        "records": sum(map(len, members.values())),
        "database": str(database),
        "duplicate_grouping": "not performed; use declared disjoint input",
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


class IndexedDataset:
    """按审计后的偏移读取；worker 只接收 load_instance 返回的无标签对象。"""

    def __init__(self, database: Path, root: Path):
        self.connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        self.root = root
        self._labels = {}
        self._numeric_ids = {
            row[0]: row[1] for row in self.connection.execute("SELECT record_id,rowid FROM records")
        }
        if self.connection.execute("SELECT value FROM metadata WHERE key='status'").fetchone() != (
            "complete",
        ):
            self.close()
            raise ValueError("索引未完成")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> IndexedDataset:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def record_ids(self, split: str, dimension: int) -> list[str]:
        return [
            r[0]
            for r in self.connection.execute(
                "SELECT representative_id FROM assignments WHERE split=? AND dimension=? "
                "ORDER BY representative_id",
                (split, dimension),
            )
        ]

    def _read_line(self, record_id: str) -> str:
        row = self.connection.execute(
            "SELECT path,byte_offset,byte_length FROM records WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        path, offset, length = row
        with (self.root / path).open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("实例文件在指定记录结束前截断")
        return raw.decode("utf-8")

    def load_instance(self, record_id: str) -> Instance:
        # 不查询 labels 表，不把标签字段加入在线实例对象。
        coordinates = [float(v) for v in self._read_line(record_id).split("output")[0].split()]
        return Instance(
            record_id,
            tuple(zip(coordinates[::2], coordinates[1::2], strict=True)),
            numeric_id=self._numeric_ids[record_id],
        )

    def load_label(self, record_id: str) -> Label:
        if record_id not in self._labels:
            _, label = parse_record(self._read_line(record_id), record_id)
            self._labels[record_id] = label
        return self._labels[record_id]
