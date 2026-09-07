"""逐记录审计、标签隔离索引和确定性的点集分组划分。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import multiprocessing
import sqlite3
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from gp_faco.data import Instance, Label, parse_record, point_set_hash

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
                   dimension INTEGER NOT NULL, source_split TEXT NOT NULL);
CREATE TABLE records(record_id TEXT PRIMARY KEY, path TEXT NOT NULL, row INTEGER NOT NULL,
                     byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL,
                     dimension INTEGER NOT NULL, point_set_sha256 TEXT NOT NULL,
                     raw_sha256 TEXT NOT NULL, UNIQUE(path,row));
CREATE TABLE labels(record_id TEXT PRIMARY KEY, cost REAL NOT NULL, tour_sha256 TEXT NOT NULL,
                    declared_status TEXT NOT NULL, certificate_status TEXT NOT NULL);
CREATE TABLE assignments(point_set_sha256 TEXT PRIMARY KEY, split TEXT NOT NULL,
                         representative_id TEXT UNIQUE NOT NULL, dimension INTEGER NOT NULL);
CREATE INDEX records_point_set ON records(point_set_sha256);
CREATE INDEX assignments_split ON assignments(split,dimension);
"""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_id(file_sha256: str, physical_row: int) -> str:
    return hashlib.sha256(bytes.fromhex(file_sha256) + struct.pack("<Q", physical_row)).hexdigest()


def source_role(path: str) -> str:
    return {"train_dataset": "train", "val_dataset": "validation", "test_dataset": "test"}[
        path.split("/", 1)[0]
    ]


def select_main_files(inventory: dict, scales: list[int]) -> list[dict]:
    """明确主研究输入；cluster/Gaussian/10K/TSPLIB 留在迁移协议。"""
    selected = []
    for item in inventory["files"]:
        path = item["path"]
        if item.get("dimension") not in scales or "/tsplib/" in path or " copy" in path:
            continue
        filename = Path(path).name
        if "uniform" in filename or (path.startswith("test_dataset/") and "concorde" in filename):
            selected.append(item)
    return sorted(selected, key=lambda item: item["path"])


def audit_partition(task: tuple[str, dict, str]) -> dict:
    """每个进程只写自己的分片；字节指纹和所有标签路线一次扫描完成。"""
    root_text, metadata, staging_text = task
    source = Path(root_text) / metadata["path"]
    staging = Path(staging_text)
    filename = hashlib.sha256(metadata["path"].encode()).hexdigest() + ".jsonl"
    target = staging / filename
    partial = target.with_suffix(".partial")
    before = source.stat()
    digest = hashlib.sha256()
    count, offset, blank = 0, 0, 0
    with source.open("rb") as stream, partial.open("w", encoding="utf-8") as output:
        for row, line in enumerate(stream):
            digest.update(line)
            if not line.strip():
                blank += 1
                offset += len(line)
                continue
            record_id = stable_id(metadata["sha256"], row)
            try:
                instance, label = parse_record(line.decode("utf-8"), record_id)
                if instance.dimension != metadata["dimension"]:
                    raise ValueError("记录维数与文件声明不一致")
                if not math.isfinite(label.cost) or label.cost <= 0:
                    raise ValueError("标签成本必须是正有限值")
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{metadata['path']} row={row}: {error}") from error
            tour_bytes = struct.pack(f"<{instance.dimension}I", *label.tour)
            output.write(
                canonical_json(
                    {
                        "record": [
                            record_id,
                            metadata["path"],
                            row,
                            offset,
                            len(line),
                            instance.dimension,
                            point_set_hash(instance),
                            hashlib.sha256(line).hexdigest(),
                        ],
                        "label": [
                            record_id,
                            label.cost,
                            hashlib.sha256(tour_bytes).hexdigest(),
                            label.declared_status,
                            label.certificate_status,
                        ],
                    }
                )
                + "\n"
            )
            count += 1
            offset += len(line)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"扫描期间源文件改变: {source}")
    if digest.hexdigest() != metadata["sha256"] or offset != metadata["bytes"]:
        raise ValueError(f"文件指纹与已登记快照不符: {source}")
    if count != metadata["instances"]:
        raise ValueError(f"实例数与文件快照不符: {source}")
    partial.replace(target)
    return {
        "path": metadata["path"],
        "partition": str(target),
        "validated_records": count,
        "blank_rows": blank,
        "sha256": digest.hexdigest(),
    }


def assign_groups(connection: sqlite3.Connection, policy: dict) -> dict:
    """先锁定已暴露开发点集与原测试/验证组；其余仅按 seed/hash 排序。"""
    groups: dict[str, list[dict]] = {}
    for row in connection.execute(
        "SELECT r.record_id,r.point_set_sha256,r.path,r.row,r.dimension,f.source_split "
        "FROM records r JOIN files f USING(path) ORDER BY r.path,r.row"
    ):
        record_id, group_id, path, physical_row, dimension, role = row
        groups.setdefault(group_id, []).append(
            {
                "id": record_id,
                "path": path,
                "row": physical_row,
                "dimension": dimension,
                "source": role,
            }
        )
    anchors = {(a["path"], a["row"]) for a in policy["development_anchors"]}
    seen_anchors = {(r["path"], r["row"]) for members in groups.values() for r in members} & anchors
    if seen_anchors != anchors:
        raise ValueError("预登记开发锚点不在输入中")
    assignments = {}
    exposure_conflicts = []
    summaries = {}

    def rank(group_id: str, purpose: str) -> bytes:
        key = f"gpfaco-split-v1|{policy['split_seed']}|{purpose}|{group_id}"
        return hashlib.sha256(key.encode()).digest()

    for dimension in policy["scales"]:
        eligible = {
            group_id for group_id, rows in groups.items() if rows[0]["dimension"] == dimension
        }
        development = {
            group_id
            for group_id in eligible
            if any((r["path"], r["row"]) in anchors for r in groups[group_id])
        }
        for group_id in development:
            other = sorted({r["source"] for r in groups[group_id]} - {"train"})
            if other:
                exposure_conflicts.append({"group": group_id, "quarantined_sources": other})
        testing = {
            g for g in eligible - development if any(r["source"] == "test" for r in groups[g])
        }
        validation = {
            g
            for g in eligible - development - testing
            if any(r["source"] == "validation" for r in groups[g])
        }
        available = eligible - development - testing - validation
        required = policy["development_groups_per_scale"] - len(development)
        if required < 0 or required > len(available):
            raise ValueError("开发组数量不满足策略")
        development.update(sorted(available, key=lambda g: rank(g, "development"))[:required])
        available -= development
        desired_validation = policy["validation_groups_per_scale"]
        if len(validation) > desired_validation:
            # 不静默丢弃已有独立验证组，也不把它们转回训练。
            raise ValueError("验证组比预登记目标多；需要显式更新划分策略")
        needed = desired_validation - len(validation)
        if needed > len(available):
            raise ValueError("独立训练组不足以补足验证池")
        supplement = set(sorted(available, key=lambda g: rank(g, "validation"))[:needed])
        validation |= supplement
        training = available - validation
        by_split = {
            "development": development,
            "validation": validation,
            "test": testing,
            "train": training,
        }
        if set().union(*by_split.values()) != eligible or sum(map(len, by_split.values())) != len(
            eligible
        ):
            raise RuntimeError("划分未覆盖所有点集或存在跨划分交集")
        summaries[str(dimension)] = {
            "groups": {k: len(v) for k, v in by_split.items()},
            "validation_supplement_from_training": len(supplement),
        }
        for role, group_ids in by_split.items():
            for group_id in group_ids:
                members = groups[group_id]
                preferred_source = "train" if role in ("development", "train") else role
                representative = min(
                    members, key=lambda r: (r["source"] != preferred_source, r["path"], r["row"])
                )
                assignments[group_id] = (group_id, role, representative["id"], dimension)
    if len(assignments) != len(groups):
        raise ValueError("输入中存在策略未覆盖的维数")
    connection.executemany("INSERT INTO assignments VALUES(?,?,?,?)", sorted(assignments.values()))
    return {
        "counts": summaries,
        "exposure_conflicts": exposure_conflicts,
        "duplicate_point_groups": sum(len(v) > 1 for v in groups.values()),
        "duplicate_records_excluded": sum(len(v) - 1 for v in groups.values()),
        "source_role_overlaps": [
            {"point_set_sha256": g, "members": members}
            for g, members in groups.items()
            if len({r["source"] for r in members}) > 1
        ],
        "parentage_limitation": policy["unknown_parentage"],
    }


def build_index(
    root: Path, files: list[dict], output: Path, policy: dict, workers: int = 1
) -> dict:
    """用内核锁保护一个构建目录，锁文件存在本身不被视为作业存活。"""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("此输出目录正由另一个构建进程持有") from error
        return _build_index(root, files, output, policy, workers)


def _build_index(root: Path, files: list[dict], output: Path, policy: dict, workers: int) -> dict:
    if not files or workers < 1:
        raise ValueError("需要有效输入文件和正数 workers")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(files, key=lambda f: f["path"])
    database = output / "instances.sqlite"
    if database.exists():
        raise FileExistsError("目标索引已经发布；使用新版本目录，不能覆盖现有实验身份")
    staging = output / "partitions"
    staging.mkdir(exist_ok=True)
    tasks = [(str(root.resolve()), metadata, str(staging.resolve())) for metadata in files]
    parts = []
    if workers == 1:
        for task in tasks:
            result = audit_partition(task)
            print(f"validated {result['path']}: {result['validated_records']}", flush=True)
            parts.append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(audit_partition, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                print(f"validated {result['path']}: {result['validated_records']}", flush=True)
                parts.append(result)
    temporary = output / "instances.building.sqlite"
    if temporary.exists():
        raise FileExistsError("存在未发布索引；先核查原作业和文件状态")
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(SCHEMA)
        with connection:
            connection.executemany(
                "INSERT INTO files VALUES(?,?,?,?,?)",
                (
                    (f["path"], f["sha256"], f["bytes"], f["dimension"], source_role(f["path"]))
                    for f in files
                ),
            )
            for part in sorted(parts, key=lambda p: p["path"]):
                with Path(part["partition"]).open() as stream:
                    for line in stream:
                        value = json.loads(line)
                        connection.execute(
                            "INSERT INTO records VALUES(?,?,?,?,?,?,?,?)", value["record"]
                        )
                        connection.execute("INSERT INTO labels VALUES(?,?,?,?,?)", value["label"])
            summary = assign_groups(connection, policy)
            connection.execute(
                "INSERT INTO metadata VALUES(?,?)", ("policy", canonical_json(policy))
            )
            connection.execute("INSERT INTO metadata VALUES(?,?)", ("status", "complete"))
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite 索引完整性检查失败")
        membership = {}
        for n in policy["scales"]:
            membership[str(n)] = {}
            for role in ("train", "development", "validation", "test"):
                ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT representative_id FROM assignments WHERE split=? AND dimension=? "
                        "ORDER BY representative_id",
                        (role, n),
                    )
                ]
                item = {
                    "count": len(ids),
                    "sorted_ids_sha256": hashlib.sha256(
                        ("\n".join(ids) + "\n").encode()
                    ).hexdigest(),
                }
                if role != "train":
                    item["record_ids"] = ids
                membership[str(n)][role] = item
    finally:
        connection.close()
    temporary.replace(database)
    digest = hashlib.sha256()
    with database.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    manifest = {
        "schema_version": 1,
        "policy": policy,
        "policy_sha256": hashlib.sha256(canonical_json(policy).encode()).hexdigest(),
        "files": [
            {k: f[k] for k in ("path", "sha256", "bytes", "dimension", "instances")} for f in files
        ],
        "validated_records": sum(p["validated_records"] for p in parts),
        "membership": membership,
        "group_audit": summary,
        "database_sha256": digest.hexdigest(),
        "label_policy": "separate table; supplied optimum declaration, no certificate inference",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


class IndexedDataset:
    """按审计后的偏移读取；worker 只接收 load_instance 返回的无标签对象。"""

    def __init__(self, database: Path, root: Path):
        self.connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        self.root = root
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
            "SELECT path,byte_offset,byte_length,raw_sha256 FROM records WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        path, offset, length, expected = row
        with (self.root / path).open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("实例字节与已审计身份不一致")
        return raw.decode("utf-8")

    def load_instance(self, record_id: str) -> Instance:
        # 不查询 labels 表，不把标签字段加入在线实例对象。
        coordinates = [float(v) for v in self._read_line(record_id).split("output")[0].split()]
        return Instance(record_id, tuple(zip(coordinates[::2], coordinates[1::2], strict=True)))

    def load_label(self, record_id: str) -> Label:
        # 只在外部 evaluator 中调用；再次核对标签成本/路线指纹。
        _, label = parse_record(self._read_line(record_id), record_id)
        stored = self.connection.execute(
            "SELECT cost,tour_sha256 FROM labels WHERE record_id=?", (record_id,)
        ).fetchone()
        fingerprint = hashlib.sha256(struct.pack(f"<{len(label.tour)}I", *label.tour)).hexdigest()
        if stored is None or stored != (label.cost, fingerprint):
            raise ValueError("标签与索引不符")
        return label
