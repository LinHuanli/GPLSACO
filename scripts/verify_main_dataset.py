#!/usr/bin/env python3
"""复核主池索引身份、无标签划分重放和抽取读取，发布不含标签的 split manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.data import point_set_hash  # noqa: E402
from gp_faco.dataset_index import (  # noqa: E402
    IndexedDataset,
    assign_groups,
    canonical_json,
    select_main_files,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(index: Path, root: Path, policy_path: Path, inventory_path: Path) -> tuple[dict, dict]:
    manifest = json.loads((index / "manifest.json").read_text())
    policy = json.loads(policy_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    require(manifest["policy"] == policy, "索引策略与登记配置不符")
    require(
        manifest["policy_sha256"] == hashlib.sha256(canonical_json(policy).encode()).hexdigest(),
        "策略指纹不符",
    )
    files = select_main_files(inventory, policy["scales"])
    expected_files = [
        {k: f[k] for k in ("path", "sha256", "bytes", "dimension", "instances")} for f in files
    ]
    require(manifest["files"] == expected_files, "索引文件清单与来源登记不符")
    require(file_hash(index / "instances.sqlite") == manifest["database_sha256"], "数据库指纹不符")
    roundtrips = {}
    with IndexedDataset(index / "instances.sqlite", root) as dataset:
        connection = dataset.connection
        require(connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "索引损坏")
        require(
            connection.execute("SELECT value FROM metadata WHERE key='policy'").fetchone()
            == (canonical_json(policy),),
            "数据库策略与 manifest 不符",
        )
        record_count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        require(
            record_count == manifest["validated_records"] == sum(f["instances"] for f in files),
            "完整核验数量不符",
        )
        # 检查完整覆盖、标签一一对应及代表记录归属，不能只看四个池的总数。
        for query in (
            "SELECT COUNT(*) FROM records r LEFT JOIN labels l USING(record_id) "
            "WHERE l.record_id IS NULL",
            "SELECT COUNT(*) FROM labels l LEFT JOIN records r USING(record_id) "
            "WHERE r.record_id IS NULL",
            "SELECT COUNT(*) FROM records r LEFT JOIN assignments a USING(point_set_sha256) "
            "WHERE a.point_set_sha256 IS NULL",
            "SELECT COUNT(*) FROM assignments a LEFT JOIN records r "
            "ON a.representative_id=r.record_id WHERE r.record_id IS NULL "
            "OR a.point_set_sha256!=r.point_set_sha256 OR a.dimension!=r.dimension",
        ):
            require(connection.execute(query).fetchone()[0] == 0, "索引存在孤立或错配记录")
        groups = dict(connection.execute("SELECT point_set_sha256,split FROM assignments"))
        require(
            record_count - len(groups) == manifest["group_audit"]["duplicate_records_excluded"],
            "重复记录计数不符",
        )
        # 在私有内存副本重放；SQLite 授权器禁止读取 labels，确保实际排名不看答案。
        replay = sqlite3.connect(":memory:")
        try:
            connection.backup(replay)
            replay.execute("DELETE FROM assignments")
            replay.set_authorizer(
                lambda action, table, *_: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_READ and table == "labels"
                    else sqlite3.SQLITE_OK
                )
            )
            summary = assign_groups(replay, policy)
            require(summary == manifest["group_audit"], "分组审计重放不一致")
            original = connection.execute("SELECT * FROM assignments ORDER BY 1").fetchall()
            require(
                replay.execute("SELECT * FROM assignments ORDER BY 1").fetchall() == original,
                "无标签划分重放与完整成员列表不同",
            )
        finally:
            replay.close()
        for dimension in policy["scales"]:
            roundtrips[str(dimension)] = {}
            for split in ("development", "validation", "test", "train"):
                ids = dataset.record_ids(split, dimension)
                entry = manifest["membership"][str(dimension)][split]
                digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
                require(
                    len(ids) == entry["count"] and digest == entry["sorted_ids_sha256"],
                    "成员数量或指纹不符",
                )
                if split != "train":
                    require(ids == entry["record_ids"], "公开的实例 ID 不符")
                # 所有开发/验证/测试记录，加每规模128条训练记录；只做格式/标签审计。
                selected = ids[:128] if split == "train" else ids
                for record_id in selected:
                    instance = dataset.load_instance(record_id)
                    label = dataset.load_label(record_id)
                    expected_group = connection.execute(
                        "SELECT point_set_sha256 FROM records WHERE record_id=?", (record_id,)
                    ).fetchone()[0]
                    require(instance.dimension == dimension, "读取维数不符")
                    require(point_set_hash(instance) == expected_group, "偏移读取的点集不符")
                    require(
                        not hasattr(instance, "tour") and not hasattr(instance, "cost"),
                        "在线对象含标签字段",
                    )
                    require(label.cost > 0, "标签成本无效")
                roundtrips[str(dimension)][split] = len(selected)
    evidence = {
        "status": "passed",
        "source_files_fully_audited": len(files),
        "records_fully_parsed_in_build": record_count,
        "point_groups": len(groups),
        "no_label_split_replay_all_groups": True,
        "index_raw_offset_roundtrips": roundtrips,
        "database_sha256": manifest["database_sha256"],
        "source_inventory_sha256": file_hash(inventory_path),
        "group_audit": manifest["group_audit"],
        "performance_unblinding": False,
        "parentage_limitation": policy["unknown_parentage"],
    }
    return manifest, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=PROJECT / "artifacts/data/main-index-v1")
    parser.add_argument("--root", type=Path, default=PROJECT.parent / "Datasets/TSP")
    parser.add_argument("--policy", type=Path, default=PROJECT / "configs/data_split_v1.json")
    parser.add_argument(
        "--inventory", type=Path, default=PROJECT / "provenance/data_inventory.json"
    )
    parser.add_argument("--publish", type=Path, default=PROJECT / "provenance/splits.v1.json")
    args = parser.parse_args()
    if not all(p.resolve().is_relative_to(PROJECT) for p in (args.index, args.publish)):
        parser.error("索引和发布输出必须在项目内")
    manifest, evidence = verify(args.index, args.root, args.policy, args.inventory)
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.publish.exists() and args.publish.read_text() != serialized:
        raise FileExistsError("已发布划分不同，不能覆盖；请显式建立新版本")
    args.publish.parent.mkdir(parents=True, exist_ok=True)
    args.publish.write_text(serialized)
    evidence["published_manifest_sha256"] = file_hash(args.publish)
    (args.index / "verification.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
