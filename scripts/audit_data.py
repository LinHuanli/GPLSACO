#!/usr/bin/env python3
"""全量文件指纹与首条标签审计；不读取或改写外部旧协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.data import parse_record, point_set_hash  # noqa: E402


def audit_file(path: Path, root: Path) -> dict:
    """流式计算完整 hash 和行数，仅解析首条，明确验证覆盖范围。"""
    digest = hashlib.sha256()
    rows = 0
    blank_rows = 0
    first = None
    before = path.stat()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                rows += 1
                if first is None:
                    first = line
            else:
                blank_rows += 1
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"审计期间源文件变化: {path}")
    name = path.relative_to(root).as_posix()
    item = {
        "path": name,
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
        "instances": rows,
        "blank_rows": blank_rows,
        "validated_records": 0,
    }
    if first is None:
        item["error"] = "empty file"
        return item
    try:
        instance, label = parse_record(first.decode("utf-8"), f"{name}:0")
        item.update(
            dimension=instance.dimension,
            validated_records=1,
            first_record_sha256=hashlib.sha256(first).hexdigest(),
            first_point_set_sha256=point_set_hash(instance),
            first_label_cost=label.cost,
            declared_status=label.declared_status,
            certificate_status=label.certificate_status,
            distance_spec=(
                "normalized_tsplib_unresolved" if "tsplib" in path.parts else instance.distance_spec
            ),
        )
    except (ValueError, OverflowError) as error:
        item["error"] = str(error)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT.parent / "Datasets/TSP")
    parser.add_argument("--output", type=Path, default=PROJECT / "provenance/data_inventory.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("所有输出必须在 GPLSACO 内")
    # 只枚举实例文本，排除下载日志；不读取旧 manifest 中的用途限制。
    paths = sorted(
        p
        for folder in ("train_dataset", "val_dataset", "test_dataset")
        for p in (root / folder).rglob("*.txt")
    )
    if not paths:
        raise RuntimeError("没有找到数据文件")
    files = []
    for index, path in enumerate(paths, 1):
        item = audit_file(path, root)
        files.append(item)
        print(f"[{index}/{len(paths)}] {item['path']}: {item['instances']}", flush=True)
    by_hash = defaultdict(list)
    for item in files:
        by_hash[item["sha256"]].append(item["path"])
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": "../Datasets/TSP",
        "hash_mode": "sha256_full",
        "record_validation": "first_nonempty_record_per_file_only",
        "files": files,
        "duplicate_groups": [names for names in by_hash.values() if len(names) > 1],
        "total_bytes": sum(item["bytes"] for item in files),
        "raw_instances_including_duplicates": sum(item["instances"] for item in files),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(f"已写入 {output}; 错误文件数 {sum('error' in item for item in files)}")
    if any("error" in item for item in files):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
