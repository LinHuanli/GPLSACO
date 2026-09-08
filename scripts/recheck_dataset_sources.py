#!/usr/bin/env python3
"""正式冻结前重读主池源文件完整hash；不解析测试性能，不改写既有数据身份。"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("复核输出必须在GPLSACO内")
    output.mkdir(parents=True, exist_ok=False)
    started, checked = time.perf_counter(), []
    index = PROJECT / "artifacts/data/main-index-v1"
    manifest = json.loads((index / "manifest.json").read_text())
    for entry in manifest["files"]:
        size, digest = fingerprint(PROJECT.parent / "Datasets/TSP" / entry["path"])
        checked.append(
            {
                "path": entry["path"],
                "bytes": size,
                "sha256": digest,
                "matches": size == entry["bytes"] and digest == entry["sha256"],
            }
        )
        print(json.dumps(checked[-1]), flush=True)
    metadata = [
        PROJECT.parent / "Datasets/manifest.json",
        *sorted((PROJECT.parent / "Datasets/TSP/logs").glob("*.log")),
    ]
    metadata_checks = []
    for path in metadata:
        size, digest = fingerprint(path)
        metadata_checks.append(
            {"path": str(path.relative_to(PROJECT.parent)), "bytes": size, "sha256": digest}
        )
    _, database_hash = fingerprint(index / "instances.sqlite")
    database_matches = database_hash == manifest["database_sha256"]
    report = {
        "status": "passed"
        if all(v["matches"] for v in checked) and database_matches
        else "source_changed",
        "utc_completed": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "files": checked,
        "database_sha256": database_hash,
        "database_matches": database_matches,
        "index_manifest_sha256": fingerprint(index / "manifest.json")[1],
        "metadata_files": metadata_checks,
        "parentage": "This byte/hash check neither establishes nor excludes rotation/scaling/"
        "subsampling families. Parent metadata requires separate verification.",
        "access_scope": "full bytes/hash only; no solver, score or test performance access",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "files": len(checked),
                "bytes": sum(v["bytes"] for v in checked),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
