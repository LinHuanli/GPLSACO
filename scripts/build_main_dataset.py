#!/usr/bin/env python3
"""完整审计500/1K主池并生成分组划分，不运行测试集上的求解器。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.dataset_index import build_index, select_main_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT.parent / "Datasets/TSP")
    parser.add_argument(
        "--inventory", type=Path, default=PROJECT / "provenance/data_inventory.json"
    )
    parser.add_argument("--policy", type=Path, default=PROJECT / "configs/data_split_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/data/main-index-v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(PROJECT):
        parser.error("所有索引和临时分片必须在 GPLSACO 内")
    inventory = json.loads(args.inventory.read_text())
    policy = json.loads(args.policy.read_text())
    files = select_main_files(inventory, policy["scales"])
    manifest = build_index(args.root, files, args.output, policy, args.workers)
    print(
        json.dumps(
            {"validated_records": manifest["validated_records"], "groups": manifest["group_audit"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
