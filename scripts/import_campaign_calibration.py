#!/usr/bin/env python3
"""串行标定停止后，将已保存的完整曲线移交给多卡队列；不重复求解。"""

import argparse
import fcntl
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.campaign import read  # noqa: E402
from gp_faco.campaign_calibration import import_calibration  # noqa: E402
from gp_faco.remote import process_alive  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT / "artifacts/v2/round-3seed")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJECT):
        parser.error("全部运行记录必须在项目内")
    handoff = read(directory / "calibration_handoff.json")
    if not handoff or any(process_alive(handoff[k]) for k in ("legacy", "coordinator")):
        parser.error("原串行标定和协调器必须先完成移交停机，避免重复求解")
    with (directory / "campaign.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(directory / "campaign.json")
        result = import_calibration(directory, PROJECT / config["source_protocol"] / "curves/exact")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
