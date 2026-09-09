#!/usr/bin/env python3
# ruff: noqa: E402
"""仅运行单树/三树的三seed预实验，可后台调度全部空闲A5000。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.campaign import read
from gp_faco.remote import environment
from gp_faco.representation_pilot import BUILD, DEFAULT_DIRECTORY, PilotScheduler, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hosts", nargs="*", default=[])
    args = parser.parse_args()
    directory = args.directory.resolve()
    config = initialize(directory, args.engineering)
    if args.check_only:
        print(json.dumps(config, indent=2))
        return
    if not args.engineering:
        checks = read(PROJECT / "results/v3/representation_verification.json", {})
        if checks.get("status") != "passed" or checks.get("build") != BUILD:
            raise ValueError("正式预实验须先通过新构建的CPU/CUDA、工程流程和性能核对")
    if args.resume:
        for name in ("pause.json", "stop_workers.json"):
            (directory / name).unlink(missing_ok=True)
    if args.launch:
        command = [sys.executable, str(Path(__file__).resolve()), "--directory", str(directory)]
        if args.engineering:
            command.append("--engineering")
        if args.hosts:
            command += ["--hosts", *args.hosts]
        with (directory / "campaign.log").open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"pid": process.pid, "directory": str(directory)}))
        return
    PilotScheduler(directory, hosts=args.hosts).run()


if __name__ == "__main__":
    main()
