#!/usr/bin/env python3
# ruff: noqa: E402
"""单树与三树各3seed，从头50代后自动选模、冻结与TSP500/1000测试。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read
from gp_faco.remote import environment
from gp_faco.representation_campaign import DEFAULT_DIRECTORY, RepresentationScheduler, initialize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    p.add_argument("--build")
    p.add_argument("--shard-size", type=int)
    p.add_argument("--engineering", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--launch", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hosts", nargs="*", default=[])
    args = p.parse_args()
    directory = args.directory.resolve()
    existing = read(directory / "campaign.json", {})
    checks = read(PROJECT / "results/v3/representation_full_verification.json", {})
    build = args.build or existing.get("build") or checks.get("build")
    shard = args.shard_size or existing.get("population_shard_size") or checks.get("shard_size", 32)
    if not build:
        p.error("需要通过测量的构建路径")
    if not args.engineering and (
        checks.get("status") != "passed"
        or checks.get("build") != build
        or checks.get("shard_size") != shard
    ):
        raise ValueError("正式完整实验需要所选构建和分片通过语义、性能与完整工程流程检查")
    config = initialize(directory, build=build, shard_size=shard, engineering=args.engineering)
    if args.check_only:
        print(json.dumps(config, indent=2))
        return
    if args.resume:
        for name in ("pause.json", "stop_workers.json"):
            (directory / name).unlink(missing_ok=True)
    if args.launch:
        command = [sys.executable, str(Path(__file__).resolve()), "--directory", str(directory)]
        if args.engineering:
            command += ["--engineering"]
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
    RepresentationScheduler(directory, hosts=args.hosts).run()


if __name__ == "__main__":
    main()
