#!/usr/bin/env python3
"""自动使用空闲 A5000，完成三次50代训练、分布式验证、FACO测试和中文报告。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.campaign import BUILD, SEEDS, initialize, read  # noqa: E402
from gp_faco.campaign_scheduler import CampaignScheduler  # noqa: E402
from gp_faco.campaign_tsp500 import BUILD as TSP500_BUILD, DEFAULT_DIRECTORY, initialize_tsp500
from gp_faco.campaign_pool import PopulationScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--legacy-two-scale", action="store_true")
    parser.add_argument("--shard-size", type=int, help="默认沿用已有轮次；新轮次使用已完成吞吐测量的选择")
    parser.add_argument("--seeds", nargs=3, type=int, default=SEEDS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--hosts", nargs="*", default=())
    parser.add_argument("--gpu-limit", type=int)
    args = parser.parse_args()
    directory = (args.directory or (PROJECT / "artifacts/v2/round-3seed" if args.legacy_two_scale else DEFAULT_DIRECTORY)).resolve()
    shard_size = args.shard_size or read(directory / "campaign.json", {}).get("population_shard_size") or \
        read(PROJECT / "results/v2/tsp500_performance.json", {}).get("selected_chunk", 32)
    build = BUILD if args.legacy_two_scale else TSP500_BUILD
    verification_path = PROJECT / ("results/v2/campaign_verification.json" if args.legacy_two_scale else "results/v2/tsp500_verification.json")
    if args.check_only:
        required = [
            PROJECT / build / "gp_faco_ext.so",
            PROJECT / "build/v2-native-continuous/faco_2022",
            PROJECT / "artifacts/data/main-index-v1/instances.sqlite",
            PROJECT / "artifacts/v2/protocol/panels.json",
        ]
        missing = [str(p) for p in required if not p.is_file()]
        verification = read(verification_path)
        print(
            json.dumps(
                {
                    "missing": missing,
                    "verification": verification,
                    "ready_to_launch": not missing
                    and bool(verification and verification["status"] == "passed" and verification.get("build") == build),
                    "seeds": args.seeds,
                    "generations": 50,
                    "population": 128,
                    "population_shard_size": shard_size,
                },
                ensure_ascii=False,
            )
        )
        return 0 if not missing and verification and verification["status"] == "passed" and verification.get("build") == build else 1
    if not args.engineering:
        verified = read(verification_path, {})
        if verified.get("status") != "passed" or verified.get("build") != build:
            parser.error("正式挂任务前需要本轮构建、调度和完整流程检查通过")
    if args.legacy_two_scale:
        initialize(directory, engineering=args.engineering, seeds=tuple(args.seeds))
        scheduler = CampaignScheduler
    else:
        initialize_tsp500(directory, engineering=args.engineering, seeds=tuple(args.seeds), shard_size=32 if args.engineering else shard_size)
        scheduler = PopulationScheduler
    result = scheduler(directory, hosts=args.hosts, gpu_limit=args.gpu_limit).run()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
