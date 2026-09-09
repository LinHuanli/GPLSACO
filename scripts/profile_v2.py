#!/usr/bin/env python3
"""保留生产并行布局的独立 CUDA event 测量；计时不进入训练。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
import numpy as np  # noqa: E402
from gp_faco.baseline_policy import BaselinePolicy  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.dataset_index import IndexedDataset  # noqa: E402
from gp_faco.experiment_v2 import representative_program  # noqa: E402
from gp_faco.gpu_session import GpuSession  # noqa: E402
from gp_faco.worker import faco_ants  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--backend", choices=("exact", "fp32", "fp32_fast"), default="exact")
    args = p.parse_args()
    session = GpuSession(args.gpu, args.backend)
    rows = []
    try:
        with IndexedDataset(
            PROJECT / "artifacts/data/main-index-v1/instances.sqlite",
            PROJECT.parent / "Datasets/TSP",
        ) as data:
            for n in (500, 1000):
                problems = {
                    name: data.load_instance(name)
                    for name in data.record_ids("development", n)[96:112]
                }
                pairs = [(name, seed) for name in problems for seed in (17, 29)]
                for name, policy in [
                    (
                        "static_mne8_region3",
                        BaselinePolicy(mne_level=2, max_mne_level=2, region=3).to_dict(),
                    ),
                    ("gp_representative", representative_program().to_dict()),
                    ("gpu_faco_uniform", None),
                ]:
                    plain, _ = session.solve(problems, pairs, 100, policy)
                    engine = session.engines[n]
                    keys = np.asarray(
                        [problems[name].numeric_id for name, _ in pairs], dtype=np.uint64
                    )
                    seeds = np.asarray([seed for _, seed in pairs], dtype=np.uint64)
                    if policy is None:
                        result = engine.evaluate_faco_evaluations(
                            keys, seeds, faco_ants(n) * 100, 8, "cached", profile_events_only=True
                        )
                    else:
                        method = (
                            engine.evaluate_program_evaluations
                            if "opcode" in policy
                            else engine.evaluate_baseline_evaluations
                        )
                        result = method(
                            keys,
                            seeds,
                            faco_ants(n) * 100,
                            policy,
                            "cached",
                            profile_events_only=True,
                        )
                    assert [r["tour"] for r in result["items"]] == [
                        r["tour"] for r in plain["items"]
                    ]
                    stages = result["production_kernel_milliseconds"]
                    rows.append(
                        {
                            "dimension": n,
                            "controller": name,
                            "stages_ms": stages,
                            "plain_seconds": plain["actual_seconds"],
                            "construction_steps": result["completed_construction_steps"],
                            "ls_evaluations": result["completed_ls_evaluations"],
                            "tour_evaluations": result["total_tour_evaluations"],
                            "same_final_tours": True,
                            "iterations": 100,
                        }
                    )
                    print(json.dumps(rows[-1]), flush=True)
        atomic_json(
            PROJECT / f"results/v2/kernel_profile_{args.backend}.json",
            {"numeric_backend": args.backend, "device": session.device, "rows": rows},
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
