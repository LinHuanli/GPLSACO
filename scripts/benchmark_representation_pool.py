#!/usr/bin/env python3
# ruff: noqa: E402
"""六个末代真实种群共享A5000池，按总完成时间比较个体分片。"""

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.experiment_v2 import write_once
from gp_faco.representation_campaign import RUN_IDS
from gp_faco.representation_pilot import DEFAULT_DIRECTORY as PILOT
from gp_faco.representation_pilot import PilotScheduler


def signature(member):
    raw = member["native_result"]
    return {
        "rows": [
            {k: r[k] for k in ("instance_id", "seed", "cost", "gap_percent", "tour")}
            for r in member["rows"]
        ],
        **{
            k: raw[k]
            for k in (
                "control_states",
                "completed_construction_steps",
                "completed_ls_evaluations",
                "total_tour_evaluations",
            )
        },
    }


class BenchmarkScheduler(PilotScheduler):
    def advance(self):
        repeats = range(self.config["measures"] + 1) if self.config["iterations"] <= 100 else [1]
        phases = [(chunk, repeat) for chunk in self.config["chunks"] for repeat in repeats]
        reference = self.directory / "reference_outputs.json"
        if self.config["iterations"] == 1000 and not reference.exists():
            expected = {}
            for run in RUN_IDS:
                original = read(
                    PILOT / "jobs" / f"population-{run}-training-g10-p000-c000/result.json"
                )
                expected.update({m["controller_id"]: signature(m) for m in original["members"]})
            atomic_json(reference, expected)
        done = read(self.directory / "measurements.json", {"rows": []})
        if len(done["rows"]) >= len(phases):
            self.stage = "complete"
            totals = {
                chunk: statistics.median(
                    r["wall_seconds"]
                    for r in done["rows"]
                    if r["chunk"] == chunk and r["repeat"] > 0
                )
                for chunk in self.config["chunks"]
            }
            fastest = min(totals.values())
            selected = max(chunk for chunk, seconds in totals.items() if seconds <= fastest * 1.03)
            atomic_json(
                self.directory / "performance.json",
                {
                    "status": "passed",
                    "build": self.config["build"],
                    "median_seconds": totals,
                    "selected_shard_size": selected,
                    "measurements": done["rows"],
                    "identical_across_shards": True,
                    "iterations": self.config["iterations"],
                },
            )
            atomic_json(
                self.directory / "stop_workers.json", {"reason": "performance_measurement_complete"}
            )
            return
        chunk, repeat = phases[len(done["rows"])]
        phase = f"c{chunk}-r{repeat}"
        started_path = self.directory / f"started-{phase}.json"
        if not started_path.exists():
            atomic_json(started_path, {"unix": time.time()})
        jobs = []
        for run in RUN_IDS:
            original = read(PILOT / "jobs" / f"population-{run}-training-g10-p000-c000/job.json")
            for left in range(0, len(original["controllers"]), chunk):
                jobs.append(
                    self.add(
                        "population",
                        f"{phase}-{run}-c{left:03d}",
                        **{
                            k: v
                            for k, v in original.items()
                            if k
                            not in (
                                "id",
                                "resource",
                                "kind",
                                "controllers",
                                "iterations",
                                "shard_index",
                            )
                        },
                        controllers=original["controllers"][left : left + chunk],
                        iterations=self.config["iterations"],
                        shard_index=left,
                    )
                )
        self.stage = f"benchmark_{phase}"
        if not self.finished(jobs):
            return
        wall = time.time() - read(started_path)["unix"]
        actual, gpu_seconds, total_fe = {}, 0.0, 0
        for result in self.results(jobs):
            actual.update({m["controller_id"]: signature(m) for m in result["members"]})
            gpu_seconds += result["solve_seconds"]
            total_fe += result["total_fe"]
        if reference.exists():
            if actual != read(reference):
                raise AssertionError("共享池分片改变结果或工作量")
        else:
            atomic_json(reference, actual)
        done["rows"].append(
            {
                "chunk": chunk,
                "repeat": repeat,
                "wall_seconds": wall,
                "gpu_seconds": gpu_seconds,
                "total_fe": total_fe,
            }
        )
        atomic_json(self.directory / "measurements.json", done)
        print(done["rows"][-1], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--measures", type=int, default=3)
    p.add_argument("--hosts", nargs="*", default=[])
    args = p.parse_args()
    root = args.directory.resolve()
    if not root.is_relative_to(PROJECT):
        p.error("工程产物必须位于项目内")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "requests", "runs"):
        (root / name).mkdir(exist_ok=True)
    write_once(
        root / "campaign.json",
        {
            **read(PILOT / "campaign.json"),
            "engineering": True,
            "scope": "engineering_campaign_only",
            "build": args.build,
            "chunks": args.chunks,
            "iterations": args.iterations,
            "measures": args.measures,
        },
    )
    write_once(root / "panels.json", {})
    BenchmarkScheduler(root, hosts=args.hosts).run()


if __name__ == "__main__":
    main()
