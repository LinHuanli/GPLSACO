#!/usr/bin/env python3
"""从仅含坐标的JSON调用LKH候选适配器；不接受标签、最优值或tour字段。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.candidate_prior import PriorSettings, prepare_candidates  # noqa: E402
from gp_faco.data import Instance  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=("ALPHA", "POPMUSIC"), required=True)
    parser.add_argument(
        "--binary", type=Path, default=PROJECT / ".deps/lkh-candidates-v1/GPLSACO-LKH-Candidates"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-candidates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=73001)
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    if set(value) != {"instance_id", "coordinates", "distance_spec"}:
        parser.error("输入必须精确包含实例身份、坐标和距离规格三个字段")
    problem = Instance(
        value["instance_id"], tuple(tuple(v) for v in value["coordinates"]), value["distance_spec"]
    )
    settings = PriorSettings(
        kind=args.kind, maximum_candidates=args.maximum_candidates, seed=args.seed
    )
    result = prepare_candidates(problem, settings, args.binary, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "dimension",
                    "settings",
                    "directed_slots",
                    "undirected_edges",
                    "degree_min",
                    "degree_max",
                    "max_distance_quantization_error",
                    "native_nodes_sha256",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
