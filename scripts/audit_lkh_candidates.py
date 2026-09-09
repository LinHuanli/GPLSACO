#!/usr/bin/env python3
"""独立复核开发点集的原始LKH输出、FP64距离视图、完整重放与实际度数。"""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.candidate_prior import PriorSettings, parse_candidates  # noqa: E402
from gp_faco.checkpoint import atomic_json  # noqa: E402
from gp_faco.data import Instance  # noqa: E402
from gp_faco.worker import content_hash, coordinate_hash, file_hash  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    build_path = args.build / "build-manifest.json"
    build = json.loads(build_path.read_text())
    require(build["exit_code"] == 0, "候选记录、固定输入或实际重放身份不符")
    require(
        file_hash(args.build / "GPLSACO-LKH-Candidates") == build["binary_sha256"],
        "候选记录、固定输入或实际重放身份不符",
    )
    rows, records, edges, pairs = ([], {}, 0, 0)
    for entry in inputs["files"]:
        path = args.inputs / entry["path"]
        require(file_hash(path) == entry["input_sha256"], "候选记录、固定输入或实际重放身份不符")
        payload = json.loads(path.read_text())
        require(
            set(payload) == {"instance_id", "coordinates", "distance_spec"},
            "候选记录、固定输入或实际重放身份不符",
        )
        problem = Instance(
            payload["instance_id"],
            tuple(tuple(v) for v in payload["coordinates"]),
            payload["distance_spec"],
        )
        require(
            coordinate_hash(problem) == entry["coordinate_sha256"],
            "候选记录、固定输入或实际重放身份不符",
        )
        for kind in ("ALPHA", "POPMUSIC"):
            old = None
            for repeat in (1, 2):
                directory = args.results / f"{path.stem}-{kind}-{repeat}"
                manifest = json.loads((directory / "manifest.json").read_text())
                require(
                    manifest["coordinate_sha256"] == coordinate_hash(problem),
                    "候选记录、固定输入或实际重放身份不符",
                )
                require(
                    manifest["module_sha256"]
                    == file_hash(PROJECT / "python/gp_faco/candidate_prior.py"),
                    "候选记录、固定输入或实际重放身份不符",
                )
                require(
                    manifest["build_manifest_sha256"] == file_hash(build_path),
                    "候选记录、固定输入或实际重放身份不符",
                )
                require(
                    manifest["binary_sha256"] == build["binary_sha256"],
                    "候选记录、固定输入或实际重放身份不符",
                )
                require(
                    manifest["problem_file_sha256"] == file_hash(directory / "problem.tsp"),
                    "候选记录、固定输入或实际重放身份不符",
                )
                require(
                    manifest["parameter_file_sha256"] == file_hash(directory / "parameters.par"),
                    "候选记录、固定输入或实际重放身份不符",
                )
                require(
                    manifest["wall_clock_limit"] is None, "候选记录、固定输入或实际重放身份不符"
                )
                settings = PriorSettings(**manifest["settings"])
                require(
                    settings.kind == kind and settings.maximum_candidates == 80,
                    "候选记录、固定输入或实际重放身份不符",
                )
                raw = json.loads((directory / "native-candidates.json").read_text())
                prior = parse_candidates(problem, settings, raw)
                if old:
                    require(old == prior, "候选记录、固定输入或实际重放身份不符")
                    pairs += 1
                old = prior
                prior = {
                    **prior,
                    "manifest_sha256": content_hash(manifest),
                    "native_output_sha256": file_hash(directory / "native-candidates.json"),
                }
                require(
                    prior == json.loads((directory / "prior.json").read_text()),
                    "候选记录、固定输入或实际重放身份不符",
                )
                resource = json.loads((directory / "resources.json").read_text())
                require(resource["exit_code"] == 0, "候选记录、固定输入或实际重放身份不符")
                require(
                    resource["log_sha256"] == file_hash(directory / "native.log"),
                    "候选记录、固定输入或实际重放身份不符",
                )
                edges += prior["directed_slots"]
                records[directory.name] = {
                    p.name: file_hash(p) for p in directory.iterdir() if p.is_file()
                }
                rows.append(
                    {
                        "instance": path.stem,
                        "dimension": problem.dimension,
                        "kind": kind,
                        "repeat": repeat,
                        **{
                            k: prior[k]
                            for k in (
                                "directed_slots",
                                "undirected_edges",
                                "degree_min",
                                "degree_max",
                                "max_distance_quantization_error",
                            )
                        },
                        "wall_seconds": resource["wall_seconds"],
                        "cpu_seconds": resource["cpu_user_seconds"]
                        + resource["cpu_system_seconds"],
                    }
                )
    report = {
        "status": "passed",
        "scope": inputs["scope"],
        "instances": len(inputs["files"]),
        "calls": len(rows),
        "exact_repeat_pairs": pairs,
        "directed_edges_verified": edges,
        "build_manifest_sha256": file_hash(build_path),
        "binary_sha256": build["binary_sha256"],
        "module_sha256": file_hash(PROJECT / "python/gp_faco/candidate_prior.py"),
        "wrapper_sha256": build["wrapper_sha256"],
        "rows": rows,
        "records": records,
        "limitation": "Candidate export only. Actual POPMUSIC degrees differ from alpha; "
        "E3 must match graph cardinality/slots explicitly before comparison.",
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "instances",
                    "calls",
                    "exact_repeat_pairs",
                    "directed_edges_verified",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
