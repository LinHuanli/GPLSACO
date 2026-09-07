#!/usr/bin/env python3
"""锁定只读外部源快照；区分计划 commit 与实际文件匹配证据。"""

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REFERENCES = PROJECT.parent / "references"


def fingerprints(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
        "bytes": len(data),
    }


def main() -> None:
    anchor = REFERENCES / "ACO-TSP-Adaptive-Tuning"
    files = {
        p.relative_to(anchor).as_posix(): fingerprints(p)
        for p in sorted((anchor / "src").rglob("*"))
        if p.is_file()
    }
    files["LICENSE"] = fingerprints(anchor / "LICENSE")
    expected = {
        "src/faco.cpp": "a446e48ef4e4f469bf67e4303325a287dca594f5",
        "src/pheromone.h": "7ed00133a56ffdb3359cfc4bc664413e920ca3a8",
    }
    if any(files[name]["git_blob"] != value for name, value in expected.items()):
        raise RuntimeError("FACO 关键文件与 v4 不一致，须先审查差异")
    sources = {
        "schema_version": 1,
        "adaptive_faco": {
            "root": "../references/ACO-TSP-Adaptive-Tuning",
            "proposal_commit": "a904e6a8786d48593ef1cac975edef5ed8920af3",
            "local_git_metadata": (anchor / ".git").exists(),
            "verification_scope": "listed file hashes; proposal key blobs matched",
            "license": "MIT",
            "files": files,
        },
        "cuopt": {
            "proposal_commit": "0cccfd3e426f391feaadc2afd1f9ead1533ab2ac",
            "local_git_metadata": (REFERENCES / "cuopt/.git").exists(),
            "usage": "read_only_design_reference",
            "files": {
                name: fingerprints(REFERENCES / "cuopt" / name)
                for name in (
                    "LICENSE",
                    "cpp/src/routing/local_search/two_opt.cu",
                    "cpp/src/routing/route/tsp_route.cuh",
                    "cpp/src/routing/cuda_graph.cuh",
                    "cpp/src/routing/util_kernels/top_k.cuh",
                )
            },
        },
        "lkh": {
            "version": "3.0.13",
            "usage": "external research tool; build pending",
            "readme": fingerprints(REFERENCES / "LKH-3.0.13/README.txt"),
        },
        "concorde": {
            "binary": fingerprints(REFERENCES / "concorde"),
            "source_root": "../references/concorde_code",
            "usage": "external academic research tool; integration pending",
        },
        "focused_aco": {
            "local_license_file_found": (REFERENCES / "FocusedACO/LICENSE").exists(),
            "usage": "read_only_reference; not copied",
        },
        "acotsp": {"usage": "external MMAS baseline; build pending"},
    }
    (PROJECT / "provenance/sources.lock.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n"
    )
    (PROJECT / "provenance/licenses/Adaptive-Tuning-MIT.txt").write_bytes(
        (anchor / "LICENSE").read_bytes()
    )


if __name__ == "__main__":
    main()
