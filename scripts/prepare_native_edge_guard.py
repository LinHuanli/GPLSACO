#!/usr/bin/env python3
"""生成隔离的 FACO-Native-EdgeGuard 副本；保留原生版本与完整补丁证据。"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = PROJECT.parent / "references/ACO-TSP-Adaptive-Tuning"
    destination = PROJECT / ".deps/native-edge-guard"
    lock = json.loads((PROJECT / "provenance/sources.lock.json").read_text())
    for name, expected in lock["adaptive_faco"]["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"原生源文件改变，需要重新审查: {name}")
    original = (source / "src/local_search.cpp").read_text()
    needle = "                        if (cost < curr) {"
    if original.count(needle) != 1:
        raise RuntimeError("原生 3-opt 接受位置不唯一")
    guard = """
                        // GPLSACO 适配：相同无向边集合不是结构改进。
                        // 连续距离下三项浮点加法重排可能产生虚假的正 gain；
                        // 此处只排除恒等边集合，不增加阈值或改变合法移动排序。
                        const auto canonical_edge = [](uint32_t a, uint32_t b) {
                            return std::make_pair(std::min(a, b), std::max(a, b));
                        };
                        std::array<std::pair<uint32_t,uint32_t>,3> removed{{
                            canonical_edge(at_x, at_x_1), canonical_edge(at_y, at_y_1),
                            canonical_edge(at_z, at_z_1)}};
                        std::array<std::pair<uint32_t,uint32_t>,3> added{{
                            canonical_edge(e1.first, e1.second),
                            canonical_edge(e2.first, e2.second),
                            canonical_edge(e3.first, e3.second)}};
                        std::sort(removed.begin(), removed.end());
                        std::sort(added.begin(), added.end());
                        if (removed == added) continue;
"""
    modified = original.replace(needle, guard + "\n" + needle)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "src", destination / "src", dirs_exist_ok=True)
    shutil.copy2(source / "LICENSE", destination / "LICENSE")
    (destination / "src/local_search.cpp").write_text(modified)
    difference = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="a/src/local_search.cpp",
            tofile="b/src/local_search.cpp",
        )
    )
    (PROJECT / "provenance/native_edge_guard.patch").write_text(difference)
    metadata = {
        "name": "FACO-Native-EdgeGuard",
        "base": "Adaptive-Tuning snapshot pinned by sources.lock.json",
        "scope": "initialization three_opt_nn identity-edge rejection only",
        "not_unmodified_native": True,
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "modified_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "patch_sha256": hashlib.sha256(difference.encode()).hexdigest(),
    }
    (PROJECT / "provenance/native_edge_guard.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
