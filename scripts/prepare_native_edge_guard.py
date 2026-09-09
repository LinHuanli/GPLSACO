#!/usr/bin/env python3
"""生成隔离的 FACO-Native-EdgeGuard 副本；保留原生版本与完整补丁证据。"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT.parent / "references/FocusedACO")
    parser.add_argument("--output", type=Path, default=PROJECT / ".deps/faco-2022-continuous")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output.resolve()
    if not destination.is_relative_to(PROJECT):
        raise ValueError("适配副本必须在项目内")
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
    # 两个 2-opt 重载的四处 gain 都保持原顺序；相邻边的恒等移动直接归零。
    for neighbor, adjacent in (("b_next", "a_next"), ("b_prev", "a_prev")):
        ending = f"- instance.get_distance({adjacent}, {neighbor});"
        if modified.count(ending) != 2:
            raise RuntimeError("原始两个 2-opt 重载的位置不符")
        modified = modified.replace(
            ending,
            ending
            + "\n                    "
            + f"if ({neighbor} == a || b == {adjacent}) diff = 0;",
        )

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "src", destination / "src", dirs_exist_ok=True)
    for notice in ("LICENSE", "README.md"):
        if (source / notice).exists():
            shutil.copy2(source / notice, destination / notice)
    (destination / "src/local_search.cpp").write_text(modified)
    difference = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="a/src/local_search.cpp",
            tofile="b/src/local_search.cpp",
        )
    )
    (destination / "continuous_distance.patch").write_text(difference)
    metadata = {
        "name": "FACO-Native-EdgeGuard",
        "base": str(source),
        "scope": "continuous distance: 2-opt identity gains zero; 3-opt identity-edge rejection",
        "adaptation_revision": 2,
        "not_unmodified_native": True,
    }
    (destination / "adaptation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
