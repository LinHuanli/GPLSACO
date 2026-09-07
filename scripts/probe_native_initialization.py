#!/usr/bin/env python3
"""诊断原生 3-opt 是否把同一无向边集合的浮点重排判为严格改进。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "references/ACO-TSP-Adaptive-Tuning/src"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "artifacts/native/initialization-probe"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("诊断产物必须在项目内")
    output.mkdir(parents=True, exist_ok=True)
    original = (SOURCE / "local_search.cpp").read_text()
    needle = "if (cost < curr) {"
    if original.count(needle) != 1:
        raise RuntimeError("源码与已审查锚点不符")
    # 仅加观察和失败出口，不修改原本的浮点计算或移动接受条件。
    instrumentation = r"""
                            auto edge = [](uint32_t a, uint32_t b) {
                                return std::make_pair(std::min(a, b), std::max(a, b));
                            };
                            std::array<std::pair<uint32_t,uint32_t>,3> removed{{
                                edge(at_x, at_x_1), edge(at_y, at_y_1), edge(at_z, at_z_1)}};
                            std::array<std::pair<uint32_t,uint32_t>,3> added{{
                                edge(e1.first,e1.second), edge(e2.first,e2.second),
                                edge(e3.first,e3.second)}};
                            std::sort(removed.begin(), removed.end());
                            std::sort(added.begin(), added.end());
                            if (removed == added) {
                                std::cerr << std::setprecision(17)
                                    << "GPFACO_PROBE same_edges=true before=" << curr
                                    << " after=" << cost << " false_gain=" << (curr-cost)
                                    << " two_opt_changes=" << two_opt_changes
                                    << " three_opt_changes=" << three_opt_changes << '\n';
                                std::exit(73);
                            }
"""
    instrumented = (
        "#include <iomanip>\n#include <cstdlib>\n#include <iostream>\n"
        + original.replace(needle, needle + instrumentation)
    )
    probe_source = output / "instrumented_local_search.cpp"
    probe_source.write_text(instrumented)
    flags = [
        "/usr/bin/g++-15",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-mavx2",
        "-fopenmp",
        "-include",
        "cstdint",
    ]
    probe_object = output / "instrumented.o"
    executable = output / "native_initialization_probe"
    objects = list((PROJECT / "build/cpu/CMakeFiles/native_faco_support.dir").rglob("*.o"))
    objects = [p for p in objects if p.name != "local_search.cpp.o"]
    wrapper = (
        PROJECT / "build/cpu/CMakeFiles/native_faco_diagnostic.dir/cpp/src/native_diagnostic.cpp.o"
    )
    if len(objects) != 4 or not wrapper.exists():
        raise RuntimeError("须先构建 CPU 原生诊断目标")
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "TMPDIR": str(PROJECT / ".tmp")}
    with (output / "build.log").open("w") as log:
        subprocess.run(
            [*flags, "-I", str(SOURCE), "-c", str(probe_source), "-o", str(probe_object)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [*flags, str(wrapper), *map(str, objects), str(probe_object), "-o", str(executable)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    task = json.loads(args.task.read_text())
    command = [str(executable), *task["command"][1:]]
    command[command.index("--results-dir") + 1] = str(output)
    with (output / "probe.log").open("w") as log:
        try:
            process = subprocess.run(
                command,
                cwd=output,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            status = process.returncode
        except subprocess.TimeoutExpired:
            status = "timeout_30s"
    evidence = [
        line for line in (output / "probe.log").read_text().splitlines() if "GPFACO_PROBE" in line
    ]
    report = {
        "purpose": "instrumented diagnosis only; not a native benchmark",
        "returncode": status,
        "confirmed_same_edge_false_improvement": status == 73,
        "evidence": evidence,
        "task": str(args.task.resolve().relative_to(PROJECT)),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
