#!/usr/bin/env python3
"""在项目隔离目录构建LKH候选适配器；外部源只读、记录参数并保留构建日志。"""

import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / ".deps/lkh-candidates-v1")
    parser.add_argument("--cc", default="/bin/gcc-15")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    output, source = args.output.resolve(), args.source.resolve()
    if not output.is_relative_to(PROJECT) or args.jobs < 1:
        parser.error("构建及日志须位于项目内，jobs为正")
    output.mkdir(parents=True, exist_ok=False)
    contents = {"README.txt": (source / "README.txt").read_bytes()}
    contents.update(
        {
            str(p.relative_to(source)): p.read_bytes()
            for p in (source / "SRC").rglob("*")
            if p.is_file() and (p.suffix in (".c", ".h") or p.name == "Makefile")
        }
    )
    for name, data in contents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    wrapper = PROJECT / "cpp/tools/lkh_candidate_export.c"
    (output / "SRC/GPLSACOCandidates.c").write_bytes(wrapper.read_bytes())
    makefile = output / "SRC/Makefile"
    text = makefile.read_text()
    if text.count("LKHmain.o") != 1:
        raise ValueError("原生入口object位置不唯一")
    makefile.write_text(text.replace("LKHmain.o", "GPLSACOCandidates.o"))
    (output / "SRC/OBJ").mkdir()
    flags = "-std=gnu11 -O3 -Wall -IINCLUDE -DTWO_LEVEL_TREE -g -fcommon -ffp-contract=off"
    command = [
        "make",
        "-C",
        str(output / "SRC"),
        f"-j{args.jobs}",
        f"CC={args.cc}",
        f"CFLAGS={flags}",
        "LKH",
    ]
    manifest = {
        "adapter": "GPLSACO-LKH-Candidates-v1",
        "upstream_version": "3.0.13",
        "build_command": command,
        "compiler": subprocess.check_output([args.cc, "--version"], text=True),
        "license": "upstream README: research use, author reserves all rights",
        "native_candidate_algorithms_modified": False,
        "entrypoint_change": "LKHmain.o replaced by candidate-only adapter; no FindTour",
    }
    path = output / "build-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (PROJECT / ".tmp").mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(PROJECT / ".tmp")}
    with (output / "build.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    manifest["exit_code"] = result.returncode
    if result.returncode == 0:
        (output / "LKH").rename(output / "GPLSACO-LKH-Candidates")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "compiler"}))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
