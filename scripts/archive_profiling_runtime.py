#!/usr/bin/env python3
"""保存/验证成本运行的源码与报告快照；已有快照只检查，不覆盖。"""

import argparse
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from gp_faco.checkpoint import load_checkpoint  # noqa: E402
from gp_faco.worker import file_hash  # noqa: E402
from summarize_profiling import check_sources, require  # noqa: E402


def archive(directory):
    manifest = load_checkpoint(directory / "manifest.json")
    runtime = directory / "runtime"
    require(file_hash(runtime / "gp_faco_ext.so") == manifest["binary_sha256"], "二进制身份不符")
    target = runtime / "source.tar"
    if not target.exists():
        # 先锁定将写入的字节再校验，避免hash后源码被修改的时间窗口。
        contents = {}
        for name, digest in manifest["sources"].items():
            path = (PROJECT / name).resolve()
            require(path.is_relative_to(PROJECT) and path.is_file(), "源文件必须位于项目内")
            contents[name] = path.read_bytes()
            require(hashlib.sha256(contents[name]).hexdigest() == digest, f"源文件改变: {name}")
        temporary = runtime / f".source-{os.getpid()}.partial"
        try:
            with tarfile.open(temporary, "x") as stream:
                for name, data in sorted(contents.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                    stream.addfile(entry, io.BytesIO(data))
            # 使用独占链接发布；另一执行器已有快照时绝不覆盖。
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    evidence = check_sources(directory, manifest)
    cfg = manifest["config"]
    if "profiling_report_sha256" in cfg:
        copy = runtime / "profiling_report.json"
        if not copy.exists():
            source = PROJECT / "docs/reports/profiling_results.json"
            data = source.read_bytes()
            require(
                hashlib.sha256(data).hexdigest() == cfg["profiling_report_sha256"],
                "预算来源报告改变",
            )
            temporary = runtime / f".report-{os.getpid()}.partial"
            try:
                with temporary.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, copy)
            finally:
                temporary.unlink(missing_ok=True)
        require(file_hash(copy) == cfg["profiling_report_sha256"], "预算来源报告快照不符")
    descriptor = os.open(runtime, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    require(directory.is_relative_to(PROJECT), "仅归档项目内运行")
    print(archive(directory))


if __name__ == "__main__":
    main()
