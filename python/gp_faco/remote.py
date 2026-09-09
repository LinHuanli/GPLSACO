"""只在派单和恢复边界使用的设备发现与进程终态查询。"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PYTHON = str(PROJECT / ".venv/bin/python")


def environment():
    return {
        **os.environ,
        "TMPDIR": str(PROJECT / ".tmp"),
        "CUDA_CACHE_PATH": str(PROJECT / ".cache/cuda"),
        "NO_COLOR": "1",
    }


def remote_python(host, code, arguments=()):
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", host):
        raise ValueError("非法 SSH 主机名")
    # 参数逐项 shell quote；脚本经 stdin 发送，不把 JSON 当 shell 转义。
    command = shlex.join([PYTHON, "-", *map(str, arguments)])
    value = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        input=code,
        text=True,
        capture_output=True,
        timeout=30,  # 仅远程管理查询，不是任何求解或进化截止。
        check=True,
        env=environment(),
    )
    return json.loads(value.stdout)


def local_process(pid, start_ticks=None):
    try:
        path = Path(f"/proc/{pid}")
        row = path.joinpath("stat").read_text().split(")", 1)[1].split()
        return row[0] != "Z" and (start_ticks is None or int(row[19]) == start_ticks)
    except FileNotFoundError:
        return False


def process_alive(receipt):
    if receipt["host"] in (platform.node(), "localhost"):
        return local_process(receipt["pid"], receipt.get("start_ticks"))
    try:
        return remote_python(
            receipt["host"],
            "import json,sys\n"
            f"sys.path.insert(0, {str(PROJECT / 'python')!r})\n"
            "from gp_faco.remote import local_process\n"
            "print(json.dumps(local_process(int(sys.argv[1]), "
            "None if sys.argv[2]=='None' else int(sys.argv[2]))))\n",
            (receipt["pid"], receipt.get("start_ticks")),
        )
    except (subprocess.SubprocessError, OSError, ValueError):
        # 失联不等于进程已退出；保持原任务，不在其他设备重复运行。
        return True


def process_receipt():
    pid = os.getpid()
    row = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
    return {"host": platform.node(), "pid": pid, "start_ticks": int(row[19])}


def parse_idle_a5000(output):
    result = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[0] == "IDLE" and fields[-3:] == ["NVIDIA", "RTX", "A5000"]:
            result.append({"host": fields[1], "index": int(fields[2])})
    return result


def discover_a5000(hosts=()):
    command = "/home/linbocheng/bin/gpu-free"
    raw = subprocess.check_output([command, *hosts], text=True, env=environment(), timeout=55)
    devices = parse_idle_a5000(raw)
    found = []
    for host in dict.fromkeys(d["host"] for d in devices):
        try:
            rows = remote_python(
                host,
                "import json,subprocess,os\n"
                "s=subprocess.check_output(['nvidia-smi',"
                "'--query-gpu=index,uuid,name,driver_version',"
                "'--format=csv,noheader'],text=True)\n"
                "print(json.dumps({'gpus':[r.split(', ') for r in s.strip().splitlines()],"
                "'cpus':os.cpu_count()}))\n",
            )
        except (subprocess.SubprocessError, OSError, ValueError):
            continue
        indices = {d["index"] for d in devices if d["host"] == host}
        for index, uuid, name, driver in rows["gpus"]:
            if int(index) in indices and name == "NVIDIA RTX A5000":
                found.append(
                    {
                        "host": host,
                        "index": int(index),
                        "uuid": uuid,
                        "driver": driver,
                        "host_cpus": rows["cpus"],
                    }
                )
    return found
