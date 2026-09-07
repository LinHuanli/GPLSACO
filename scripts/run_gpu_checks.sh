#!/usr/bin/env bash
# 在目标主机上使用指定空闲 UUID；所有缓存、测试临时文件和日志留在项目内。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
gpu_uuid="${1:?Usage: bash scripts/run_gpu_checks.sh GPU-uuid}"
mkdir -p artifacts/gpu .tmp .cache/cuda
export TMPDIR="$project_root/.tmp"
export CUDA_CACHE_PATH="$project_root/.cache/cuda"
export PYTHONPATH="$project_root/python:$project_root/build/cuda"

# 驱动查询直接验证此时存在的进程；不用历史快照或持久锁文件判断空闲。
GPF_CHECK_UUID="$gpu_uuid" .venv/bin/python - <<'PY'
import csv
import os
import subprocess
from pathlib import Path
uuid = os.environ['GPF_CHECK_UUID']
gpu = subprocess.check_output([
    'nvidia-smi', f'--id={uuid}',
    '--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version',
    '--format=csv,noheader,nounits'], text=True)
apps = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row and row[0].strip() == uuid for row in csv.reader(apps.splitlines())):
    raise SystemExit('目标GPU已有计算进程，本次不启动')
row = next(csv.reader(gpu.splitlines()))
if int(row[2].strip()) > 1024 or int(row[3].strip()) > 5:
    raise SystemExit('目标GPU显存/利用率不满足空闲阈值')
Path('artifacts/gpu/device.csv').write_text(gpu)
print(gpu.strip())
PY
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
GP_FACO_REQUIRE_NATIVE=1 GP_FACO_REQUIRE_CUDA=1 .venv/bin/python -m pytest -q \
    --basetemp="$project_root/.tmp/pytest-cuda" > artifacts/gpu/pytest.log 2>&1
.venv/bin/python scripts/check_program_backends.py --cuda \
    --output artifacts/gpu/program-scores.json > artifacts/gpu/program-scores.log 2>&1
/opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 42 \
    .venv/bin/python scripts/check_program_backends.py --cuda --programs 32 \
    --output artifacts/gpu/memcheck-scores.json > artifacts/gpu/memcheck.log 2>&1
/opt/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 42 \
    .venv/bin/python scripts/check_program_backends.py --cuda --programs 32 \
    --output artifacts/gpu/synccheck-scores.json > artifacts/gpu/synccheck.log 2>&1
cat artifacts/gpu/pytest.log artifacts/gpu/program-scores.log
tail -1 artifacts/gpu/memcheck.log
tail -1 artifacts/gpu/synccheck.log
