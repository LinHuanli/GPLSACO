#!/usr/bin/env bash
# worker协议、外部fitness与Python GPU入口回归；目标卡先核查实时占用。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
gpu_uuid="${1:?Usage: bash scripts/run_worker_checks.sh GPU-uuid}"
mkdir -p artifacts/gpu/worker-engine .tmp .cache/cuda
exec 9>.tmp/worker-engine-checks.lock
flock -n 9 || { echo '此项目已有worker接口检查在运行'; exit 2; }
export TMPDIR="$project_root/.tmp"
export CUDA_CACHE_PATH="$project_root/.cache/cuda"
export PYTHONPATH="$project_root/python:$project_root/build/cuda"
GPF_CHECK_UUID="$gpu_uuid" .venv/bin/python - <<'PY'
import csv
import os
import subprocess
from pathlib import Path

uuid = os.environ['GPF_CHECK_UUID']
apps = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row and row[0].strip() == uuid for row in csv.reader(apps.splitlines())):
    raise SystemExit('目标GPU已有计算进程，本次不启动')
gpu = subprocess.check_output([
    'nvidia-smi', f'--id={uuid}',
    '--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version',
    '--format=csv,noheader,nounits'], text=True)
row = next(csv.reader(gpu.splitlines()))
if int(row[2].strip()) > 1024 or int(row[3].strip()) > 5:
    raise SystemExit('目标GPU显存/利用率不满足空闲阈值')
Path('artifacts/gpu/worker-engine/device.csv').write_text(gpu)
print(gpu.strip(), flush=True)
PY
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
GP_FACO_REQUIRE_NATIVE=1 GP_FACO_REQUIRE_CUDA=1 .venv/bin/python -m pytest -q \
    --basetemp="$project_root/.tmp/pytest-worker-engine" \
    > artifacts/gpu/worker-engine/pytest.log 2>&1
cat artifacts/gpu/worker-engine/pytest.log
