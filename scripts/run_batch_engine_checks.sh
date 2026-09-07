#!/usr/bin/env bash
# 运行操作级CPU/CUDA差异、内存、同步与共享内存竞争检查。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
gpu_uuid="${1:?Usage: bash scripts/run_batch_engine_checks.sh GPU-uuid}"
mkdir -p artifacts/gpu/batch-engine .tmp .cache/cuda
exec 9>.tmp/batch-engine-checks.lock
flock -n 9 || { echo '此项目已有CUDA操作检查持有内核锁'; exit 2; }
export TMPDIR="$project_root/.tmp"
export CUDA_CACHE_PATH="$project_root/.cache/cuda"
export PYTHONPATH="$project_root/python:$project_root/build/cuda"

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
Path('artifacts/gpu/batch-engine/device.csv').write_text(gpu)
print(gpu.strip(), flush=True)
PY
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
ctest --test-dir build/cuda --output-on-failure \
    > artifacts/gpu/batch-engine/ctest.log 2>&1
GP_FACO_REQUIRE_NATIVE=1 GP_FACO_REQUIRE_CUDA=1 .venv/bin/python -m pytest -q \
    --basetemp="$project_root/.tmp/pytest-batch-engine" \
    > artifacts/gpu/batch-engine/pytest.log 2>&1
for sanitizer in memcheck synccheck racecheck; do
    /opt/cuda/bin/compute-sanitizer --tool "$sanitizer" --error-exitcode 42 \
        build/cuda/batch_engine_semantics "artifacts/gpu/batch-engine/$sanitizer.json" --small \
        > "artifacts/gpu/batch-engine/$sanitizer.log" 2>&1
done
cat artifacts/gpu/batch-engine/ctest.log artifacts/gpu/batch-engine/pytest.log
tail -1 artifacts/gpu/batch-engine/memcheck.log
tail -1 artifacts/gpu/batch-engine/synccheck.log
tail -1 artifacts/gpu/batch-engine/racecheck.log
