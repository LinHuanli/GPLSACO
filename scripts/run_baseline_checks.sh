#!/usr/bin/env bash
# 同底座Static/Rule、GP/count旧路径回归；每轮使用新目录，保留全部失败。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
gpu_uuid="${1:?Usage: bash scripts/run_baseline_checks.sh GPU-uuid [new-output-directory]}"
output="${2:-artifacts/gpu/baselines/checks-v1}"
export TMPDIR="$project_root/.tmp"
export CUDA_CACHE_PATH="$project_root/.cache/cuda"
export PYTHONPATH="$project_root/python:$project_root/build/cuda"
output="$(.venv/bin/python - "$output" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
if not path.is_relative_to(Path.cwd()):
    raise SystemExit('输出必须在GPLSACO内')
path.mkdir(parents=True, exist_ok=False)
print(path)
PY
)"
exec 9>.tmp/worker-"$gpu_uuid".lock
flock -n 9 || { echo '目标GPU已有本项目执行器'; exit 2; }
GPF_CHECK_UUID="$gpu_uuid" GPF_CHECK_OUTPUT="$output" .venv/bin/python - <<'PY'
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
Path(os.environ['GPF_CHECK_OUTPUT'], 'device.csv').write_text(gpu)
print(gpu.strip(), flush=True)
PY
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
ctest --test-dir build/cuda --output-on-failure > "$output/ctest.log" 2>&1
GP_FACO_REQUIRE_NATIVE=1 GP_FACO_REQUIRE_CUDA=1 .venv/bin/python -m pytest -q \
    --basetemp="$project_root/.tmp/pytest-baselines" > "$output/pytest.log" 2>&1
for sanitizer in memcheck synccheck racecheck; do
    /opt/cuda/bin/compute-sanitizer --tool "$sanitizer" --error-exitcode 42 \
        build/cuda/baseline_engine_semantics "$output/$sanitizer.json" \
        > "$output/$sanitizer.log" 2>&1
done
cat "$output/ctest.log" "$output/pytest.log"
tail -n 1 "$output/memcheck.log"
tail -n 1 "$output/synccheck.log"
tail -n 1 "$output/racecheck.log"
