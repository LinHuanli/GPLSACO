#!/usr/bin/env bash
# 使用调用方指定的空闲UUID；次数型开发测试与完整日志，不给求解器设置墙钟截止。
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
gpu_uuid="${1:?Usage: run_escape_checks.sh GPU-uuid LABEL [ctest|pytest|memcheck|synccheck|racecheck]}"
label="${2:?需要独立结果标签}"
sanitizer="${3:-}"
case "$label" in *[!a-zA-Z0-9_-]*|'') echo '结果标签只能使用字母、数字、下划线和连字符'; exit 2;; esac
case "$sanitizer" in ''|ctest|pytest|memcheck|synccheck|racecheck) ;; *) echo '未知检查模式'; exit 2;; esac
output="$project_root/artifacts/escape-engine/$label"
mkdir "$output"
export TMPDIR="$project_root/.tmp" CUDA_CACHE_PATH="$project_root/.cache/cuda"
project_python="${GPF_PYTHON:-$project_root/.venv/bin/python}"
gpu_lock_dir="${GPF_GPU_LOCK_DIR:-$project_root/.tmp}"
exec 9>"$gpu_lock_dir/worker-$gpu_uuid.lock"
flock -n 9 || { echo '目标GPU已有本项目持锁任务'; exit 3; }
GPF_CHECK_UUID="$gpu_uuid" GPF_CHECK_OUTPUT="$output" "$project_python" - <<'PY'
import csv
import datetime
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

uuid = os.environ['GPF_CHECK_UUID']
output = Path(os.environ['GPF_CHECK_OUTPUT'])
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
gpu = subprocess.check_output(['nvidia-smi', f'--id={uuid}', '--query-gpu=uuid,name,memory.used,utilization.gpu,driver_version',
    '--format=csv,noheader,nounits'], text=True)
report = {'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host': socket.gethostname(),
          'uuid': uuid, 'device': gpu.strip(), 'compute_processes': apps.strip()}
(output / 'device-before.json').write_text(json.dumps(report, sort_keys=True) + '\n')
if any(row and row[0].strip() == uuid for row in csv.reader(apps.splitlines())):
    raise SystemExit('目标GPU已有计算进程，不启动')
row = next(csv.reader(gpu.splitlines()))
if int(row[2]) > 1024 or int(row[3]) > 5:
    raise SystemExit('目标GPU显存/利用率不满足空闲阈值')
binary = Path('build/cuda-hard/escape_engine_semantics')
content = binary.read_bytes()
snapshot = output / 'escape_engine_semantics'
snapshot.write_bytes(content)
snapshot.chmod(0o755)
(output / 'binary-sha256.txt').write_text(hashlib.sha256(content).hexdigest() + '\n')
identities = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('build/cuda-hard').iterdir()
              if p.is_file() and (p.suffix == '.so' or os.access(p, os.X_OK))}
(output / 'build-binaries.json').write_text(json.dumps(identities, sort_keys=True) + '\n')
print(report['device'], flush=True)
PY
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
command=("$output/escape_engine_semantics" "$output/results.json")
if [[ "$sanitizer" == ctest ]]; then
    command=(ctest --test-dir build/cuda-hard --output-on-failure)
elif [[ "$sanitizer" == pytest ]]; then
    export GP_FACO_REQUIRE_CUDA=1 GP_FACO_REQUIRE_NATIVE=1
    export PYTHONPATH="$project_root/python:$project_root/build/cuda-hard"
    command=("$project_python" -m pytest -q --basetemp="$project_root/.tmp/pytest-escape-$label")
elif [[ -n "$sanitizer" ]]; then
    command=(/opt/cuda/bin/compute-sanitizer --tool "$sanitizer" --error-exitcode 42 "${command[@]}" --small)
fi
set +e
/usr/bin/time -f '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"exit_code":%x}' \
    -o "$output/resources.json" "${command[@]}" > "$output/run.log" 2>&1
result=$?
set -e
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$output/device-after.csv"
cat "$output/run.log"
exit "$result"
