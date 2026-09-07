# 启动工作复现命令

以下从GPLSACO根执行；外部Datasets/references路径按默认布局存在。所有生成物位于项目内。精确依赖见provenance/python.lock.txt；环境安装见资源文档。

## 来源与数据

```bash
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p .tmp artifacts/data artifacts/environment artifacts/native
.venv/bin/python scripts/lock_sources.py
.venv/bin/python scripts/audit_data.py > artifacts/data/inventory.log
```

审计读取约16GB并生成全局文件manifest；这是文件全hash和首条记录验证。500/1K主池的完整逐条核验与split使用下面独立命令。

```bash
# 使用不存在已发布 instances.sqlite 的新目录；已有身份不能覆盖。
.venv/bin/python scripts/build_main_dataset.py --workers 4 \
  --output artifacts/data/main-index-v1 > artifacts/data/main-index-v1-build.log
.venv/bin/python scripts/verify_main_dataset.py --index artifacts/data/main-index-v1
```

构建读取128,416条，保存项目内SQLite偏移索引和独立labels表。核验完成后发布 `provenance/splits.v1.json`；已存在的公开manifest必须逐字节相同，否则拒绝覆盖。重放划分禁止访问labels；原测试记录仅作格式审计。

## CPU 和评分测试

```bash
cmake -S . -B build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/cpu -j 4
GP_FACO_REQUIRE_NATIVE=1 PYTHONPATH="$PWD/python:$PWD/build/cpu" \
  .venv/bin/python -m pytest -q --basetemp="$PWD/.tmp/pytest-cpu"
```

没有外部FACO快照的机器可设置 `-DGP_FACO_NATIVE=OFF`，只检验程序IR与CPU评分，不能声称已检验FACO语义。

## CPU FACO操作与内存检查

```bash
OMP_NUM_THREADS=1 ctest --test-dir build/cpu --output-on-failure \
  > artifacts/environment/ctest-core-semantics.log
GP_FACO_REQUIRE_NATIVE=1 PYTHONPATH="$PWD/python:$PWD/build/cpu" \
  .venv/bin/python -m pytest -q --basetemp="$PWD/.tmp/pytest-core-data" \
  > artifacts/environment/pytest-core-data.log

cmake -S . -B build/cpu-sanitize -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 -DGP_FACO_PYTHON=OFF \
  '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build build/cpu-sanitize --target faco_native_semantics -j 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  OMP_NUM_THREADS=1 ctest --test-dir build/cpu-sanitize --output-on-failure \
  > artifacts/environment/ctest-cpu-sanitize.log
.venv/bin/python scripts/summarize_core_data.py
```

`faco_native_semantics`直接编译原始FACO作oracle，结果分别写入两个build目录的 `native_semantics_results.json`。数据与CPU摘要单独保存为 `docs/reports/data_cpu_results.json`，不覆盖历史启动结果。

## 原生连续目标诊断

```bash
# 原生版本的已知失败复现；120秒超时后保留日志。
.venv/bin/python scripts/native_smoke.py --output artifacts/native/smoke
# 以上命令预期可能非零；随后单独运行观察诊断。
.venv/bin/python scripts/probe_native_initialization.py \
  --task artifacts/native/smoke/500-mfaco-17/task.json

.venv/bin/python scripts/prepare_native_edge_guard.py
cmake -S . -B build/native-edge-guard -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 -DGP_FACO_PYTHON=OFF \
  -DFACO_SOURCE_DIR="$PWD/.deps/native-edge-guard"
cmake --build build/native-edge-guard -j 4
.venv/bin/python scripts/native_smoke.py --variant native-edge-guard \
  --build build/native-edge-guard --output artifacts/native/edge-guard-smoke
```

EdgeGuard与原生版本有明确身份区分；原始文件不变。复现时不要覆盖有未结束进程的目录；上述为全新运行命令，不是恢复协议。

## RTX A5000 CUDA 验证

```bash
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGP_FACO_NATIVE=OFF -DGP_FACO_CUDA=ON \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/cuda -j 4
# 在选定目标host的共享项目目录执行，UUID由实时gpu-free/nvidia-smi检查得到。
bash scripts/run_gpu_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
```

上面UUID是启动时cuda02的一张A5000历史身份；脚本会重查当前compute进程、显存和利用率，忙则不启动。其他型号需重新选择对应CUDA架构和独立构建目录。GPU脚本依次执行pytest、量化评分、memcheck和synccheck；所有日志在artifacts/gpu。此诊断程序每次分配显存，不是持久Engine或完整运行的计时基准。

`scripts/summarize_bootstrap.py` 只汇总历史启动日志中的CPU27/GPU29项测试。当前CPU为31项；应使用本节新增的日志路径与 `scripts/summarize_core_data.py` 保留不同阶段的证据，不能将更改过的日志当作旧结果。
