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

审计读取约16GB并生成本研究manifest；这是文件全hash和首条记录验证，正式面板还需逐条验证。

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

所有命令成功后可用 `scripts/summarize_bootstrap.py` 提取精简报告；其固定检查数量对应本次启动阶段，新增测试后须更新汇总口径。
