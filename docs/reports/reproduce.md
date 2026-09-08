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

## CUDA FACO构造与LS操作

上述CUDA构建现在同时生成 `faco_cuda_semantics`，使用人工点集和给定选点排列对照CPU操作，不执行主数据集求解。

```bash
cmake --build build/cuda -j 4
# 在当前空闲目标host的共享项目目录执行；脚本再次实时核查UUID。
bash scripts/run_cuda_faco_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
.venv/bin/python scripts/summarize_cuda_faco.py
```

完整差异结果在 `build/cuda/cuda_faco_results.json`，其他日志在 `artifacts/gpu/faco-operations`；pytest为33项，memcheck/synccheck/racecheck各使用540组保留全部类型的面板。此脚本使用内核锁避免同项目重复执行；锁文件存在不表示作业仍活跃。脚本退出后再归集报告，共享文件系统有缓存时应从运行host确认日志尾部与进程终态，不能据观察延迟重启。

## 固定迭代GPU FACO与开发池试跑

```bash
cmake --build build/cpu -j 4
OMP_NUM_THREADS=1 ctest --test-dir build/cpu --output-on-failure \
  > artifacts/environment/ctest-cpu-fixed.log
cmake --build build/cuda -j 4
# 在空闲目标GPU的host运行，脚本会实时检查指定UUID。
bash scripts/run_fixed_faco_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
# 上一作业确认结束后单独试跑；入口再次检查实时compute进程。
CUDA_VISIBLE_DEVICES=GPU-056fae3f-b504-efe0-2d9d-b1186860e643 \
  CUDA_CACHE_PATH="$PWD/.cache/cuda" TMPDIR="$PWD/.tmp" \
  .venv/bin/python scripts/fixed_faco_smoke.py \
  --gpu-uuid GPU-056fae3f-b504-efe0-2d9d-b1186860e643
.venv/bin/python scripts/summarize_fixed_faco.py
```

当前GPU CTest包含原操作对照和固定迭代流程两个目标，pytest为39项。新入口的memcheck/synccheck/racecheck各检查20种配置、240个蚂蚁批次。所有新日志写入 `artifacts/gpu/fixed-faco`，保留之前报告的日志。

开发试跑默认500/1K各8个已预登记开发ID、三个seed、50批次；重复执行须使用新的 `--output` 目录，不覆盖已有summary。标签在完整求解返回后才由外部Python evaluator读取。一次 `FixedFacoGpu.run_iterations` 完成全部迭代；这是开发接口，尚无deadline或GP决策。

## 主底座控制Engine、档案和重启

```bash
TMPDIR="$PWD/.tmp" cmake --build build/cpu -j 6
ctest --test-dir build/cpu --output-on-failure
TMPDIR="$PWD/.tmp" cmake --build build/cpu-sanitize -j 6
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build/cpu-sanitize --output-on-failure
TMPDIR="$PWD/.tmp" cmake --build build/cuda -j 6
# 在实时空闲目标host的共享项目目录执行；脚本再次核查UUID。
bash scripts/run_control_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
# 必须确认上一运行句柄已结束，再启动计时；已有summary时另给新的--output。
CUDA_VISIBLE_DEVICES=GPU-056fae3f-b504-efe0-2d9d-b1186860e643 \
  CUDA_CACHE_PATH="$PWD/.cache/cuda" TMPDIR="$PWD/.tmp" \
  .venv/bin/python scripts/control_engine_smoke.py \
  --gpu-uuid GPU-056fae3f-b504-efe0-2d9d-b1186860e643
.venv/bin/python scripts/summarize_control_engine.py
```

新增核验为8项GPU CTest、62项Python测试，三种sanitizer执行完整控制循环的小面板。CPU原语包含手算和独立枚举；C++ GPU诊断包含强制双指纹碰撞。生产调用 `engine.evaluate_program(keys, seeds, seconds, program_dict, preparation_mode, experiment_mask)` 不导出完整诊断轨迹、强制碰撞开关或迟到候选。

计时试跑固定32 colony、每colony 32 ants，500/1K各16个已有开发实例×2 solve seeds，两个未训练的一终端程序和两种准备模式。它核验实际控制与提交成本，不是正式GP训练或统计比较。原始日志位于 `artifacts/gpu/control-engine`，精简报告为 `control_engine_results.json`。
