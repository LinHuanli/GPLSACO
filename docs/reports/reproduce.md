# 启动工作复现命令

最新主实验遵循[评价次数协议](../planning/12_evaluation_count_protocol.md)。本文件已执行的wall-clock任务是历史工程检查，不构成主时间预算冻结。

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

## 常驻worker与外部fitness

```bash
# CPU协议/聚合检查，不启动CUDA。
PYTHONPATH="$PWD/python:$PWD/build/cpu" GP_FACO_REQUIRE_NATIVE=1 \
  .venv/bin/python -m pytest -q --basetemp="$PWD/.tmp/pytest-worker-cpu"
# 在当前空闲目标host运行；接口检查和真实worker诊断分别保存。
bash scripts/run_worker_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
TMPDIR="$PWD/.tmp" PYTHONPATH="$PWD/python" \
  .venv/bin/python scripts/worker_smoke.py \
  --gpu-uuid GPU-056fae3f-b504-efe0-2d9d-b1186860e643
PYTHONPATH="$PWD/python" .venv/bin/python scripts/summarize_worker.py
```

`worker_smoke.py` 拒绝覆盖任何已有任务目录，重新运行须给新的 `--output`。其协调进程不导入 `gp_faco_ext`，GPU选择在spawn子进程内完成。Future超时后保持原句柄，不据共享文件系统日志可见性重新提交。当前正式准备费用表、DEAP代际及checkpoint仍待实现；普通任务错误会原样保存，初始化/进程故障通过Future显式报错。

## 旧时间模式DEAP训练、固定费用与恢复

以下显式选择历史时间配置。当前默认已改为次数模式，见下一节；旧`pilot-v2`的精确重审计须恢复其原版本Python源码和二进制。

```bash
# 在GPLSACO目录内执行；CPU状态机测试的临时产物也留在项目内。
TMPDIR="$PWD/.tmp" .venv/bin/python -m pytest -q \
  tests/python/test_evolution.py tests/python/test_training.py \
  --basetemp="$PWD/.tmp/pytest-training"
# 在实时空闲GPU的host运行；此处为本轮cuda13的历史UUID，worker仍会重查占用。
bash scripts/run_training_checks.sh GPU-da4490c9-1c09-49b2-00dc-b175417ae7bb
# 使用新的输出目录；不要覆盖历史pilot-v1/v2。
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/train_gp.py \
  --config configs/training_pilot.json \
  --gpu-uuid GPU-da4490c9-1c09-49b2-00dc-b175417ae7bb \
  --output artifacts/gpu/gp-training/reproduce-new --stop-after-tasks 5
cp artifacts/gpu/gp-training/reproduce-new/checkpoint.json \
  artifacts/gpu/gp-training/reproduce-new/paused-checkpoint.json
# 上一命令终态已确认、同一GPU再次空闲后，继续同一配置和任务身份。
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/train_gp.py \
  --config configs/training_pilot.json \
  --gpu-uuid GPU-da4490c9-1c09-49b2-00dc-b175417ae7bb \
  --output artifacts/gpu/gp-training/reproduce-new --resume
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/summarize_training.py \
  --input artifacts/gpu/gp-training/reproduce-new \
  --output artifacts/gpu/gp-training/reproduce-new/audit.json
```

这关闭了上节worker阶段的代际/恢复待办。旧配置 `configs/training_pilot.json`只读取开发池；正式128×50配置和E1参数未冻结。归档包含带版本/hash的JSON envelope，实际IR位于 `selected_program.json` 的 `state.program`，其内容可直接用于 `Program.from_dict` 后传给对应版本的原生Engine。`programs/` 保存全部出现过的IR，`generations/` 保存完整已评种群，`tasks/` 保存逐任务返回与外部核验，checkpoint保存RNG、准备记录和完成表。

摘要校验拒绝源码、配置、软件、数据或GPU身份变化；不能换GPU后把旧成绩继续拼入当前run。占用检查在提交前失败则保留待办，确认该句柄已终止且设备空闲后才恢复。运行中Future超时保持原句柄，不另起同一任务。当前真实恢复验收是 `pilot-v2`，`pilot-v1` 的归档摘要失败证据保留在训练报告中。

## 评价次数主入口与训练恢复

默认配置`configs/training_counts_pilot.json`使用8个体×3代、500/1K各256 search-tour evaluations/colony、32 colonies×32 ants；缓存模式为`cached`，无求解秒数上限、缓存扣费或迟到批丢弃。次数IR要求feature_spec_id=2；程序终端0为`progress`。这里的256次数档只用于开发验收。

```bash
# 完成当前CUDA构建后，在实时空闲GPU所在host的共享项目目录执行。
# 本轮实际host为cuda04；UUID只记录历史身份，启动时仍须检查占用。
bash scripts/run_evaluation_count_checks.sh GPU-34b223c6-7502-b097-19e0-a411b1708f06
# 用新目录保留独立运行，不覆盖已经验收的training-v1。
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/train_gp.py \
  --config configs/training_counts_pilot.json \
  --gpu-uuid GPU-34b223c6-7502-b097-19e0-a411b1708f06 \
  --output artifacts/gpu/evaluation-count/reproduce-new --stop-after-tasks 5
cp artifacts/gpu/evaluation-count/reproduce-new/checkpoint.json \
  artifacts/gpu/evaluation-count/reproduce-new/paused-checkpoint.json
# 确认原句柄正常结束且目标UUID空闲后恢复同一任务。
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/train_gp.py \
  --config configs/training_counts_pilot.json \
  --gpu-uuid GPU-34b223c6-7502-b097-19e0-a411b1708f06 \
  --output artifacts/gpu/evaluation-count/reproduce-new --resume
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/summarize_training.py \
  --input artifacts/gpu/evaluation-count/reproduce-new \
  --output artifacts/gpu/evaluation-count/reproduce-new/audit.json
```

当前次数验收为GPU CTest 10项、Python 110项、三个CUDA工具；完整日志位于`artifacts/gpu/evaluation-count`。脚本使用固定检查日志路径，后续不同版本复现检查前先归档该目录内已有日志及hash，避免覆盖历史证据。本轮负FE参数异常类型修复前的108通过/2失败、二进制及绑定源码位于`initial-checks/`。

真实`training-v1`的暂停及恢复命令外层使用GNU time记录CLI实际墙钟/CPU/max RSS，不设置时间截止。独立摘要重算全部1,600条路线和fitness，重放三代DEAP与面板RNG，验证原7条完成记录、准备资源历史和409,600 FE账目。训练与验证FE、LS移动检查工作量分别汇总；重新准备的实际成本不替换原记录，也不扣搜索次数。原二进制SHA256为`2269faaafd558e2ec5e0f8ec770fc89c357cc231ae9a69092c9391a6fb747b66`；后续代码变化后重审计仍需其对应源码和环境，不能绕过身份检查。

## 分层成本与旧时间入口工程校准

历史GPU运行命令如下；当时使用空目录和cuda04的空闲A5000。本轮配置与秒数只保留为工程证据，新主实验使用评价次数。再次运行需新输出目录及与新计时报告对应的新校准配置，不覆盖现有报告或恢复已完成矩阵。

```bash
bash scripts/run_profiling_checks.sh GPU-34b223c6-7502-b097-19e0-a411b1708f06
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/profile_solver.py \
  --gpu-uuid GPU-34b223c6-7502-b097-19e0-a411b1708f06
# 冻结当前原始源码字节；后续构建不能覆盖历史运行副本。
.venv/bin/python scripts/archive_profiling_runtime.py \
  artifacts/gpu/profiling/cost-matrix-v1
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/summarize_profiling.py \
  artifacts/gpu/profiling/cost-matrix-v1 --output docs/reports/profiling_results.json
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/calibrate_budget.py \
  --gpu-uuid GPU-34b223c6-7502-b097-19e0-a411b1708f06
.venv/bin/python scripts/archive_profiling_runtime.py \
  artifacts/gpu/profiling/deadline-v1
TMPDIR="$PWD/.tmp" .venv/bin/python scripts/summarize_calibration.py \
  artifacts/gpu/profiling/deadline-v1 --output docs/reports/budget_calibration_results.json
```

实际GPU命令外层另用GNU time记录完整CLI墙钟、CPU user/system、max RSS与exit code，保存在同名`.time.json`。两个摘要脚本只读开发坐标和已归档结果，独立重算全部路线，不启动CUDA、不读取标签。重审计可将`--output`指向项目内新的audit路径；`archive_profiling_runtime.py`验证已有快照而不覆盖。快照通过内容摘要与原始manifest对应，即使后续源码/二进制已重建仍能核验旧身份。

已验收训练二进制`5f9211…`在重建前另存于`artifacts/binaries/<完整SHA256>/`，附12875d6的Git源快照。旧训练的完整重放还要求对应的旧Python源和协议；不能把当前源码的版本检查绕过后拼接旧结果。
