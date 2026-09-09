# v2 实验执行协议

2026-09-09 最新执行轮次见[三 seed、50 代协议](round_three_seed_v2.md)：本轮仅 Full GP 三次进化与两类 FACO 比较，使用所有空闲 A5000，由 `scripts/run_campaign_v2.py` 自动接续。下文五 seed、Static/Rule/NoFeedback 和 E2–E4 保留为后续完整研究范围，不作为本轮的前置任务。旧 `run_v2.py --through training` 接续器已经移交，不能与新队列同时启动。

本轮先建立 FACO 基线、验证 GPU 效率、确定训练预算，再重新训练。旧的 32 蚂蚁记录只作为 v1 历史证据，不拼接到 v2 fitness，不从旧种群继续进化。

## 两种 FACO 对照

| 对照 | 目的 | 固定设置 |
|---|---|---|
| 原始 FACO 2022 | 检查论文方法的可复现表现 | 作者构造、初始化、LS；论文蚂蚁数；MNE 8；5000 迭代；8 CPU 线程；每实例 30 次 |
| GPU FACO 无 GP | 测量 GP 在同一底座上的净提升 | 节点重定位；MNE 8；全路线均匀起点；不重启；与 GP 相同的距离、初始化、LS、数值后端和 FE |

首批原始复现实例为 att532、u1817、pr2392、fl3795、fnl4461、rl5915。使用原始 TSPLIB 定义，外部评分器计算平均/最好 gap 和实际运行时间。初始化条数记录实际 `omp_get_num_procs()`，不能误写成线程数 8。连续合成问题保持原始连续距离；若需要规避原生 3-opt 的恒等移动循环，使用明确命名的适配版本，单独说明改动。

## GPU 与数值选择

只用实时空闲 RTX A5000；一次性能比较只占一张。每种实现一次预热、五次测量，记录准备、完整求解、最终评分、核函数时间、FE/s、显存及整代/整次进化预计成本。

候选为精确计算、FP32 搜索、FP32 fast-math。固定开发面板上比较固定 MNE 对照和代表性 GP。以实例为配对单位，先平均同实例的多个 seed；每个规模的“近似相对精确的平均 gap 增加”的单侧 95% 上置信界都必须不超过 **0.01 个百分点**，完整求解速度至少提高 **10%**。失败案例计入结果，不按质量选择性剔除。没有通过则保留精确计算。最终路线一律按原目标评分，TSPLIB 整数距离不得近似。

## 训练 ACO 迭代数

使用开发集每个规模排序后 `[96,128)` 的 32 个实例和 5 个求解 seed。沿用已登记的 24 个 Static/Rule 控制器，记录迭代 0、100、250、500、1000、2500、5000 的结果。不使用进度终端的控制器可以一次跑 5000 并记录中间点；GP 使用进度时不能据此假设短预算等价。

选择在每个规模都满足以下要求的最短预算：保留从共同初解到 5000 迭代平均改进的至少 95%，且控制器排名相对 5000 的 Spearman 相关至少 0.90。若无预算满足、改进非正或排名未定义，则使用 5000。95% 和 0.90 是本实验的设计阈值，不是文献结论。

## GP 训练、验证与报告

- 种群 128 × 50 个评价代；进化 seed 为 1103、2207、3313、4409、5519。GP 参数及依据见 [实现说明](../design/implementation_v2.md)。
- 普通 GP 训练和候选验证按 128 个体在同一 GPU 展开，共享几何数据。每个体保留独立状态、完整 FE；批次结果一次落盘并按成员恢复。并行布局不改变 seed 派生、fitness 或候选筛选，详见 [种群 GPU 设计](../design/population_gpu_v2.md)。
- 面板 seed 为 73001。每代在 TSP500、TSP1000 各抽 16 个训练实例，每实例 2 个求解 seed；50 代面板在方法间及进化重复间共享。每个 GP fitness 含 64 个独立 ACO 求解。
- 冻结数值后端和训练预算后，预先计算这些面板上两类 FACO 的配对基线。共享基线结果不作为 GP 训练 fitness 缓存。
- 每代报告 GP gap、配对 FACO gap、改进量 `FACO gap − GP gap`、胜率和成本。fitness 仍是按实例、再按规模等权汇总的原始 gap，不加额外奖励。
- 每 5 代冠军在固定开发监控面板评价，用于可比的学习曲线。
- 最终验证覆盖所有不同的代际冠军及最终种群，使用完整验证集、seed 17/29、5000 迭代。测试 seed 为 17/29/41/53/67/79/97/109/127/149，方法冻结后才读取测试分数。
- E1 比较 Full 与原始 FACO、GPU 无 GP、Static、Rule、NoFeedback，五个主要对比使用 Holm 校正。Static/Rule 保留完整配置族搜索；E2 析因、E3 四种候选条件、E4 迁移的科学范围继续保留。

标签是用户提供的参考解，未独立证明最优；报告使用 `gap = 100 × (cost/reference − 1)`，不把标签来源声明写成新的最优性证明。

## 执行入口和产物

所有命令从 GPLSACO 项目目录运行。环境、构建和输出都在项目内；`../Datasets`、`../references` 和 MTGP_ACO 只读。先用 `gpu-free`（当前环境可用 `/home/linbocheng/bin/gpu-free`）找到空闲 A5000，再在对应主机执行，GPU 参数使用完整 UUID。

```bash
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
```

GPU 三个后端使用相同源码，只改变搜索数值编译选项。以下以 exact 为例；另两个构建目录为 `build/v2-fp32` 和 `build/v2-fp32-fast`，选项值分别为 `fp32`、`fp32_fast`。

```bash
cmake -S . -B build/v2-exact \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGP_FACO_NATIVE=OFF -DGP_FACO_2022=OFF \
  -DGP_FACO_CUDA=ON -DGP_FACO_PYTHON=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)" \
  -DGP_FACO_SEARCH_NUMERIC=exact
cmake --build build/v2-exact -j 6
```

GP 种群构建独立放在 `build/v2-population-exact`，避免替换正在运行的 FACO 校准二进制。
采用上面相同命令，把两处构建目录改为 `build/v2-population-exact`；当前数值选择已经保留 exact。
`scripts/train_gp.py` 对普通 Full/NoFeedback/M10/M01 自动选择该构建与 128 个体并行；
E3 Hard/Escape 保留兼容入口。完整种群测速与真实 worker 检查入口为：

```bash
.venv/bin/python scripts/benchmark_population_v2.py --gpu GPU-实际UUID \
  --iterations 100 --measures 5 --output artifacts/v2/gpu-checks/population-exact.json
.venv/bin/python scripts/check_population_v2.py --gpu GPU-实际UUID \
  --directory artifacts/v2/population-training-check
```

以上是工程检查；两个规模的 128 个体都完成实际配置的 FE。测速不写正式训练 fitness。

原始 2022 和连续适配分别构建，不能混用距离定义：

```bash
cmake -S . -B build/v2-cpu -DCMAKE_BUILD_TYPE=Release \
  -DGP_FACO_NATIVE=OFF -DGP_FACO_CUDA=OFF -DGP_FACO_PYTHON=OFF \
  -DGP_FACO_2022=ON
cmake --build build/v2-cpu --target faco_2022 -j 6
.venv/bin/python scripts/prepare_native_edge_guard.py
cmake -S . -B build/v2-native-continuous -DCMAKE_BUILD_TYPE=Release \
  -DGP_FACO_NATIVE=OFF -DGP_FACO_CUDA=OFF -DGP_FACO_PYTHON=OFF \
  -DGP_FACO_2022=ON -DGP_FACO_2022_CONTINUOUS=ON \
  -DFACO_2022_SOURCE_DIR="$PWD/.deps/faco-2022-continuous"
cmake --build build/v2-native-continuous --target faco_2022 -j 6
```

连续适配用原始 double 坐标构建连续距离矩阵，关闭内部仍按整数距离排序的 KD-tree，转用作者已有的 NN-list 分支。连续适配 revision 2 将恒等 2-opt 的增益明确归零；3-opt 只排除三个无向边集合完全相同的恒等移动。原始 TSPLIB 目标不编译这些改动。配对原始初始化固定为本轮 cuda10 主机的 24 条原生初始路线、8 线程。

下面的 `GPU_UUID` 应替换为发现的空闲 A5000 UUID：

```bash
.venv/bin/python scripts/run_v2.py --gpu GPU_UUID --through baselines
```

此入口在开始质量评价前登记 `panels.json` 和 `selection_rules.json`；依次读取/生成 `final-v2-*.json` 性能结果、数值质量面板、24 控制器曲线和两类 FACO 配对缓存。`--through training` 会在所有前置步骤完成后执行 E1 的完整基线调参及五个 Full、五个 NoFeedback 训练和完整验证。不会读取正式测试性能。

| 文件或入口 | 含义 |
|---|---|
| `artifacts/v2/protocol/panels.json` | 50 代共享训练面板、开发、验证、测试成员与 seeds |
| `selection_rules.json` | 看质量结果之前登记的选择阈值和候选顺序规则 |
| `numeric_selection.json` | 数值后端与质量/速度证据 |
| `frozen.json` | 后端、训练 H、正式验证/测试 5000 的冻结设置 |
| `baselines.sqlite` / `baselines_ready.json` | 逐成员 FACO 结果及完整性覆盖状态；使用显式参数，不计算摘要 |
| `pipeline_status.json` / `baseline_progress.json` | 队列阶段及 baseline 进度 |
| `scripts/calibrate_v2.py quality/curves` | 可单独执行、按追加日志恢复的固定评价次数阶段 |
| `scripts/cache_baselines_v2.py` | 先完成 GPU FACO 后释放 GPU，再完成原始 CPU FACO |
| `scripts/tune_baselines_v2.py` | 完整 Static/Rule 族；开发64筛选、每类前10在完整验证256选择 |
| `scripts/train_gp.py` | 检查两类 baseline 覆盖后启动新种群；`--resume` 只用于该 v2 运行 |
| `scripts/summarize_v2.py` | 读取精简现有结果更新当前报告，不求解、不重新评分 |

数值质量面板固定开发 `[64,96)`，每规模 32 实例 × 5 seeds；分别检查真 GPU FACO 和代表性 GP，两规模共四个配对 t 上界。性能比较采用固定 Static 和代表性 GP，两规模共四组。先选择四组均至少 1.1 倍的候选，再按四组耗时之和从快到慢检查质量；较快候选通过后不额外运行较慢候选的质量面板。

训练前 FACO 缓存覆盖 3200 个训练成员、64 个监控成员、1024 个验证成员，每方法共 4288 个配对位置。缓存使用方法、实例、seed、迭代、蚂蚁数、后端和初始化设置，相同成员可共享；测试对照定义先固定，方法冻结前不计算测试分数。

24 个校准控制器和完整 776 个调参配置的声明见 [controller_families_v2.json](../../configs/controller_families_v2.json)。该文件只包含控制策略；旧配置中的 32 蚂蚁和旧 FE 档不进入新入口。正式调参族为 160 个 Static 和 616 个 Rule。
