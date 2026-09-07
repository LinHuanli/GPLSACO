# FACO 操作语义、来源与外部基线

## 1. 来源锁定

本地参考目录是源码快照，没有 `.git` 元数据。可计算 Git blob 与 SHA-256；关键文件与 v4 所列 blob 相同，只证明这些文件相同，不能据此声称整个目录已验证为某 commit。将计划 commit、实际关键文件 hash、所有参与构建的源文件 hash 和许可分别写入 `provenance/sources.lock.json`。

`ACO-TSP-Adaptive-Tuning` 的 MIT 许可明确；`FocusedACO` 本地未发现项目 LICENSE，暂作为只读参考，不推定继承另一仓库许可。LKH 的 README 声明 research use；Concorde 本地 `references/concorde` 实际为可执行文件，源码位于 `concorde_code`。LKH/Concorde/ACOTSP 通过独立进程适配，构建副本和输出均在 GPLSACO。cuOpt 只读审查其任务分解；首版无 cuOpt 强依赖。

## 2. 三条实现身份

| 身份 | 修改边界 | 可以承担的证据 |
|---|---|---|
| FACO-Native | 作者 `run_mfaco` 源码不改；明确编译选项、参数和转换开销 | CPU 基线、操作语义来源 |
| FACO-Control | 保留选点→重定位→MNE→checklist；增加固定区域、档案和 epoch 重启 | 共同实验底座 |
| GP-FACO | 仅替换共同接口的动作决策 | 相同底座上的学习增益 |

源码 CLI 必须显式 `--alg mfaco`；默认 `faco_apt` 是另一算法。对 FACO-Native 保留显式的个体记忆等原配置；共同底座统一 `keep_better_ant_sol=false`、`source_sol_local_update=false`，不把这两项固定适配算为 GP 贡献。

## 3. 必须逐操作核验的语义

构造从完整参考 tour 复制开始。起点来自选定区域；原生版起点全 tour 均匀。当前节点主候选 roulette 按信息素×距离权重抽样，备用表及全未访问回退的选择方式逐分支对照源码，不能统称为相同 roulette。visited 随路径推进，选中节点移到 curr 后，curr 更新为选中节点。

MNE 只在参考 tour 中没有无向 `{curr,sel}` 边时增加，且与本次操作是否非恒等、最终 tour 新边数不同。构造最多访问 n 节点；无合法候选或 deadline 到达时返回当前可行 tour。checklist 的加入时机及 old predecessor 从移动前读取；不以任意 2-opt 替代。

LS 每次处理一个 pending 节点，按原生两类候选交换、真实距离有序行和提前 break；每节点内选 move 的顺序和 tie 逐项记录。接受移动后修复 inverse positions，并允许处理过的节点再次激活。反转用原生片段/补片段语义，不能仅因无向成本一样而忽略后续方向。

稀疏信息素保留有向行条目；图外边走 default，非零且蒸发。强化与 bounds 取原生 LS 公式；共同底座用 epoch-best 替换 GB 的相应角色。重启需重置 stored、default、产品缓存、epoch 反馈、ant 工作状态；GB 和档案保留。

## 4. CPU 对照路线

先构建完全未修改的原生程序，使用 500/1K 开发样本进行连续 EXPLICIT 目标的短迭代 smoke run，验证输出 tour 成本。然后建立测试 harness 直接调用锁定源中的 Route/pheromone 等操作，对人工构造和随机序列逐步比较新实现。

不得将“能运行原生程序”当成 CUDA 语义等价证明。共同实现还需比较 relocate 后 tour/position、增量成本、MNE、checklist 执行、pheromone lookup/update，并用相同输入选择权重检查统计分布。CPU/GPU 原始随机数和 FP 运算顺序不同的完整运行只要求明确的分布与不变量校验，不能要求无依据的逐位相等。

## 5. 基线顺序与适配要求

最先完成 Native mfaco 和 faco_apt；再有充分配置的 Static、Rule、同接口 bandit。ACOTSP MMAS+LS 和 LKH 作为 CPU 竞争力参照；支持相同原目标、版本和许可条件下再纳入 cuOpt/DyNACO。下载/复现可用性如实披露，不能因列在计划中就标为已实现。

每个 adapter 输出原始 tour、成本、可行性、初始化/转换/求解/总时间、线程/GPU 配置、seed、命令和输入 hash。独立 Python 距离实现重算返回路线。原生没有 wall-clock 截止接口时，迭代 pilot 只用于正确性，不与固定预算的 GP 混为正式结果；正式时间对照需可提取截止前 incumbent 的适配与单独验证。

## 6. 启动实测发现：连续距离的初始化恒等移动

2026-09-08，原生 `run_mfaco` 在500开发样本初始化120秒未结束。项目内观察副本捕获 `three_opt_nn` 对同一组三条无向边因浮点求和顺序产生 `2.7755575615628914e-17` 的伪gain，并按 `cost < curr` 接受。此初始化3-opt无迭代上限；不能把“支持EXPLICIT读取”推断为“所有连续目标运行稳定”。

为继续核验，新增独立 **FACO-Native-EdgeGuard** 变体：只在初始化3-opt排除删除/添加无向边多重集合完全相同的候选，不加新的gain阈值，不改原始源码。生成脚本、完整diff、原/新SHA-256见 `scripts/prepare_native_edge_guard.py` 与 `provenance/native_edge_guard.*`。该版本不能冒充未修改FACO-Native；其CLI与诊断wrapper成本一致，并由外部连续目标重算核验。

还发现 `par_build_initial_routes` 初始化先2-opt再3-opt，初始tour数量默认取 `omp_get_num_procs()/2`，不是threads参数。本机为8个。正式共同底座必须显式固定初始解数量/改进流程并计时；原生对照保留其实际行为并披露硬件。原生整数目标表现、更多连续实例、非恒等近零gain行为仍待专项验证。
