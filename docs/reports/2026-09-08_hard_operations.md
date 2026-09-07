# Hard全新增边的CPU/CUDA操作核验

日期：2026-09-08。**Hard合法性图、重定位/2-opt逐边检查和受限真实随机选点已通过操作级验收。** 证据来自人工数据，不构成alpha/POPMUSIC候选质量实验。完整Engine尚未接入Hard，Escape和候选导出仍待实现，E3未执行。

机器证据：[hard_operations_results.json](hard_operations_results.json)。前序 [批量Engine报告](2026-09-08_batch_engine.md) 对应提交 `6a791a3` 的计时结果；本次修改共同内核后的正确性已重新回归，不把旧计时数值宣称为本次版本的性能。

## 执行规则

`SparseUndirectedGraph` 将有向候选成员与指定共同初始tour的边合并、无向化、去重，保存为按节点ID排序的CSR。图成员和LS枚举行独立，节点实际度数不截回k。测试中每行只有一个候选的6节点星形例得到最大度5、9条边，覆盖“无向化以后实际度数可能超过2k”的情况。

重定位检查移出补边和两条插入边；2-opt检查两条新增连接。先以无向边多重集合消去被删除后原样恢复的连接，再检查真正新增边。恒等、前驱/后继相邻、环边和小规模退化均纳入完整移动对照。仅检查当前节点到被选节点的边会漏掉约束；两个明确反例分别使重定位另一连接、2-opt闭合连接出图，检查均拒绝。

Hard选点先以完整重连合法性过滤主候选，再使用原有正权重roulette/全零均匀规则；主候选耗尽后按备用行和全局最近未访问顺序继续检查。全部声明来源均无合法移动时返回显式耗尽状态，构造不增加实际步数/MNE，不写 `visited[n]`，也不退回图外边。无约束路径仍只做原有visited筛选。

LS保留真实距离有序候选、两类交换、距离提前break、严格gain顺序和节点重新激活。通过距离前缀的候选即消耗一次move-evaluation；因合法性被拒绝的候选也计入限额，并另列 `constraint_rejections`。两个后端同样丢弃被评价上限中途截断的节点最佳move。

构造/LS使用同一个模板内核，Hard图以设备CSR视图传入；CPU选点/LS接受显式合法性谓词。CUDA显式序列诊断遇到不合法给定移动会停止，用于复核操作，不能以这条诊断序列替代真实选择器。另一入口直接调用生产 `StochasticChoices`，返回实际Philox随机数供CPU重放。

## 检查结果

| 检查 | 结果 |
|---|---|
| CPU完整边集oracle | n=3..7，固定城市0消去旋转重复；34,404次重定位、68,808次2-opt；370,416个谓词×图检查，237,648个拒绝 |
| 每条新边必要性 | 先比较移动前后完整tour，再逐条删除新增边；局部判断与完整图可行性一致 |
| CPU Release及ASan/UBSan | 各3项CTest通过，包含原生FACO操作与deadline回归 |
| CPU/CUDA Hard操作 | 576组，n=3/4/5/7/17/100/500/1000，连续/舍入整数/重复坐标，LS上限0/7/100000，8 ants |
| 实际LS与耗尽 | 20,216次接受；602,794次图拒绝；123个显式构造序列因图限制停止；所有构造与LS终态均在图中 |
| 真实GPU选点 | 6,144次逐项CPU重放一致；五分支数量依次为2,001/2,165/82/1,038/858 |
| 成本独立重算 | 普通GPU面板最大绝对误差3.41e-13；sanitizer面板最大5.12e-13 |
| GPU完整回归 | 6项CTest、49项Python测试通过；包括原有显式操作、固定迭代和批量deadline |
| CUDA工具 | memcheck/synccheck为0 errors；racecheck为0 errors、0 warnings、0 hazards；各288组操作和1,536次随机选点，结果一致 |

GPU为 cuda02 的 RTX A5000，UUID `GPU-056fae3f-b504-efe0-2d9d-b1186860e643`、driver 610.43.02。每次启动前核查实时进程占用。CUDA任务与CPU sanitizer任务均以正常终态归档。CPU期望值通过真正执行移动、重新提取全部tour边获得，未使用待测三/两边公式构造期望。

## 仍需完成的工作

当前两个公开诊断入口限制显式矩阵规模≤1024；10K的合法性图本体支持O(nk)CSR，但尚无10K Hard性能证据。普通批量Engine继续采用v4主比较的无约束原生候选语义，没有开放未经验证的Hard开关。

E3 Engine接入前需锁定候选工具、同一共同初始tour与E0的身份、廉价incumbent在准备超时时的图可行性、实际图边数/槽位匹配，以及图缓存的预算收费。不能只给普通Engine挂一张图便认为准备与截止语义也已验收。

Escape须先明确有界例外边的登记/去重/溢出/失效、固定槽位替换和受影响节点激活，再验证。主比较的区域、档案、epoch重启与GP动作仍可在现有无约束Engine上推进，不需要把E3的独立方法适配伪装成主底座规则。

复现命令（目标卡空闲、项目环境/构建就绪）：

```bash
ctest --test-dir build/cpu --output-on-failure
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/cpu-sanitize --output-on-failure
bash scripts/run_hard_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
.venv/bin/python scripts/summarize_hard_operations.py
```

原始日志保存在 `artifacts/gpu/hard-operations` 与 `artifacts/environment`；源码和精简报告入Git，不复制外部数据或二进制。
