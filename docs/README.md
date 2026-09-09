# 文档导航

**当前先读 [单树与三树预实验](reports/单树与三树预实验.md)**：修复共同繁殖流程后，比较两种表示，各3 seed×10代；仍只训练TSP500。设置与依据见[当前协议](experiments/representation_pilot_v3.md)。旧的[3 seed×50代实验](reports/三次进化实验.md)已暂停，保留历史结果。

[实验说明与结果](reports/实验说明与结果.md)保留研究背景、FACO/GP 参数依据、原始 FACO 复现和 GPU 测速。GP 整种群并行实现见 [种群 GPU 设计](design/population_gpu_v2.md)。只想看正在运行什么，读 [当前进度](reports/progress.md)。

[训练速度与信号分析](reports/训练速度与信号分析.md)对照 RMTGP 的实际 TSP500 耗时，记录第 10 代的表达式重复与 fitness 区分情况；第 5 节解释 GP 的 12 个输入、32 个动作评分，以及第 12 代快照下的训练冠军与开发监控最佳树。

| 目录 | 放什么 | 主要入口 |
|---|---|---|
| `design/` | 研究目的、算法设计、参数依据 | [研究方案 v4](design/GP_ACO_TSP_Research_Proposal_v4.md)、[当前实现与参数依据](design/implementation_v2.md) |
| `experiments/` | 怎样开展实验，如何选择和比较 | [当前单树/三树预实验](experiments/representation_pilot_v3.md)、[完整 v2 协议](experiments/protocol_v2.md)、[E1](experiments/E1_feedback.md)、[E2](experiments/E2_factorial.md)、[E3](experiments/E3_candidates_and_escape.md)、[E4](experiments/E4_transfer.md) |
| `reports/` | 人能直接阅读的结果与进度 | [单树与三树预实验](reports/单树与三树预实验.md)、[已有复现与参数依据](reports/实验说明与结果.md)、[当前进度](reports/progress.md) |
| `archive/v1/` | 旧的 32 蚂蚁设计、工程报告和阶段记录 | [历史说明](archive/v1/README.md) |
| `archive/v2/` | 被最新单规模方案替代的双规模说明 | 只读历史说明，不是当前运行入口 |
| `../results/v2/` | 精简机器结果、图和个体展示，支持报告中的表格 | [原始 FACO 复现](../results/v2/faco_2022_tsplib.json)、`round-tsp500-3seed/` |
| `../results/v3/` | 单树/三树预实验检查、精简结果和6条曲线 | `representation_verification.json`、`representation-pilot/` |
| `../results/v1/` | 旧版机器结果和图 | 只读历史证据 |
| `../artifacts/v2/` | 原始 tour、日志、检查点、数据库和实验中间产物 | 不放入 docs，不提交完整原始输出 |
| `../artifacts/v3/` | 新预实验的原始情境、checkpoint和逐任务结果 | `gp-representation-pilot/`，不提交原始运行输出 |

当前轮次以[单树/三树预实验协议](experiments/representation_pilot_v3.md)为准；[完整 v2 协议](experiments/protocol_v2.md)继续规定后续 E1–E4 的范围。历史文件中的50代自动流程、32蚂蚁、按秒预算、其他GPU型号和“正在运行”字样都不代表当前设置。
