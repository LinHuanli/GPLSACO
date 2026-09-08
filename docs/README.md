# 研究文档索引

本组文档将 [v4 研究方案](design/GP_ACO_TSP_Research_Proposal_v4.md) 转换为可以实现、测试和预注册的工作包。原方案中的建议参数不自动成为已验证配置；本组文档中标注为“开发候选”的定义须在正式测试前冻结。

| 顺序 | 文档 | 解决的问题 |
|---|---|---|
| 1 | [范围与关键决策](planning/01_scope_and_decisions.md) | 研究贡献、歧义、哪些假设可能失败 |
| 2 | [数据与标签](planning/02_data_and_labels.md) | 格式、距离、最优性依据、划分和防泄漏 |
| 3 | [FACO 语义与外部基线](planning/03_faco_semantics_and_baselines.md) | 原生与共同扩展如何区分、源码怎样核验 |
| 4 | [C++/CUDA 架构](planning/04_solver_and_cuda.md) | 内存、控制事务、计时、并行边界 |
| 5 | [GP 与程序契约](planning/05_gp_and_program_ir.md) | 特征编号、数值规则、演化与验证选择 |
| 6 | [验收与质量门槛](planning/06_validation_gates.md) | 什么证据允许进入下一阶段 |
| 7 | [公共实验与统计](planning/07_evaluation_and_statistics.md) | 预算、失败、推断单位和结果格式 |
| 8 | [资源与执行顺序](planning/08_execution_and_resources.md) | 环境、GPU、训练成本、Git 与里程碑 |
| 9 | [主底座控制层契约](planning/09_control_contract.md) | 档案、区域、十二特征、完整重启和碰撞边界 |
| E1 | [反馈价值](experiments/E1_feedback.md) | 完整程序、去反馈重训及状态分叉 |
| E2 | [两尺度关系](experiments/E2_factorial.md) | 2×2 析因与交互 |
| E3 | [候选与出口](experiments/E3_candidates_and_escape.md) | Hard/Escape 的可执行定义与公平性 |
| E4 | [冻结迁移](experiments/E4_transfer.md) | 10K、TSPLIB、失败与适用范围 |

当前事实以 [进度记录](reports/progress.md) 和其链接的机器可读报告为准。历史 v4 中“已核查”“待安装”等描述保留为方案写作时点，不替代本地执行状态。
