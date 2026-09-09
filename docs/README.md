# 文档导航

## 想了解现在做到了哪里

先读[实验进展总览](reports/实验进展总览_2026-09-09.md)。它集中说明研究问题、参数、数据划分、已得到的质量结果、各次训练进度和未完成项。

[当前进度](reports/progress.md)用于后续更新。历史工程报告记录代码检查，不代表每篇都是一个已完成的科学实验。

## 想核对实验设置

| 内容 | 文档 |
|---|---|
| 研究目标和四个科学问题 | [原始v4方案](design/GP_ACO_TSP_Research_Proposal_v4.md) |
| 当前资源及取消完整性校验 | [运行管理简化](planning/29_runtime_simplification.md) |
| 按evaluation次数执行 | [次数预算协议](planning/12_evaluation_count_protocol.md) |
| E1：反馈价值 | [E1设计](experiments/E1_feedback.md)、[正式配置](../configs/e1_protocol_v1.json) |
| 主基线搜索 | [Static/Rule配置](../configs/baseline_tuning_v1.json) |
| E2：局部扰动和重启 | [析因设计](experiments/E2_factorial.md) |
| E3：候选先验和出口 | [E3设计](experiments/E3_candidates_and_escape.md) |
| E4：规模和结构迁移 | [E4设计](experiments/E4_transfer.md) |

E2完整设计、E3执行参数和E4原始距离适配还分别位于项目内对应专题工作树；总览已经集中解释其主要设置。它们继续属于完整研究范围。

## 想看已经完成的数值结果

- [FE校准曲线](reports/2026-09-09_fe_calibration.md)
- [E3四组Static的选择结果](reports/2026-09-09_e3_static_terminal.md)
- [各正式训练保存的进度](reports/experiment_progress_2026-09-09.json)

目前还没有GP正式测试结果。原先的按秒工程入口、任意GPU授权、文件摘要流程与历史“运行中”状态，均不能覆盖[当前规则](planning/29_runtime_simplification.md)。

[旧文档索引](reports/archive/docs_index_before_runtime_changes_20260909.md)保留各早期工程报告链接，供需要时追溯。
