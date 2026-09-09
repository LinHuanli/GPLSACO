# GPLSACO / GP-FACO

用遗传编程学习FACO的重启、扰动起始区域和MNE阈值，研究反馈、两类控制的协同、候选图出口和跨规模迁移。

**先读：[实验进展总览：做了什么、参数怎么设、现在有什么结果](docs/reports/实验进展总览_2026-09-09.md)。**

当前E1和E3已进入正式训练，四组E3 Static调参完成；尚无完整GP正式测试结论。训练进度已保存，正在取消运行管理中的重复完整性校验。恢复后只使用空闲RTX A5000，进化和ACO按evaluation次数，不设算法墙钟上限。

- [当前进度与下一步](docs/reports/progress.md)
- [研究计划与文档导航](docs/README.md)
- [原始v4研究方案](docs/design/GP_ACO_TSP_Research_Proposal_v4.md)
- [当前资源和运行规则](docs/planning/29_runtime_simplification.md)

## 项目目录

| 目录 | 内容 |
|---|---|
| docs/design | 原始研究方案 |
| docs/planning、docs/experiments | 参数、方法和E1–E4设计 |
| docs/reports | 实测结果与当前总览；历史报告保留当时状态 |
| python/gp_faco | 数据、GP进化、外部评价和任务管理 |
| cpp、cuda | CPU参考、CUDA求解和绑定 |
| configs | 实验参数和当前运行规则 |
| scripts、tests | 实验入口与必要的实现检查 |
| provenance | 历史来源记录、第三方声明及已选参数 |
| artifacts、build、.venv、.tmp | 本地结果、构建、环境和临时工作树，不提交原始数据与二进制 |

所有工作都在本项目目录内。外部Datasets和references只读。历史方案中的文件摘要要求与旧GPU授权已由最新运行规则替代。
