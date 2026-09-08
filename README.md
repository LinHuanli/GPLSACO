# GPLSACO / GP-FACO

研究局部搜索之上的结构控制：用一棵遗传编程树选择 FACO 的重启、扰动起始区域和原生 MNE 阈值。研究依据为 [v4 方案](docs/design/GP_ACO_TSP_Research_Proposal_v4.md)，不预设 GP、重启或图外探索一定有收益。

**状态：主底座GP、worker和完整DEAP训练/验证/恢复已通过当前开发验收。尚无正式训练结果或 E1–E4 结论。**

按用户最新要求，进化与ACO主实验按评价次数终止，默认训练入口不设求解秒数上限。原生次数入口、`progress`特征版本、完整训练及恢复已通过[开发验收](docs/reports/2026-09-08_evaluation_counts.md)；计数口径和正式冻结顺序见[最新协议](docs/planning/12_evaluation_count_protocol.md)。

- [详细研究计划与阅读顺序](docs/README.md)
- [准备情况、实际证据和下一步](docs/reports/progress.md)
- [环境、GPU 与复现实务](docs/planning/08_execution_and_resources.md)
- [数据与标签协议](docs/planning/02_data_and_labels.md)
- [可复核的启动结果](docs/reports/2026-09-08_bootstrap.md)
- [主池划分与CPU FACO语义核验](docs/reports/2026-09-08_data_and_cpu_semantics.md)
- [CUDA构造与局部搜索操作核验](docs/reports/2026-09-08_cuda_operations.md)
- [固定迭代GPU FACO与开发池结果](docs/reports/2026-09-08_fixed_faco.md)
- [固定并发Engine、预算扣费与截止核验](docs/reports/2026-09-08_batch_engine.md)
- [Hard全新增边与受限CPU/CUDA操作核验](docs/reports/2026-09-08_hard_operations.md)
- [主底座GP控制、档案、完整重启与开发计时](docs/reports/2026-09-08_control_engine.md)
- [常驻GPU worker、任务身份与外部fitness](docs/reports/2026-09-08_worker.md)
- [真实DEAP训练、固定验证与checkpoint恢复](docs/reports/2026-09-08_training.md)
- [完整动作成本剖析与旧时间入口校准](docs/reports/2026-09-08_profiling.md)
- [评价次数入口、progress版本与实际训练恢复](docs/reports/2026-09-08_evaluation_counts.md)
- [构建、CPU/GPU 检查与原生试跑命令](docs/reports/reproduce.md)

已完整核验500/1K主池128,416条记录并发布划分。主底座GP Engine已接通区域/十二特征、实际动作、档案和完整重启。RTX A5000次数模式开发训练8×3的50任务完成409,600次tour evaluation，1,600条路线及恢复通过独立核验。成本矩阵1,156条件、13,056条路线和旧时间入口424任务、5,136条路线另保留原身份。正式次数档、Static/Rule调参、G4冻结及E1–E4仍待完成。

## 目录

```text
docs/design/          原始研究方案（保留原文）
docs/planning/        决策、接口、验收和资源计划
docs/experiments/     E1–E4 独立实验协议
docs/reports/         实测报告和进度
python/gp_faco/       数据、DEAP、程序 IR、外部评价
cpp/include/gp_faco/  C++ 公共契约
cpp/src/             CPU 参考与绑定
cuda/                CUDA 实现
scripts/             构建、审计、实验入口
configs/             显式配置；pilot 与正式冻结配置分离
provenance/          来源指纹、环境锁和第三方声明
tests/               科学正确性与跨后端校验
artifacts/           本地产物，不进入 Git
build/ .venv/ .tmp/   本地构建、环境、临时文件，不进入 Git
```

外部数据和参考实现按路径读取，不在其目录运行产生文件的构建或实验命令。仓库为 <https://github.com/LinHuanli/GPLSACO>。
