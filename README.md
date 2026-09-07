# GPLSACO / GP-FACO

研究局部搜索之上的结构控制：用一棵遗传编程树选择 FACO 的重启、扰动起始区域和原生 MNE 阈值。研究依据为 [v4 方案](docs/design/GP_ACO_TSP_Research_Proposal_v4.md)，不预设 GP、重启或图外探索一定有收益。

**状态：研究启动与正确性基础建设。尚无正式训练结果或 E1–E4 结论。**

- [详细研究计划与阅读顺序](docs/README.md)
- [准备情况、实际证据和下一步](docs/reports/progress.md)
- [环境、GPU 与复现实务](docs/planning/08_execution_and_resources.md)
- [数据与标签协议](docs/planning/02_data_and_labels.md)
- [可复核的启动结果](docs/reports/2026-09-08_bootstrap.md)
- [主池划分与CPU FACO语义核验](docs/reports/2026-09-08_data_and_cpu_semantics.md)
- [CUDA构造与局部搜索操作核验](docs/reports/2026-09-08_cuda_operations.md)
- [固定迭代GPU FACO与开发池结果](docs/reports/2026-09-08_fixed_faco.md)
- [构建、CPU/GPU 检查与原生试跑命令](docs/reports/reproduce.md)

已完整核验500/1K主池128,416条记录并发布划分。CPU FACO操作、DEAP→IR→CPU/CUDA评分，以及包含真实随机选点/信息素的固定迭代GPU流程已核验；RTX A5000上开发池48次运行全部通过独立路线重算。来源适配、共享内存竞争修复和各阶段证据均有记录。正式wall-clock/GP Engine、训练闭环和E1–E4仍在后续工作中。

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
