# GPLSACO / GP-FACO

用遗传编程学习 FACO 的局部扰动与重启控制，研究反馈价值、两类控制的协同、候选图出口和跨规模迁移。

**本轮入口：[单树与三树50代实验——参数、曲线、比较和个体解释](docs/reports/单树与三树50代实验.md)。** 已完成的10代预实验见[分析结论](docs/reports/预实验结论与后续加速.md)。

当前执行 v2：蚂蚁数采用 2022 FACO 论文公式，只使用空闲 RTX A5000，ACO 和 GP 按评价次数运行。旧的 32 蚂蚁训练只读保存；新训练必须先确定数值后端、开发集训练预算并准备两类配对 FACO baseline。已有六个 TSPLIB 实例各 30 次的原始 FACO 复现结果；GP 正式测试尚未执行。

当前轮次为 **TSP500，修复后的单树/条件三树各3 seed×128个体×50代，从头初始化**，每实例1个ACO seed、1000迭代训练。使用全服务器空闲A5000；每5代轻量开发监控，结束后两级验证选模，六个Controller冻结后自动测试TSP500和TSP1000。运行 `.venv/bin/python scripts/run_representation_campaign.py --launch`，恢复用 `--resume --launch`；参数与依据见[本轮协议](docs/experiments/representation_50gen_v3.md)。

- [当前进度](docs/reports/progress.md)
- [文档导航](docs/README.md)
- [参数与依据](docs/design/implementation_v2.md)
- [128 个体在 GPU 上并行](docs/design/population_gpu_v2.md)
- [实验执行协议与命令](docs/experiments/protocol_v2.md)

| 目录 | 内容 |
|---|---|
| `python/gp_faco` | GP 进化、外部评分、数据和实验编排 |
| `cpp`、`cuda` | 原生 FACO 对照、CUDA 求解器和绑定 |
| `configs`、`scripts`、`tests` | 实验参数、入口和必要测试 |
| `docs` | 设计、实验协议、可读报告及 v1 历史 |
| `results` | 按版本保存的精简统计结果 |
| `artifacts`、`build`、`.venv`、`.tmp`、`.deps` | 本地原始结果、构建、环境和中间产物 |

所有写入都在本项目内，外部数据、参考源码和 MTGP_ACO 只读。文件、源码、二进制、坐标与实验管理记录不计算完整性哈希；算法内部的路线指纹保留用于档案去重。
