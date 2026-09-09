# 评价次数主入口与训练恢复验收

2026-09-08，按用户补充把进化与ACO主实验改为evaluation次数终止。默认`train_gp.py`已使用次数配置，求解入口不接受秒数预算；主机和GPU实际时间仅记录资源消耗。正式研究仍需次数档校准、Static/Rule调参及G4冻结，本报告不构成E1–E4结果。

机器证据：[训练独立审计](../../../../results/v1/evaluation_count_training_results.json)、[回归与运行资源](../../../../results/v1/evaluation_count_checks.json)。原始产物位于`artifacts/gpu/evaluation-count/`，实际训练目录为`training-v1/`。

## 计数及兼容边界

当前暂按一只蚂蚁完成构造与局部搜索、产出带成本的完整候选tour计一次search-tour evaluation。每colony是一个实例/求解seed，限额不以整面板总量替代。内部LS候选移动检查、构造步数和共同初始准备另列；此计数不等于全部底层目标函数调用次数。精确口径已向用户询问，正式冻结前可按回复调整，详见[次数协议](../planning/12_evaluation_count_protocol.md)。

次数入口按完整蚂蚁批次提交，要求限额是ants的整数倍，不隐式取整。零FE仍获得共同初解而不启动蚂蚁搜索。缓存准备与重新准备不扣FE；没有“很大秒数”的隐藏截止，也没有墙钟迟到批丢弃。单次LS仍受共同的移动检查/接受移动上限约束。

`feature_spec_id=2`将终端0显式定义为`progress=已完成FE/单colony限额`，其余编号与数值规格保持原身份。IR导出、DEAP交叉/变异、JSON恢复和任务hash均保留版本。次数入口拒绝旧elapsed模型，旧时间入口拒绝新progress模型。历史wall-clock实现、配置和结果保留原身份。

## 针对性回归

GPU为cuda04的RTX A5000，UUID `GPU-34b223c6-7502-b097-19e0-a411b1708f06`，driver 610.43.02。GCC 15、CUDA 13.3、SM86及禁用FMA融合的编译约定沿用。当前验收二进制SHA256为`2269faaafd558e2ec5e0f8ec770fc89c357cc231ae9a69092c9391a6fb747b66`。

- GPU原生CTest **10/10**、Python **110/110**；CPU Release和ASan/UBSan各 **4/4**。
- 新次数语义在memcheck、synccheck、racecheck下均无错误，racecheck无hazard/warning。
- 4 colonies×8 ants×6 batches精确完成每colony48 FE；4组单/批量结果与控制状态一致。
- 强制首批延迟200 ms、开启诊断插桩、切换缓存/端到端准备，完整特征、动作、档案、信息素及工作tour不变。
- 每批进度与CPU/CUDA评分核验；旧缓存费用设为十亿秒仍不改变FE。CPU时钟达到`1e100`时次数账本仍允许提交，不依赖有限时间替代值。
- 覆盖零FE、非整批、负数、浮点、布尔、超范围输入和特征版本互拒；DEAP连续变化及恢复保留v2。

首轮GPU检查有108项Python通过、2项失败：负FE经pybind转换后抛出通用RuntimeError，未满足参数错误契约。修复为显式非负整数解析后，完整10/110及三个CUDA检查通过。失败日志、旧绑定源码和旧二进制保留于`initial-checks/`，未删除或归入成功运行。

## 真实开发训练及恢复

两个规模各有48个development训练实例和16个不重叠development验证实例；未读取正式测试成绩。种群8、代数3、演化seed 1103；每个规模的完整面板为16实例×2求解seed、每colony32 ants和256 FE，即8个完整批次。

| 项目 | 实际值 |
|---|---:|
| 完整个体fitness评价 | 24 |
| 训练/固定验证原生调用 | 48 / 2 |
| 提交批次 | 400 |
| 训练search-tour evaluations | 393,216 |
| 验证search-tour evaluations | 16,384 |
| 总search-tour evaluations | **409,600** |
| 搜索内LS候选移动检查 | 1,889,552,857 |
| 搜索构造步数 | 3,942,502 |
| 独立核验返回路线 | 1,600，全部合法 |
| 最大成本重算绝对误差 | 7.46e-14 |
| 丢弃批次/墙钟扣费/超限 | 全为0 |

在第5次求解后正常暂停，复制原checkpoint，再实际启动新worker恢复。原7条完成记录（包括2次准备）及准备资源历史保持不变；32个实例恢复后重新测得的准备时间发生变化，但未产生时间或FE扣减。训练、恢复、验证分别使用3个真实spawn进程。审计从原始演化与面板seed重放三代全部IR、fitness及最终RNG；精英和重复树均重新评价，最终代已完成评价，固定验证及选择与原始日志一致。

本次发生4次交叉、3次变异，无长度限制回退；shortlist退化为1个候选。选中`AQ(stagnation, mne_level)`，3节点，开发验证宏平均gap为4.537978%。它只用于验收完整数据流；单一小种群、单演化seed及单候选不能支持控制器效果或正式选择结论。

## 实际资源与下一步

暂停和恢复两个CLI的GNU time墙钟分别为8.27 s和45.75 s，共54.02 s；CPU user合计48.42 s、system合计2.08 s，两个CLI的max RSS最大250,032 KiB。累计worker记录32.6701 s、原生调用30.0694 s、外部evaluator 1.1610 s。原生时间包含于worker时间，不能相加后称为总资源；完整CLI时间另含启动、数据/身份核验、日志和checkpoint。

这些数值都是实际资源测量，不是算法截止参数。准备资源记录及内部LS工作量由原始任务保留；完整阶段成本矩阵另见[历史剖析](2026-09-08_profiling.md)，其源码及二进制身份不被本次迁移覆盖。

下一步在共同次数底座接入真实Static/Rule控制器，按development池确定短/中/长FE限额并充分调参，随后冻结五演化seed、方法配置和测试manifest。256 FE及8×3属于开发验收配置，不能直接当作正式128×50主实验。E3完整Hard/Escape和E4目标接口仍独立待完成。
