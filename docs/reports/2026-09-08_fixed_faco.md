# 固定迭代GPU FACO流程与开发池核验

日期：2026-09-08（NZ）。证据见 [fixed_faco_results.json](fixed_faco_results.json)。本轮将坐标距离、真实设备随机选点、稀疏信息素、批次最好解和持久显存接入了共同构造/LS内核，并开始在预留开发池运行完整固定迭代流程。正式wall-clock Engine、GP训练和E1–E4仍待完成。

## 实现身份与状态

该实现登记为 **FACO-GPU-IterationPilot**。每个 `FixedFacoGpu` 对象持有一个实例、一个colony及其蚂蚁缓冲区；Python一次 `run_iterations` 调用由C++/CUDA完成全部批次，无逐步Python回调。构造、LS与旧显式选点诊断共用 `cuda/faco_device.cuh`，没有另写第二套路线更新。

| 环节 | 本轮明确实现 |
|---|---|
| 距离与候选 | FP64坐标视图，`sqrt(dx*dx+dy*dy)`；NN候选按距离/节点ID排序，LS单独视图；没有n²持久矩阵 |
| 开发初始化 | 节点0起始的最近邻tour，再用同一CPU checklist 2-opt与显式评价上限改进；固定一个初始解 |
| 默认配置 | 32 ants、主候选16、备用64、LS候选20、beta=1、保留比例0.5、p_best=0.1、epoch强化概率0.01；MNE显式传入 |
| 随机选点 | cuRAND Philox；保留53个随机比特得到[0,1)，主候选roulette、全零均匀回退、备用首合法及全局最近回退 |
| 批次状态 | 所有ant复制同一冻结parent；只归约当前批次；分别维护GB与epoch-best；实际强化tour成为下一批parent |
| 信息素 | 按有向候选条目独占写入；对称tour两方向分别匹配；stored/default正确蒸发、按epoch成本更新bounds并刷新产品缓存 |
| 跨调用 | 每次run重置parent、GB/epoch、stored/default、产品缓存、ant/visited/checklist；相同对象仅允许一个活动评价 |
| 持久内存 | 注册时分配、跨seed复用；本阶段同时预留诊断轨迹缓冲，普通run不写/下载轨迹 |

初始化与原生 `par_build_initial_routes` 的多初始解＋2-opt/3-opt流程不同，且当前起点仍全tour均匀。它是开发阶段的完整迭代实现，不冒充未修改Native或已冻结的32动作FACO-Control。后续控制器共享的初始化与区域定义仍须校准、登记和统一。

随机流以坐标派生的实例key和solve seed生成有序混合key；Philox子序列由batch/ant编号确定，构造step各占counter区。起点拒绝采样使用保留counter区，colony强化决策使用保留ant编号，避免因某只蚂蚁多走一步而改变其他蚂蚁的流。任务到达顺序不参与随机key。当前验证覆盖重复执行与跨seed重置；大规模随机统计和多colony调度仍需后续验收。

## 逐批重放与检查

人工实例包含n=7/17/100/500/1000，窄/宽候选、重复坐标、保留比例0.25/0.5/0.75/0.9及epoch强化概率0/0.01/1/0.3。C++测试下载诊断轨迹，用CPU重新执行同一实际随机数下的选点、构造、LS、批次归约和信息素操作。

- 20组colony配置、1,280个蚂蚁批次通过；61,654次实际选点一致，其中2,744次备用选择、13,996次全局回退。
- 构造tour/MNE/checklist、最终tour/positions、LS工作量、GB/epoch、强化源与下一批parent一致；最大成本误差4.9737991503207013e-14，最大信息素误差6.9388939039072284e-17。
- 每个配置先运行seed17，再运行seed29，再恢复seed17；关闭轨迹后的路线、成本与工作量一致。零批次调用恢复初始状态。
- 原有2,160组显式选点CPU/CUDA对照重新通过，Python GPU环境下39项通过。
- memcheck、synccheck、racecheck各运行20配置、240个蚂蚁批次及同样的重置检查，均为0 errors；racecheck亦为0 warnings/0 hazards，三种模式结果一致。
- CPU原生对照补充非0.5的信息素保留率并通过：原生底层 `evaporate` 接收蒸发比例，`ACOModel`传入`1-rho`；项目API直接接收保留比例。只测试0.5会掩盖该接口含义差别。

## 预登记开发池的实际运行

仅使用已发布split中的500/1K开发ID，每规模8实例，各seed17/29/43，32 ants×50批次、MNE8。输入只含坐标、参数与坐标点集key；返回后由独立Python evaluator读取标签、验证排列并用 `math.dist`/`math.fsum` 重算。manifest保存在机器可读报告中，完整tour留在 `artifacts`。

| 规模 | 合法运行 | 最大独立重算误差 | 注册准备中位秒 | 50批次求解中位秒 | 程序设备缓冲字节 |
|---|---:|---:|---:|---:|---:|
| 500 | 24/24 | 4.62e-14 | 0.00691 | 0.54740 | 1,271,248 |
| 1K | 24/24 | 4.62e-14 | 0.01799 | 1.13301 | 2,529,248 |

48次均优于各自初始解，没有负reference gap。开发样本的平均reference gap分别为0.8704%和1.4483%，仅用于描述本次试跑；不能外推为测试集表现、GPU加速或GP贡献。设备缓冲统计包含预留诊断空间，不含driver/context、分配器碎片和其他运行时开销，不能当作实际显存峰值。

执行设备为cuda02的RTX A5000，UUID `GPU-056fae3f-b504-efe0-2d9d-b1186860e643`。各GPU任务启动前实时检查compute进程、显存和利用率；检查与试跑均已正常结束。

## 计时边界与下一步

注册准备计时包含CPU候选生成、开发初始解和设备缓冲准备；`solve_seconds`包含run内重置、全部批次、必要同步与结果下载。相同对象多次run复用注册内容，准备时间单列。本轮固定迭代没有进行候选缓存收费、deadline前incumbent提交或迟到批次丢弃，因此这些耗时只供工程分析，不能作为正式同预算比较。

下一步实现wall-clock Engine：将廉价可行解、分阶段准备费用、同步完成时刻、迟到结果处理和多实例固定batch shape一并验证；随后接入Hard/Escape、参考档案/重启事务、区域与12个特征、GP决策和worker评价闭环。单colony固定迭代入口不会替代这些要求。
