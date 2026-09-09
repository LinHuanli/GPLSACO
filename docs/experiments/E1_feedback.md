# E1：反馈是否改善样本外表现

当前采用 [v2 协议](protocol_v2.md)。旧 32 蚂蚁训练未完成，全部只读保存；新训练从相同五个进化 seed 的新种群开始。没有正式测试结果。

## 比较什么

主要比较 GP-Full 与原始 FACO 2022、GPU FACO 无 GP、经过完整配置搜索的 Static、Rule，以及独立重训的 GP-NoFeedback。五个主要比较使用 Holm 校正。主要终点为 TSP500/TSP1000 在 5000 迭代下按实例和规模等权汇总的 reference gap。

NoFeedback 只删除 `stagnation`、`return_rate`、`ls_work` 三个 LS 后反馈终端；保留评价次数进度、结构和信息素信息。它独立训练，不能拿 Full 的树临时屏蔽反馈代替。

Full/NoFeedback 各 5 个进化 seed、128 个体、50 个完整评价代；同一代使用相同实例和求解 seed。每 5 代冠军在固定开发监控面板评价。最后所有不同的代际冠军与最终种群进入完整验证，5000 迭代选择每个进化 seed 的程序。测试每规模 128 实例，每实例 10 个求解 seed，方法封存后才运行。

训练和候选验证采用 [种群 GPU 入口](../design/population_gpu_v2.md)，一次展开至多 128 个体；重复程序仍完成全部 FE。并行顺序不参与随机 seed 派生，监控和配对 baseline 的定义不变。

Static/Rule 保留 [原登记配置族](../../configs/baseline_tuning_v1.json) 的全部策略组合；旧文件中 32 蚂蚁和 FE 数不再使用。v2 入口 `scripts/tune_baselines_v2.py` 在开发集每规模前 64 个实例筛选，每类前 10 个候选在完整验证集 256 实例/规模、5000 迭代选择。两个阶段均为 seed 17/29。

## 怎样解释反馈

状态分叉继续保留：在同一完整状态、同一 reference 和 region 上，配对比较 MNE 2 与 16，随后用相同的继续策略和额外 FE。两条分支分别恢复原始状态，保留原进度分母和批次编号。必须预先指定分层、分叉 seed、每层上限、继续策略和缺层处理，不能只挑 Full 成功的轨迹。

原生保存/恢复、单 colony 选择和分叉接口已合入主分支并通过语义测试；正式状态分叉数据仍未生成。[历史详细设计](../archive/v1/planning/21_e1_state_fork_engine.md) 可用于理解接口，运行参数以 v2 为准。

结果需要同时报告效果大小、实例簇配对 bootstrap 区间、每个进化 seed 的结果和失败率。胜过未调优 Static 或少量工程实例，不足以支持反馈价值结论。
