# 公共实验、结果格式与统计

2026-09-08用户补充后的主协议以[evaluation次数](12_evaluation_count_protocol.md)终止。以下旧wall-clock主预算描述保留为时间入口工程/可能补充实验的历史计划；正式主终点、配置和横轴须按新次数协议冻结。

## 1. 预算和公平比较

只用开发池的统一底座校准短/中/长预算、ants、batch shape 与 LS move-eval 上限；中预算是主终点，其他预算检验稳定性。预处理、特征、GP、重启、LS和维护均计入；缓存规则对所有控制器相同。报告 GPU 批量预算和单实例端到端两个独立执行配置。

Static/Rule 的离线配置搜索也登记总运行数和 GPU/CPU小时，不能只选一个方便击败的 MNE。先冻结共同底座、基础候选、区域生成器、archive、Tie/EMA/LS，再比较学习；所有变体共享同一内核。外部 CPU/GPU 算法分别列硬件和实现身份。

## 2. 原始结果与失败

每 task 一条结构化记录：实验/协议版本、instance/parent ID、split、program hash、evolution seed、solve seed、配置/source hash、host/GPU UUID/型号/driver、threads/batch shape、budget、preprocess charge、cost、label status、gap、validity/status、commit time、overshoot、late-work count、work counters和少量anytime检查点。

运行目录含不可变 `task.json`、日志、返回tour（主要至少最终tour）、独立重算结果、`result.json` 与终态标志；先写临时文件再原子rename。文件锁不是存活证据：恢复先查实际PID/会话/调度job，确认终态或句柄缺失才重跑，观察超时不重启。

可行性/程序错误计为算法失败；明确的基础设施失败按冻结的重试次数重跑相同任务。仍失败则保留记录，报告失败率及包含惩罚的分析，不能删除难例获得低平均gap。+infinity不混入常规bootstrap产生无意义区间；完整病例摘要只能作为有明确失败率的补充。

## 3. 统计单位与主比较

建议合成每实例10个solve seeds、TSPLIB20个；以实例为配对单位先汇总求解重复。每个 evolution seed 的验证选定程序均报告；同时给出跨实例与跨训练重复的不确定性。不得把10 seeds当10个独立实例，或只报告最好的训练seed。

以完整实例为簇做配对bootstrap（预定10,000重采样，固定统计seed），95%区间。针对GP还报告按训练seed分别计算的效果和两层重采样敏感性。E1三项主比较采用配对检验并Holm校正；方向定义为“对照gap−GP gap”，正值代表GP更好。检查点分析不新增一批未声明主检验。

E2在每个实例内先形成四格损失和交互；E3以图/限制/控制器条件分组；E4按规模和原距离类型报告，不能把100易例加权淹没10K负结果。效果大小、CI和实际收益均报告，p值不替代意义。

## 4. 输出最小集

- E1质量表/anytime＋状态分叉收益图；E2四格表＋交互区间；E3四图条件下Static/GP成对结果和覆盖率；E4规模曲线＋10K anytime＋TSPLIB逐实例表。
- 一张统一资源表：候选/传输、特征＋GP、构造、LS、信息素/档案占比、峰值显存、总离线小时。
- 所有主表从原始JSON由脚本生成，图用标准绘图工具输出独立SVG/PDF/PNG；记录任务集合hash和统计配置。

正式结果生成前检查全部预注册任务是否有终态、成本重算与标签定义是否一致、未按测试删实例/挑seed。缺任务的表标为partial，不能视为完整实验。
