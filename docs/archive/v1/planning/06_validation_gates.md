# 实施验收门槛

每道门槛由具体报告证明，不因测试命令返回零而自动覆盖整项。状态记录在 `docs/reports/progress.md`。未实现的测试应标为 pending，不能添加永远通过的占位测试。

| 门槛 | 必须交付与检查 | 允许进入的下一阶段 |
|---|---|---|
| G0 来源与数据 | 外部输入只读；来源/许可/环境 manifest；纳入面板全量格式与标签目标核验；split 去重；原生构建与可行性 smoke | 无 GP CPU/CUDA 内核开发；原型可与审计同步 |
| G1 FACO 语义 | 源码直接操作 oracle；relocate/flip/position/cost；原生 MNE；候选/备用/全回退；LS 顺序和重激活；bounds/default | 共同底座算法行为可信 |
| G2 CUDA 与事务 | 同一操作 CPU/GPU；竞争和越界检查；Hard所有新增边；deadline迟到丢弃；restart完整reset；状态隔离 | 完整 GPU pilot |
| G3 GP 与完整评价 | IR三后端；错误输入拒绝；12 terminal 成本/值；无在线Python回调；全部个体重评；最终代/验证/checkpoint | 离线成本估算与正式协议冻结 |
| G4 预注册 | 按用户最新指示接通并校准500/1K评价次数限额/progress语义；Static/Rule调参、资源和冻结hash；五训练seed；统计比较和测试manifest锁定 | 完整G4后放行正式测试；全部自身依赖已冻结的训练可按planning/16与独立基线调优并行 |
| G5 E1/E2/E3 | 完整任务矩阵、所有失败/训练重复、配对统计、机制诊断和负结果 | 已冻结模型 E4 揭盲 |
| G6 E4与结论 | 原目标10K/TSPLIB、冻结签名、逐实例/分组表、资源成本、证据与主张对应 | 可审阅完整研究结果 |

## 专项验证用例

2026-09-09执行顺序细化见[planning/16](16_e1_freeze_and_execution.md)：完整FE校准已经通过，GP训练不读取基线成绩，因此在固定全部共同设置及基线搜索/验证规则后可以并行执行。此调整不降低正式测试门槛，不把尚未结束的基线调优或训练记录为已完成。

1. **距离/成本：** 小型手算点集和不同舍入规则；闭环索引；增量 delta 与完整重算；零距离、重复坐标与极值。完整 tour 不合法立即失败。
2. **MNE：** 相同显式选点序列在原生和移植路径中逐步计数一致；保存曾计入却最终撤销的新边反例，证明不能用最终改边数替代。
3. **候选：** 权重正、为零、下溢、所有已访问、备选耗尽；roulette端点与 padding；统计频数与输入概率匹配。候选先验乱序经 LS 距离视图后检查集合正确。
4. **LS：** 两类型 move、相邻/环边、较短补段、受影响节点重新进入 pending；接受上限、move-eval限额、截止分别触发。
5. **图约束：** Hard relocate 的移出补边与两个插入边逐条核查；2-opt两条新边；初始tour含图。Escape 槽位保持相等，例外预算不能无限增长。
6. **重启：** 用明显不同的旧stored/default/cache/EMA/ant状态初始化，事务后逐字段断言；没有alt时mask；GB不恶化；常规批次不能被旧ant精英恢复。
7. **IR：** 差异测试覆盖非交换SUB/AQ、FP32/ERC往返、剪裁顺序、稳定ID、非法版本/栈/深度/常数、合法mask少于32和近似平局。
8. **计时：** 可控延迟的完整批次，检查deadline前后提交；最后批被丢弃、实际超限被记录；缓存收费与独立端到端核对；不使用标签早停。
9. **调度：** 同程序重复状态隔离；task乱序返回仍正确归集；所有精英在新panel实评；异常任务不从平均剔除；恢复同一任务而非另抽随机样本。
10. **评价次数：** 每colony完成精确的完整蚂蚁tour次数；零限额保留共同初解；拒绝非整批限额和旧特征版本。主机延迟、插桩及准备缓存不能改变FE进度或搜索轨迹；无墙钟截止、准备扣费或迟到批丢弃。内部LS移动检查次数单独记录。

## 验证层级与报告

CPU 单元/差异测试优先在本地执行；CUDA 测试在空闲目标 GPU 上跑并记录 UUID/driver/compiler。Release 下测试仍使用不会被 NDEBUG 删除的断言。memcheck 和 synccheck 报告保存原日志与摘要；性能 pilot 不在 sanitizer 模式中计时。

性能只在正确性相关门槛通过后解释。每个结果表注明已验证范围；IR评分通过不代表完整FACO通过，原生内部检查通过不代表独立 evaluator 核验已完成。

## 当前deadline证据与未覆盖项

[批量Engine报告](../reports/2026-09-08_batch_engine.md) 已覆盖可控单调时钟、截止处提交、真实更优迟到候选拒绝、昂贵准备中断保留廉价解、缓存分阶段费用、重新准备、重复/乱序任务、上一任务GB不泄漏，以及32-colony开发面板的独立路线核验。GPU检查同时回归共同构造/LS和固定迭代路径。

这只关闭G2中的批量状态隔离与当前提交边界验证项。Hard全新增边、Escape同槽位/例外、epoch重启全字段事务、动作特征、GPU内核的时间粒度校准仍待实现或补充；不会因为4项CTest和49项Python测试通过而把G2/G3整体标为完成。

后续 [Hard操作报告](../reports/2026-09-08_hard_operations.md) 已补齐全新增边谓词、CSR无向化实际成员、CPU/CUDA受限选择/LS的操作证据；6项GPU CTest和49项Python测试通过。Hard在完整Engine的初始/截止/费用集成、Escape、重启及特征仍pending；G2整体状态不变。

## 主底座控制层新增证据

[控制Engine报告](../reports/2026-09-08_control_engine.md) 补齐区域起点、全部十二特征、GP实际动作、双指纹及完整邻接去重、档案裁剪、pre-action旧信息素、epoch完整重启、反馈和跨任务隔离的核验。人为GPU指纹碰撞仍保留结构不同的档案成员。重启前保留GB/档案/全局stagnation，重启后逐字段检查工作tour、逆映射、stored/default/products、visited/pending/scratch/gains及ant统计。

当前8项GPU CTest、62项Python测试及三种CUDA工具通过，CPU Release和ASan/UBSan各4项CTest通过。G2主底座控制/事务部分已有工程证据；E3完整Hard/Escape预算集成仍待完成。G3的IR→原生在线控制已接通，worker调度、真实个体重评、最终代/验证/checkpoint及分项成本仍pending；G4未冻结，E1–E4未执行。

[worker报告](../reports/2026-09-08_worker.md) 进一步补齐显式spawn、协调进程无CUDA导入、task身份、同PID连续任务、Future超时不重启、容量重建、准备错误恢复、缺失解失败保留和两规模外部宏平均。CPU Python回归42通过/29个GPU专属跳过，目标GPU上73通过；64个开发成员独立核验通过。DEAP代际、精英重评、最终代/验证/checkpoint、成本剖析和G4冻结仍未完成。

## 真实代际与恢复新增证据

[训练报告](../reports/2026-09-08_training.md) 补齐标准DEAP、精英/重复树重评、最终已评种群、固定验证和可信JSON恢复。实际A5000运行8×3，48训练任务＋2验证任务，1,600条路线核验通过；原始seed重放三代IR/fitness/RNG一致。第5任务暂停后保留原任务结果和费用；新worker实际重新测量不会改变扣费。Python 103和原生CTest 8项通过。

首轮数据成员归档存在JSON整数键摘要缺陷，已保留失败证据并修复，在新目录完整重跑验收。真实pilot的shortlist退化为1候选，多候选排序/全失败无导出由编排回归覆盖。G3的协调/特征/档案分项成本仍pending，G4未冻结，不能据此次开发训练标记E1–E4完成。

## 成本证据与用户最新次数协议

[成本报告](../reports/2026-09-08_profiling.md)补齐当前底座的CPU准备、特征/区域、GP、动作/重启、融合构造/LS、信息素/档案、D2H、外部核验与协调总成本。1,156条件、13,056路线完整审计通过；1,088组单/批量轨迹一致。GPU 9项CTest、105项Python及三个CUDA工具通过，CPU Release/ASan各4项通过。融合阶段以event墙钟和block周期工作分布分别报告；显存为显式缓冲加边界采样，不宣称精确瞬时峰值。

旧时间入口424任务、5,136路线也完整通过工程校准。用户随后要求按evaluation次数作为进化/ACO主标准，因此G4须先接通[次数协议](12_evaluation_count_protocol.md)，这些秒数不冻结为主预算。G3旧路径的分层成本已有实测，新次数入口/特征版本和训练恢复需按变更范围验收。

## 评价次数路径新增证据

[次数报告](../reports/2026-09-08_evaluation_counts.md)补齐原生完整FE提交、延迟/插桩/准备模式不改变轨迹、feature_spec_id=2及真实DEAP恢复。GPU CTest 10项、Python 110项、memcheck/synccheck/racecheck和CPU Release/ASan各4项通过。两规模各256 FE/colony的8×3开发运行完成50任务、400批、409,600 FE，1,600条路线独立核验通过；实际暂停/恢复保持已完成结果和准备资源记录，扣费为零。

G3次数路径已有当前变更范围内的验收。G4仍须开发次数档校准、Static/Rule充分调参、五演化seed和测试manifest冻结；此小规模训练不构成E1结论。

[基线报告](../reports/2026-09-08_baselines.md)进一步补齐Static/Rule原生动作选择、独立重启随机流、停滞升级及epoch冷却。32组等动作GP完整轨迹、1,168次实际动作和12组单/批量结果一致；GPU CTest 12/Python 114、CPU Release/ASan各5项及三个CUDA工具通过。基线配置搜索/完整development调参仍未执行，G4状态不变。
