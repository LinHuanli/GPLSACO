# E3 实际返回后的 GPU 资源偏差恢复

`GP-alpha-Hard-5519` 在 cuda00 的原 UUID 上完成了第 66 次原生求解，但结束时观察到一个外来 compute process。严格资源守卫保存原返回后令 CLI 以 1 退出；此前 65 次求解已入账。该事件属于真实资源争用，不能写成独占 GPU 或查询故障。原队列句柄 37720 已确认退出，原 worker PID 3010338 已不存在；其他队列继续运行。

## 原证据与执行偏差政策

原 checkpoint、manifest、待核验任务、独立原生返回、CLI 资源记录及队列收据封存在 `artifacts/e3/resource-recovery-v1/original-alpha-Hard-5519/`。其 `evidence.json` SHA256 为 `b866df517f5dde1fed4b80dd6a6479fb65b76e54e63603823e3b8302fa08622f`，另列出此前全部 67 条完成记录的摘要（两条准备、65 条求解）。

待处理任务 `14992391d33c2ce02ef45c770bad256beda7d4300897270f85668cf865ab4675` 已有唯一实际返回：原生状态 completed，128 批、131,072 FE、32 个成员；后观察保留 `foreign_processes=1`、显存 1,249 MiB、利用率 98%。原始求解结果不再提交，也不因资源观察重新抽样或改变失败评分。

[附加政策](../../provenance/e3_resource_recovery_v1.json)在正式 TEST 放行前登记，统一适用于 E3 的 GP 和 Static。依据是原次数入口及 progress 特征均使用 FE，GPU 占用和墙钟时间不进入求解器输入。政策保留实际返回并交给原 evaluator；受干扰任务及包含它的整次运行计时明确分类，排除于独占设备的速度比较。新的未提交任务只有观察到严格整数 0 个外来进程后才准入；繁忙或未知时等待，不设算法时间上限。

原科学执行 `f70f8691c8643f346ddb44ab07f6a87fe026d61ba89cf19dd06c3f1d4d2c6cbb`、20 个 GP 的 128×50、四个 Static 的 160 配置族、实例、seed、FE、图、native、评分与选择规则均保留。原 host/UUID 仍须精确匹配。恢复通过新增独立入口 [resume_e3_observed_return.py](../../scripts/resume_e3_observed_return.py) 实现，不改写冻结模块或原执行 manifest。原科学身份与新增编排实现分别记录，不能把恢复后的全部实现宣称为原提交 7fbe5c8。

每次资源偏差先原子发布原始 task 字节，再写独立 journal，验证实际返回与独立 raw-return 一致后才继续原核验和入账事务。恢复 sidecar 固定政策和新入口的摘要；独立审计复核 journal、全部原始返回、原完整训练身份、DEAP 重放与 FE 账本。

## 验证与实际恢复状态

`tests/python/test_e3_resource_recovery.py` 的七项针对性检查通过，涵盖实际返回后资源争用/查询未知、提交前等待、非法返回仍失败计分、原始返回篡改拒绝、Static 共享语义，以及原始字节发布与 journal 发布之间中断的恢复。恢复演化、选择和 FE 与无资源偏差的固定随机输入对照一致，待核验返回未发生新增 native 提交。最终记录为 `.tmp/pytest-e3-resource-recovery-v4.log`；Ruff 与差异格式检查通过。

实际恢复分两步：先只入账原第 66 次返回并暂停，独立审计和比较原收据；通过后继续同一运行的完整训练。此文件初次提交时实际恢复尚未启动，后续实测以追加报告和机器证据为准。原始异常、失败 CLI 及队列记录保留，不重建为从未中断的运行。E1–E4 与 E3 完整训练、选择、正式 TEST 均未完成。
