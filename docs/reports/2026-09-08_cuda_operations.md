# CUDA FACO构造与局部搜索操作核验

日期：2026-09-08。实现依据为 [已直接对照作者源码的CPU层](2026-09-08_data_and_cpu_semantics.md)，结果见 [cuda_faco_results.json](cuda_faco_results.json)。此阶段完成显式选点下的GPU构造与checklist 2-opt；完整GPU FACO及E1–E4仍未完成。

## 实现与核验对象

`cuda/faco_operations.cu` 每只蚂蚁使用一个128线程block。路线重定位与反转通过独立scratch协作搬移，之后修复inverse positions；候选gain并行计算，按原生两类检查顺序归约为严格最大值，tie保持首遇顺序。MNE和构造checklist来自原始参考tour与移动前端点；LS允许已处理端点重新入队，保留接受上限n和共同底座的有限评价上限。原生反转在first=0时的数组结果也予以保留。

诊断输入包含同一个显式FP64对称距离矩阵、LS候选成员、各蚂蚁初始tour、给定选点排列及MNE目标。候选成员先建立真实距离有序视图。输入不含标签。矩阵路径限定n≤1024，仅用于隔离距离运算差异的操作核验，不能作为10K内层数据表示或完整性能基准。

## 实测结果

设备为cuda02的NVIDIA RTX A5000，UUID `GPU-056fae3f-b504-efe0-2d9d-b1186860e643`，driver 610.43.02；正式启动前检查该UUID没有计算进程且显存/利用率满足空闲阈值。CUDA架构86，禁用FMA contraction并保留line info。精确构建环境延用启动阶段记录。

| 检查 | 覆盖与结果 |
|---|---|
| CPU/CUDA操作差异 | n=3/4/5/7/8/17/100/500/1000，连续/舍入整数/重复坐标三类，每组16蚂蚁 |
| MNE与评价预算 | MNE=2/4/8/16，LS评价上限0/1/7/127/无限；合计2,160组 |
| 逐项输出 | 构造tour、最终tour、inverse positions、checklist及全部构造/LS计数一致 |
| 实际工作量 | 12,110次构造转换，1,935,832次LS评价，158,214次重新入队 |
| FP64成本 | 相对CPU增量值及完整重算的最大绝对误差1.1084466677857563e-12 |
| Python回归 | GPU环境下33项通过，包括原有程序评分与新增数据索引测试 |
| memcheck | 540组，0 errors |
| synccheck | 同一540组，0 errors |
| racecheck | 同一540组，0 hazards、0 errors、0 warnings |

sanitizer面板仍覆盖全部规模、距离类型、MNE档位和评价上限，只将每组蚂蚁数降为4；三种sanitizer输出的路线核验与计数结果相同。测试过程没有访问主研究测试池性能。

## 首轮发现及修复

首轮数值、内存和同步检查均通过，但racecheck报告共享内存竞争：线程0可能在其他warp读完构造循环条件前增加 `steps`。因此数值输出一致不足以完成并行验收。

修复在循环条件之后、更新steps之前加入block barrier，使所有warp先完成条件读取。随后完整2,160组对照、33项Python测试和三种sanitizer全部重跑通过。首轮源码与失败日志保存在项目 `artifacts/gpu/faco-operations-initial-race`；其hash、失败摘要和修复说明一并写入机器可读报告。

## 尚未覆盖

本入口每次分配和同步，是操作级诊断。没有设备roulette、随机流隔离、信息素更新、Hard/Escape执行、wall-clock截止、完整重启、特征、持久Engine与真实GP评价；不得称其为完整无GP求解器或正式GPU性能结果。下一步继续接入这些状态和调度环节，再执行G2完整事务门槛。
