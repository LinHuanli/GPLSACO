# 固定并发面板与截止提交 Engine

日期：2026-09-08。**已实现不含 GP 的固定形状、多实例/seed、wall-clock Engine，完成缓存扣费与重新准备两种模式的工程验收。** 该阶段仍统一使用全tour均匀起点与固定MNE；32动作、参考档案、epoch重启、Hard/Escape、worker与训练尚未接入，E1–E4尚未执行。

机器证据：[batch_engine_results.json](../../../../results/v1/batch_engine_results.json)。前序固定迭代结果保留在 [历史报告](2026-09-08_fixed_faco.md)，其提交和数值不因本次共享初始化细化而改写。

## 实现与边界

`FacoBatchEngine(n, colonies, settings)` 预分配通用形状缓冲；注册实例只接收稳定key与FP64坐标。一次 `evaluate(keys, seeds, budget, mne, mode)` 在C++中运行完整面板并释放GIL。每个colony独立持有参考/GB/epoch、信息素、随机流和ant工作区；随机流不依赖任务到达顺序。批次使用同一个构造/LS内核，每只蚂蚁一个128线程block。

面板先按稳定实例key顺序为所有不同实例建立节点ID顺序的廉价闭环，再准备真实距离有序候选、节点0最近邻解与有上限的CPU checklist 2-opt。昂贵初始化若没有严格改进，保留较早廉价解。这是共同初始化的明确细化，所有共享控制器必须使用相同规则；不称为原生多初始解/3-opt初始化。

两种计时模式：

- `cached_charged`：实际C++评价时间加注册时测得的廉价阶段/昂贵阶段费用。同一固定面板内同实例的多个seed只收一次CPU准备费用；GPU数据复制、reset、缓存建立与搜索按实际时间计入。费用在所对应阶段使用前扣除；超预算的阶段不能提交其初始解。
- `end_to_end`：从已注册原始坐标重新建立廉价解和候选/初始解，不读取缓存的候选或初始tour，不另收缓存费用。NN逐行、贪心逐步检查deadline；CPU初始LS在有限评价上限内完成后再次检查，超时不提交。

计时入口为C++ `evaluate`，此时坐标与任务数组已在host；Python参数整理、磁盘读入、结果封装和通用Engine缓冲分配不在该入口内。注册费用从C++持有坐标开始，不含磁盘IO；正式单实例端到端报告必须明确该边界并另列所需IO。当前费用是同一服务器注册时的测量值，尚未作为G4正式固定费用发布。

GPU每批完成构造/LS、状态归约与信息素更新后同步，下载GB、在CPU验证排列/成本，再取整个面板共同完成时间戳。仅当该时间不晚于截止，才提交host incumbent。迟到批即使产生实际更优解也被丢弃；Python结果不暴露迟到tour/成本。设备状态可能已随该批更新，但返回值只来自host已提交记录，下一次搜索统一reset，因此迟到状态不能变成下一任务的初始解。

这采用保守的整面板完成边界，不等于每只蚂蚁的精确完成时刻。GPU没有内核内wall-clock中断，单批耗时受构造和LS评价上限约束；动作/规模改变后仍须校准提交粒度。`elapsed_seconds = actual_seconds + charged_seconds`，`overrun_seconds`是该有效时间超过B的量。实际返回wall时间超限为 `max(actual_seconds - B, 0)`，摘要另列 `actual_wall_overrun_seconds`；缓存模式不能把逻辑扣费超限描述成真实多运行了同样时长。

## 正确性与状态隔离

| 检查 | 已取得证据 |
|---|---|
| CPU截止账本 | 2项CTest通过，包含原生操作回归；可控时钟检查B处可提交、B后拒绝、零预算、费用和时间回退 |
| 多实例结果对照 | n=17/100/500/1000各4个colony，16项单实例对照；成本与累计构造/LS工作量一致 |
| 重复/乱序/模式 | 重复instance+seed的tour一致；任务顺序反转后正确映射；重新准备与缓存结果一致 |
| 超时准备 | 昂贵准备未完成时仍保留已完成廉价解，不启动GPU搜索 |
| 真实迟到改进 | 固定fixture首批确实改善CPU初始化；加入1秒完成延迟后超过0.5秒预算，返回先前解；下一评价恢复相同结果 |
| 空预算隔离 | 前次成功运行后再给0预算，不读取旧GB，不返回旧incumbent |
| GPU回归 | 4项CTest通过：截止、显式操作、固定迭代、多实例Engine；49项Python测试通过 |
| CUDA工具 | memcheck/synccheck各0 errors；racecheck为0 errors、0 warnings、0 hazards；缩小蚂蚁/迭代数但保留4种规模与16项对照 |

诊断延迟只存在C++专用入口，不暴露为Python生产选项。普通诊断迟到完成时间为1.002857秒，三个sanitizer下为1.049610、1.009690、1.525229秒，均覆盖真实更优解的拒绝。所有这些任务均已正常结束。

## 开发池固定32-colony计时面板

设备为 cuda02 的 RTX A5000，UUID `GPU-056fae3f-b504-efe0-2d9d-b1186860e643`，driver 610.43.02。启动前确认目标卡无计算进程。各规模只使用已冻结development池的16个实例、每实例seed 17/29，共32个colony，每colony 32 ants，MNE=8。预算是整个面板的共同时间，不是单实例独占GPU时延。

| 规模 | 模式 | 预算B/秒 | 截止前完成批次 | 末批丢弃 | 有效预算超限/毫秒 |
|---|---|---|---|---|---|
| 500 | cached_charged | 0.25 / 0.5 / 1 | 2 / 9 / 25 | 各1 | 3.25 / 3.27 / 34.48 |
| 500 | end_to_end | 0.25 / 0.5 / 1 | 3 / 11 / 25 | 各1 | 9.62 / 25.03 / 30.66 |
| 1000 | cached_charged | 0.5 / 1 / 2 | 2 / 9 / 23 | 各1 | 22.78 / 6.39 / 25.69 |
| 1000 | end_to_end | 0.5 / 1 / 2 | 2 / 9 / 23 | 各1 | 25.11 / 24.27 / 14.51 |

另有每种规模/模式各一次B=0.002秒，共4个准备截止面板，均保留32条廉价可行解且未启动GPU搜索。16个面板共512个任务结果全部为截止前合法闭环，独立Python距离重算最大绝对误差 `9.094947017729282e-13`。未删除或替换短预算结果。

完整准备的缓存费用分别为500面板0.136710秒、1K面板0.306691秒；重复seed未重复收费。通用设备数组分别占32,488,448和64,552,448字节，不含CUDA context/runtime或未来档案/特征缓冲，不能作为完整方法的峰值显存结论。末批丢弃占已启动批次的比例为3.85%–33.33%，短预算端的浪费需要在正式预算/动作接入后继续校准。

两种准备模式在较长预算下完成相同批数，在500的较短预算下略有不同；该面板只能核验当前计时协议可执行，不能证明两种计时方式统计等价。reference gap仅在求解返回后由外部evaluator读取开发标签计算，逐面板保留在JSON用于诊断。没有读取正式测试性能，也没有把这些预算选为正式E1–E4配置。

## 复现与后续

在已有CPU/CUDA构建和已发布索引上：

```bash
ctest --test-dir build/cpu --output-on-failure
bash scripts/run_batch_engine_checks.sh GPU-056fae3f-b504-efe0-2d9d-b1186860e643
CUDA_VISIBLE_DEVICES=GPU-056fae3f-b504-efe0-2d9d-b1186860e643 \
  .venv/bin/python scripts/batch_engine_smoke.py \
  --gpu-uuid GPU-056fae3f-b504-efe0-2d9d-b1186860e643 \
  --output artifacts/gpu/batch-engine/smoke-reproduction
.venv/bin/python scripts/summarize_batch_engine.py
```

检查脚本必须在有空闲目标卡的服务器执行；项目根目录、TMPDIR和CUDA_CACHE_PATH约定同前序报告。归集脚本默认读取本次原始目录，复现实验使用新输出目录避免覆盖；重算公开摘要前应明确所采用的证据批次。

下一项是Hard/Escape操作合法性、区域/特征、参考档案和完整epoch重启事务，然后将GP动作评分接入同一Engine。G2仍未整体通过；worker、真实DEAP求解评价、正式成本校准与G4冻结继续保持pending。
