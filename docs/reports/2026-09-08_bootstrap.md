# 2026-09-08 研究启动报告

本报告只描述准备、正确性检查和两个开发实例的短程原生试跑。尚未实现完整CUDA FACO Engine，未执行正式GP训练或E1–E4，不提供加速比或算法优越性结论。机器可读证据见 [bootstrap_results.json](bootstrap_results.json)。

## 1. 数据实查

101个实例文本文件，完整流式读取 **15,960,485,947字节**；包括重复文件共2,829,681条记录。逐文件计算完整SHA-256和非空行数，首条记录101条均能解析且标签tour合法。其余记录未因此获得逐实例认证。细目见 [data_inventory.json](../../provenance/data_inventory.json)。

| 数据 | training文件中记录 | validation文件中记录 | test文件中记录 |
|---|---:|---:|---:|
| TSP100 uniform | 1,280,000 | 1,280 | 1,280（排除逐字节copy） |
| TSP500 uniform | 64,000 | 128 | 128 |
| TSP1K uniform | 64,000 | 32 | 128 |
| TSP10K | 6,400 | 16 | 16 |
| TSP500 cluster / Gaussian | 无对应训练池 | 无 | 各128 |
| TSPLIB衍生文本 | 无 | 无 | 49个，n=51..1002 |

另有TSP50/200已完整登记，但不纳入本轮500/1K主训练。10K路径中的train/val名称不改变其在v4中的迁移用途。验证建议200–500/规模不能由现有val文件直接满足；需按parent groups从训练池保留独立组。10K默认test仅16个，建议100个需从未参与开发的6400条池预先选择并去重，不能声称已有100个测试实例通过审计。

一对重复文件为 `tsp100_concorde_7.756 copy.txt` 和原文件。TSPLIB `.txt` 缺原距离类型且坐标已归一化，尚不能用于官方目标gap。合成标签是给定tour的连续欧氏成本；用户声明最优与独立最优性证明分别记录，目前不由文件名推断certificate。

## 2. 环境与来源

项目已建立独立venv。实际依赖包括DEAP1.4.4、NumPy2.5.3、pybind11 3.1.0，完整依赖锁见 [python.lock.txt](../../provenance/python.lock.txt)。CPU/CUDA均采用C++17、GCC15和CUDA13.3，CUDA目标SM86；精确版本/driver/UUID和选项见 [environment.json](../../provenance/environment.json)。

FACO关键blob与v4锚点一致；本地源码快照没有.git，所以全部参与构建文件另存SHA-256，不声称已证明整个目录对应commit。外部文件保持只读。原生GCC15编译需要命令行补入cstdint头；不改变算法源码，区别于上游默认Ofast的项目Release构建选项已记录。

## 3. 程序评分链路

已实现DEAP grammar→后缀IR→严格验证→C++/CUDA评分；JSON常数保存FP32位模式。CPU与CUDA共享函数语义，CUDA一warp对同树32动作评分，特征按 `[12,colony,32]` 排布。绑定先验参与复制、释放GIL运行、恢复GIL返回；当前是诊断接口，不是完整Engine。

- CPU测试27项通过；目标A5000上的CPU＋CUDA测试29项通过。
- 固定seed下263个程序，每个7个colony，共58,912个动作分数；包括NoFeedback、常数树、单特征、SUB/AQ、裁剪、最大63节点/深5树。
- Python与C++本批最大绝对分差0；CUDA最大绝对分差 **4.76837158203125e-7**；动作选择不一致数0。
- compute-sanitizer memcheck和synccheck各39程序、8,736分数，均报告0错误。

这些是指定样本和容差内的数值证据，不保证所有可能树在所有近似平局上逐位一致，不证明完整TSP求解器正确或高效。全树/参数非法输入、NaN、错误dtype、非连续数组、空mask、超深链和消融ID均有检查。

## 4. 原生初始化反例与适配核验

原始mfaco在500开发样本首条、seed17、32ants、50iterations、单线程配置下120秒未完成初始化。观察副本在未改变接受判断的前提下捕获：

```text
same_edges=true
before=0.14583575006734209
after=0.14583575006734206
false_gain=2.7755575615628914e-17
two_opt_changes=0 three_opt_changes=0
```

原因是初始化3-opt对同一组三条无向边的浮点加法重排产生伪严格改进；原函数没有迭代上限。此反例证明有错误接受，不把一次观察外推为所有距离/实例都受影响。诊断程序和失败原日志留在artifacts中。

新增 **FACO-Native-EdgeGuard**：保留原生版本，在项目内隔离快照中只拒绝初始化3-opt前后完全相同的无向边多重集合。完整diff/原新hash已版本化，未新增gain阈值，也未改主构造和checklist LS。

该适配版在500/1K各一个训练开发实例、mfaco/faco_apt各三个seed17/29/43完成12次运行，全部合法。Python独立原目标重算与原生增量成本最大绝对差 **1.1084466677857563e-12**。每个规模/算法的seed17另与适配版CLI比较，4次成本一致。

以上每次50iterations、32ants、单线程，是工程核验。两个开发点集及六个重复不能支持实例级推断，不报告方法胜负。原生初始化数量实际受CPU可见核数影响，本机为8条起始tour；正式底座需要显式冻结该项。

## 5. 后续准备与下一步

现有GPU、编译器和数据足够继续开发，不需要用户补环境。仍需完成：选定500/1K面板的全量逐记录/亲缘核验和划分；原TSPLIB距离恢复；源操作oracle；无GP CUDA FACO/LS；完整重启与deadline；Engine、训练闭环与预算pilot。只有这些门槛通过后，才能运行可解释的E1。

复现入口见 [reproduce.md](reproduce.md)。
