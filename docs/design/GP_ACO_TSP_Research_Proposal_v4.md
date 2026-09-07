# 局部搜索驱动的结构控制：利用遗传编程设计大规模 TSP 蚁群算法

## Learning Structural Search Control for Large-Scale TSP with Genetic Programming

**版本：** v4.0 · 2026 年 9 月 8 日 · Python/DEAP 与 C++/CUDA 实现分层版  
**文档性质：** 研究方案；所有假设、算法参数与实验配置均为待验证设计。  
**研究范围：** 对称 TSP，以现有 TSP100、500、1K、10K 合成数据和带最优标签的 TSPLIB 数据为基础。  
**方法暂名：** GP-FACO，表示 Genetic-Programming-Controlled Focused Ant Colony Optimization。基础语义来自作者公开的节点重定位 FACO 实现；学习的是固定扩展接口上的控制规则，而不是从头生成整套 ACO。
**核查范围：** 沿用 v3 的 FACO、GPU-MMAS 与 cuOpt 源码审查；本轮核对 DEAP 树与算子、pybind11 边界和 CUDA 进程使用的官方文档。尚未安装 DEAP/pybind11、构建求解器或进行 GPU 性能测试。第 10 节保留来源依据，第 11 节给出实现分层与接口。

> **研究主线：** 学习何时保留已有解结构，何时进行局部改变，以及何时切换搜索中心。全文围绕四个科学问题组织四组主实验。GPU/CUDA 是实现与实验支撑，不单独构成 research question。

---

### 本轮修订范围

保留 v3 的四个科学 RQ、四组主实验和 FACO 源码语义。明确使用 Python＋DEAP 管理离线 GP，C++17＋CUDA 完成在线求解和程序评分，pybind11 只提供粗粒度评价接口。增加稳定程序格式、terminal 编号、常驻 GPU worker、全部个体重评和 checkpoint 规则。第一版不增加多保真、代理模型、PyTorch 依赖或 GPU 上的遗传算子。

## 摘要

大规模 TSP 的混合元启发式需要协调两种相互制约的需求：一方面复用已经获得的高质量结构，另一方面打破限制后续改进的结构。局部搜索能够快速改进给定候选解，但其后续效果取决于候选起点、结构变化和搜索记忆。因此，本研究将 ACO 的自动设计对象定义为局部搜索之上的结构控制规则，而不是任意重写整个求解器。

研究拟回答四个问题：局部搜索后的反馈是否支持有价值的状态相关控制；局部扰动与搜索中心重启是互补还是替代；高质量稀疏候选先验能否替代探索出口；学到的规则能否跨问题规模和实例结构迁移。方法以节点重定位 FACO 为基础，仅演化一棵实值 GP 树，对“重启模式—扰动起始区域—MNE 阈值”构成的有限动作集合评分。函数集采用有界、逐元素数值运算，终端集采用十二个低开销状态与动作特征，fitness 为固定运行预算下的平均最优差距。外层使用标准代际 GP，不引入多保真评价、代理模型或强化学习训练器。

CUDA 实现将同一棵树的不同动作评分映射到一个 warp，将独立蚂蚁的扰动和局部搜索映射到线程块，并在实例与随机重复之间进行批量并行。四组实验分别对应四个研究问题，使用统一求解底座、最优标签隔离、等运行预算和实例级统计推断，区分性能改善与可支持的机制结论。

# 1. Introduction

## 1.1 从求解组件转向搜索过程

旅行商问题的高质量启发式通常依赖候选结构、随机搜索和局部改进之间的配合。对于 ACO 与局部搜索的组合，重要的不只是单次生成的 tour 是否较短，还包括它经过局部搜索之后会到达什么结果、消耗多少计算，以及对未来搜索形成怎样的引导。

MMAS 强调对优质历史解的利用并维护信息素边界；FACO 则进一步限制相对于参考解的结构变化，并围绕变化位置组织局部搜索。后者说明，重新探索整条 tour 并不是每次搜索尝试的必要条件，已有结构可以被大量复用。[R1]

由此，本研究将核心矛盾表述为：**保留已有结构能够节约搜索与修复成本，但过度保留也可能导致重复；扩大结构变化可能创造新的搜索机会，也可能使局部搜索反复重建已经获得的好结构。** 如何在不同搜索状态下处理这一矛盾，是一个独立于具体学习技术的科学问题。

## 1.2 自动设计的对象

我们不以“GP 能替代某一人工公式”为主要目标，而关注能够直接改变完整搜索过程的决策：扰动发生在哪里、改变多少结构，以及是否继续围绕当前参考解及其信息素记忆搜索。

这些决策具有潜在的状态依赖性。同样的改动程度，在搜索仍持续改善时与反复返回相似终态时，可能具有不同价值；同样的重启，在存在高质量替代参考解时与没有可靠替代结构时，也可能产生不同结果。这些关系不能通过最终 tour length 的单一比较直接解释，需要围绕明确的研究问题设计控制实验。

## 1.3 与已有研究的区别

FACO 已有自适应选择扰动幅度的研究，因此“动态调整一个扰动参数”不是本文足够的创新点。[R2] 动态神经 ACO 引导也已有工作：DyNACO 根据搜索状态更新边级引导信号并使用完整搜索轨迹训练。[R3] 本文不宣称首次进行状态感知 ACO、首次结合学习与局部搜索，或首次采用稀疏候选图。

本文拟研究的差异在于：**使用紧凑符号程序，联合选择局部结构变化与搜索中心重启，并通过控制实验解释这种联合决策是否有价值。** 学习器输出的是低维结构动作，不是一个新的全图边权矩阵。GP 的角色是发现规则；真正需要验证的是规则所利用的信息、不同结构动作之间的关系以及规则的适用范围。

# 2. Motivation：Why → How → What

## 2.1 Why：需要优化的是有价值的结构变化，而不是变化数量

设当前参考解为 $r_t$，扰动后的候选为 $y_t$，局部搜索后的结果为 $z_t$：

$$
r_t\xrightarrow{\text{结构动作 }a_t}y_t
\xrightarrow{\text{固定局部搜索 }L}z_t.
\tag{1}
$$

扰动的作用不应只由 $C(y_t)$、改变边数或最终保留了多少扰动边衡量。某些暂时引入的边可能在局部搜索中被删除，但仍然改变搜索轨迹；另一些改动虽然产生不同终态，却可能只增加修复成本。因此，研究目标是给定预算下的整体改进，而不是最大化多样性或 shuffle 次数。

这里不把所有 checklist 终态称为完整邻域下的局部最优。本文统一使用“LS 终态”，除非实际实现已经穷尽声明的完整邻域。

## 2.2 How：将结构控制组织为两个相关尺度

**局部尺度：** 保留当前搜索中心，选择从哪里开始重新处理已有结构及 MNE 改动程度。

**搜索中心尺度：** 保留历史最优结果，但切换活动参考解并重置信息素，使后续搜索不必持续围绕同一结构展开。

这两类动作可能互补，也可能替代。较准确的局部扰动可能减少重启需求；适当重启也可能使较小的局部改动重新变得有效。本文不预设答案，而将其作为直接研究对象。

## 2.3 What：学习一个评分程序，而不是搭建复杂控制体系

给定状态 $s_t$ 和有限动作集合 $\mathcal A_t$，只学习一棵 GP 树 $f$：

$$
a_t=\arg\max_{a\in\mathcal A_t}f\bigl(\phi(s_t,a)\bigr).
\tag{2}
$$

函数集定义可组合的数值运算；终端集定义程序可见的信息；fitness 评价完整运行结果。基础状态转移、常规信息素更新、档案维护和局部搜索保持固定。

该方法选择不需要多保真、代理模型、多个独立策略网络或额外价值函数。复杂性主要放在高效执行与公平评价上，而不是继续增加学习器层次。

# 3. Research Questions：四个科学问题

## RQ1：状态依赖——局部搜索反馈是否提供了可利用的结构控制信息？

> **在当前解结构和搜索时钟之外，局部搜索后的进展、重复与修复反馈，能否帮助判断何时应保留结构、扩大扰动或重启，从而提高有限预算内的搜索质量？**

研究的不是“有状态程序在表达上是否更复杂”，而是新增反馈是否具有样本外决策价值。若仅根据当前结构与预设时间计划就能达到同样结果，则没有充分证据支持使用这些历史反馈。

**H1：** 使用 LS 后反馈的控制程序优于删除这些反馈后重新训练的程序，并且动作的相对价值会随相关状态发生变化。

**对应主实验：E1。**

## RQ2：协同关系——局部扰动与搜索中心重启是互补还是替代？

> **学习“改哪里、改多少”与学习“何时换一个搜索中心”分别能带来什么收益？联合学习是否产生超过单独学习的收益，还是其中一种机制已足以替代另一种？**

这是关于两个结构尺度的关系，不是把每一个代码模块都设置成 RQ。第一版将参考解切换与信息素重置定义为一个固定重启动作，因此不单独宣称识别两者各自的必要性。

**H2：** 两类学习控制均具有贡献；在指定损失尺度和预算条件下，联合控制可能具有正交互。是否存在交互必须通过析因对照检验，不能仅因为完整方法成绩最好就宣称协同。

**对应主实验：E2。**

## RQ3：结构约束——高质量稀疏先验能否替代探索出口？

> **当 alpha-nearness 或 POPMUSIC 已提供较好的候选边时，把扰动与局部搜索永久限制在同一图内是否足够？还是仍需要有限的图外探索，才能发挥结构控制的价值？**

本问题区分“候选边更有希望”与“搜索仍具有足够自由”。它不把候选图构建速度单独作为科学问题，也不将最优边覆盖率直接等同于搜索有效性。

**H3：** 高质量先验与有限探索出口并非必然相互替代；是否保留出口会改变结构控制的作用及收益。

**对应主实验：E3。**

## RQ4：规律的可迁移性——程序学到的是结构原则还是训练规模的经验？

> **在中等规模实例上学到的、依据局部与归一化反馈进行决策的规则，能否迁移到 TSP10K 与 TSPLIB，而不依赖重新选择阈值或重新训练？**

这里的核心是经验规律的适用范围，而不是 GPU 的扩展效率。规模迁移与实例结构迁移分别报告；不能把同一生成器的新随机种子当成跨分布证据。

**H4：** 冻结程序在未参与训练和选择的大规模及外部实例上仍保留相对于统一底座的收益，但不预设所有距离类型和实例结构均有效。

**对应主实验：E4。**

## 3.1 一一对应关系

| 科学问题 | 主实验 | 实验的核心对比 | 主要结论形式 |
|---|---|---|---|
| RQ1：反馈是否有决策价值？ | E1：状态依赖与整体有效性 | 完整反馈 vs 去反馈重训；固定/预设策略 | 是否存在样本外反馈收益及动作状态依赖 |
| RQ2：两个结构尺度如何相互作用？ | E2：$2\times2$ 析因实验 | 是否学习局部扰动 × 是否学习重启 | 独立贡献、替代关系与交互 |
| RQ3：结构先验是否足以保证有效探索？ | E3：候选先验与探索出口 | alpha/POPMUSIC × 硬图/有限出口 | 先验质量与探索自由的关系 |
| RQ4：规则是否具有可迁移性？ | E4：冻结程序迁移 | 训练规模内 vs 10K/TSPLIB | 规模与实例结构上的适用边界 |

正确性、计时和资源统计作为所有实验的共同前提，不另编号为独立主实验。

# 4. Method：一棵 GP 树、一个动作空间、一个 fitness

## 4.1 问题与目标

对称 TSP 实例记为 $I=(V,d)$，$|V|=n$。可行 tour $x=(v_1,\ldots,v_n)$ 的长度为：

$$
C_I(x)=\sum_{i=1}^{n}d(v_i,v_{i+1}),\qquad v_{n+1}=v_1.
\tag{3}
$$

令 $C_I^*>0$ 为项目提供并经距离定义核对的最优值。预算 $B$ 下的历史最优解为 $x_f^{\mathrm{best}}(I,\xi,B)$，评价差距为：

$$
g(f;I,\xi,B)=
\frac{C_I(x_f^{\mathrm{best}}(I,\xi,B))-C_I^*}{C_I^*}.
\tag{4}
$$

在线求解器不读取 $C_I^*$ 或最优 tour。标签由外部 evaluator 用于训练 fitness 和测试评分。

## 4.2 基础算法：节点重定位 FACO，而非重新定义一套 ACO

### 源码锚点与三层实现

算法语义以作者仓库 `RSkinderowicz/ACO-TSP-Adaptive-Tuning` 的提交 `a904e6a8786d48593ef1cac975edef5ed8920af3` 为锚点。具体选择 `src/faco.cpp` 中的 **`run_mfaco`**，而不是把整个仓库默认运行的自适应算法都纳入 GP 内层。该文件同时包含 `Route::relocate_node`、`Route::two_opt_nn` 和 `run_faco_apt`，对应节点重定位、实际调用的 checklist LS 以及 bandit 自适应版本。[S1, S2]

| 层次 | 本文名称 | 固定内容与变化 | 用途 |
|---|---|---|---|
| 原生实现 | FACO-Native | 保持作者 `run_mfaco` 的源码与显式运行配置；不添加 GP 或新重启 | CPU 语义参照、已有算法对照 |
| 共同扩展 | FACO-Control | 移植相同转移、MNE 与 checklist 规则；明确增加起始区域、档案和 epoch 重启接口 | Static、Rule、GP-NoFeedback 与 GP-Full 的共同底座 |
| 学习方法 | GP-FACO | 只用 GP 替换上述接口的决策规则 | 本文方法 |

CUDA 版本是我们的实现，不称为作者已经提供的 GPU-FACO。移植后的共同底座先接受原生 CPU 对照的操作级与分布级校验；不能只因名称都含 FACO 就默认两者等价。

### 固定候选生成内核

从完整参考 tour 复制工作解，在一个起始节点处开始，按原生 `select_next_node` 的信息素×距离启发式选择下一未访问节点，将选中节点重定位到当前节点之后，再从该选中节点继续。这不是“从区域中任意选一个节点，再任意寻找插入边”的新算子。[S2]

原生候选机制依次使用主候选表、备用表和必要时的未访问节点回退，因而**原生 FACO 并不将所有 tour 边硬限制在候选图中**。主比较先保留这一语义；E3 的 Hard/Escape 是单独声明、对所有控制器共同实施的候选限制，不冒称原生行为。[S2]

### 参考解、档案与 epoch：明确声明的固定扩展

维护历史最优解 $x^{\mathrm{gb}}$、当前活动参考解 $r$、本 epoch 最优解 $x^{\mathrm{eb}}$ 和最多四条解的有限档案 $\mathcal H$。档案始终保留历史最优解，其余解需在相对历史最优长度的固定质量范围内；建议起始范围为 2%，开发阶段确定后冻结。

替代参考解 $r^{\mathrm{alt}}$ 由固定规则从质量合格且结构不同的档案解中选出；不学习档案容量或排序器。差异可由固定 64 个节点估计，但去重与“确实不同”的判定须有完整边集或可靠指纹核验，不能因样本没有检出差异就断言两个 tour 完全相同。

原生 `run_mfaco` 在 iteration-best 与 global-best 间按固定概率选择强化 tour，并将同一 tour 作为下一次参考解。共同扩展保留这种**随机混合调度形式**，但把其中历史 global-best 的角色替换为 epoch-best；历史 global-best 仅负责保存结果、质量比较和档案。该替换是重启扩展，不是原生规则。[S2]

记当前批次最优为 $x^{\mathrm{ib}}$，固定概率 $p_{\mathrm{best}}^{\mathrm{source}}$ 下选择 $z_t^{\mathrm{dep}}=x^{\mathrm{eb}}$，否则选择 $x^{\mathrm{ib}}$；强化后令 $r_{t+1}=z_t^{\mathrm{dep}}$。该概率在核查的参数结构中默认 0.01，但最终实验显式配置并统一冻结，不把某个默认值称为最优设置。[S3]

为避免旧 epoch 的个体解在重启后悄悄重新进入当前搜索，共同底座明确设置 `keep_better_ant_sol=false`，每批次使用本批实际结果；并设置 `source_sol_local_update=false`，禁止批次内异步替换共享参考解。前者不同于核查代码中的默认个体精英保留，后者与 `run_faco_apt` 的可选即时更新不同。原生复现保留其原配置；所有共同底座控制器使用同一显式设置。它们是固定适配，不计入 GP 的贡献。[S2, S3]

### 信息素更新：保留原生计数和默认值语义

统一用 $\lambda$ 表示**保留比例**，避免与其他论文中用 $\rho$ 表示蒸发比例混淆。核查代码中的 `rho_` 实际上是保留比例。对已存储候选边，常规更新为：

$$
\begin{aligned}
\bar\tau_{e,t}&=\max(\tau_{\min},\lambda\tau_{e,t}),\\
\tau_{e,t+1}&=
\begin{cases}
\min\!\left(\tau_{\max},\bar\tau_{e,t}+1/C(z_t^{\mathrm{dep}})\right),
 &e\in E(z_t^{\mathrm{dep}}),\\
\bar\tau_{e,t},&\text{otherwise}.
\end{cases}
\end{aligned}
\tag{5}
$$

信息素上下界沿用开启 LS 时的 `calc_trail_limits_cl` 形式；共同扩展用本 epoch 最优成本确定边界，并在重启时重新初始化。未存储边使用独立默认值 $\tau_{\mathrm{default}}$；它随蒸发衰减并受下界约束，但不逐边获得精英强化。不能把未存储边隐式设为零，也不能把双向候选条目机械合并后仍声称与原生有向行存储完全相同。[S2, S4]

GP 不修改这些更新公式、边界计算或主转移概率。候选条目的对称性处理、随机调度和浮点运算顺序均进入配置与校验记录。

## 4.3 动作空间：重启模式、起始区域、MNE 阈值

定义动作：

$$
a=(b,j,k),\qquad
b\in\{0,1\},\quad j\in\{1,2,3,4\},\quad
k\in\{2,4,8,16\}.
\tag{6}
$$

最多 $2\times4\times4=32$ 个候选动作。动作的数目和 GP 的单树评分结构不变，但各分量与源码接口精确对应。

### 重启模式 $b$

$b=0$ 保持当前活动参考解与信息素；$b=1$ 执行固定的 colony 级事务：切换到 $r^{\mathrm{alt}}$、设定 epoch-best、重新计算上下界、重置**已存储信息素和图外默认信息素**、失效并重建 $\tau\eta$ 缓存、清理本 epoch 的反馈与残留 ant 工作状态。历史最优解与有限档案保留。

核查到 `CandListPheromone::set_all_trails()` 只修改已存储条目，并不修改 `default_pheromone_value_`。因此不能直接调用该函数就声称完成上述重启。这是将旧类用于新增重启时的接口风险，不是断言原生初始化本身存在算法错误。[S4]

没有合法替代参考解时屏蔽 $b=1$，不隐藏地调用额外求解器。GP 只学习何时触发这一固定动作，不学习重置强度或多个重启子策略。

### 起始区域 $j$

每个候选参考解产生四个等大小的**起始节点候选集合**，建议每组 16 个节点：两个 tour 连续片段、一个候选图局部邻域、一个分散节点集合。生成器、补齐和去重规则固定，不读取最优标签。每只蚂蚁从所选集合独立抽取起始节点，替代原生的全 tour 均匀起点。

此后完整沿用原生信息素引导的选点、访问标记及节点重定位顺序。**区域控制的是扰动起始位置，不硬限制所有后续改动节点。** 论文和图表采用“起始区域选择”，不把它误写成精确定位整个变化子图。区域编号和城市编号不进入 GP 终端。

### 程度 $k$：原生 MNE，而不是重定位次数

令 $m_{\mathrm{new}}$ 为构造阶段累计的新连接计数，初始化为 0。每次选择 `sel` 并执行 `relocate_node(curr, sel)` 后，只在参考 tour 中不存在无向边 $\{curr,sel\}$ 时令 $m_{\mathrm{new}}\leftarrow m_{\mathrm{new}}+1$。其余因重定位产生的重连边不在该计数中；后续变化也可能撤销早先的改动。停止条件为 $m_{\mathrm{new}}\ge k$ 或访问完节点，而不是“成功重定位 $k$ 次”。[S2]

因此必须分别记录：目标 MNE、累计计数、构造步数、实际非恒等重定位次数、LS 前最终新边数、LS 后结构差异。不能沿用 v2 中将最终改边数绑定为 $3k$ 的说法；$3$ 倍非恒等重定位次数才给出相应的宽松上界，而 MNE 不等于这个次数。

本轮移除 v2 的 `10k` 尝试上限，保留原生有限访问步数及统一 wall-clock 截止。在 E3 Hard 模式下，若声明的候选与回退枚举中不存在符合全新增边约束的合法移动，则终止本次构造并返回当前可行 tour，不无限重试，也不悄悄回退到图外边。

### 一次控制作用于一个蚁群批次

一个批次只选一次 $(b,j,k)$。蚂蚁共享参考解、起始区域和 MNE，但独立抽起点与转移随机数；批次内信息素和源 tour 只读，结果统一归约。该 colony 级选择不同于原生 adaptive FACO 的逐蚂蚁 bandit 决策频率，作为固定控制接口明确披露。[S2]

不能让各只蚂蚁独立重置同一份全局信息素。静态、规则与学习控制使用相同批次语义，不把同步方式变化当成 GP 效果。

## 4.4 Function set：七个数值函数

GP 使用单一实数类型，不引入树内循环、图查询、排序、局部搜索调用或动态内存分配。

$$
\mathcal F=
\{\mathrm{ADD},\mathrm{SUB},\mathrm{MUL},
\mathrm{MIN},\mathrm{MAX},\mathrm{ABS},\mathrm{AQ}\}.
\tag{7}
$$

| 函数 | 元数 | 数学定义 | CUDA 执行考虑 |
|---|---:|---|---|
| ADD | 2 | $x+y$ | 逐元素浮点加法 |
| SUB | 2 | $x-y$ | 逐元素浮点减法 |
| MUL | 2 | $xy$ | 用于状态与动作特征的交互 |
| MIN | 2 | $\min(x,y)$ | 形成分段规则，无需树内控制流 |
| MAX | 2 | $\max(x,y)$ | 形成阈值、门控与截断 |
| ABS | 1 | $\lvert x\rvert$ | 单目数值运算 |
| AQ | 2 | $x/\sqrt{1+y^2}$ | 对应 `x * rsqrtf(1 + y*y)`，无零分母 |

AQ 是本方案选择的平滑缩放算子，不将其名称或使用声称为算法创新。使用 AQ 而不是任意除法、幂函数和指数函数，是为了控制数值范围与执行路径。第一版不加入显式 `if`；MIN/MAX 与乘法已能表达分段和条件交互。

所有终端有限，每个中间结果裁剪到 $[-8,8]$，以形成一致的数值语义。此裁剪是函数执行定义的一部分，在训练、验证和部署中保持不变。GP 特征与评分可用 FP32；tour 长度、合法性和 move gain 的计算采用另行验证的目标精度，不能因为评分使用快速数学就改变 TSP 的距离定义。

## 4.5 Terminal set：十二个有计算边界的特征

定义：

$$
\mathcal T=\{u,s,q,c,b,\kappa,g_r,d_r,e_R,v_R,p_R,\ell_R\}
\cup\mathrm{ERC}.
\tag{8}
$$

ERC 为 $[-2,2]$ 内的常数，并显式允许 $-1,0,1$。所有特征都能在动作执行前获得，不能包含该候选动作尚未运行的 LS 结果。

| 终端 | 含义与建议定义 | CUDA 计算位置 |
|---|---|---|
| $u$ | 已用预算比例 $t/B\in[0,1]$ | 每个蚁群一个标量 |
| $s$ | global-best 未改进的连续批次数，$\min(\Delta_t/32,1)$ | 每批次更新一个计数器 |
| $q$ | 近期候选经 LS 后返回参考 tour 的指纹匹配比例 | 每批次归约；保留有界 EMA |
| $c$ | 近期每只蚂蚁的 LS move-evaluation 数相对固定调用上限的均值 | 已有 LS 计数器归约 |
| $b$ | 是否切换参考解并重启，取 0 或 1 | 候选动作描述，不读取数组 |
| $\kappa$ | 预计算的 $\log_2(k)/4$，$k$ 为原生 MNE 阈值 | 四个常量，不在线调用 log |
| $g_r$ | 候选参考解相对 global-best 的长度差，除以档案质量带并裁剪到 $[0,1]$ | 已维护的 tour 长度 |
| $d_r$ | 候选参考解与当前参考解的采样边差异比例 | 固定 64 个节点的稀疏读取；保持模式为 0 |
| $e_R$ | 区域内当前边长相对静态局部距离尺度的异常程度，统一裁剪归一化 | 16 节点归约，静态尺度预计算 |
| $v_R$ | 区域当前边在四个档案解中的不一致程度 | 有界 $16\times4$ 次邻接检查 |
| $p_R$ | 区域当前边的平均归一化信息素强度 | 稀疏读取与 16 节点归约 |
| $\ell_R$ | 区域在候选参考 tour 上的分散程度：区域节点向集合外的相邻连接数除以 $2\lvert R\rvert$ | 16 节点邻接检查 |

**统一约定。** $q,c$ 使用固定 EMA 系数，例如 $1/16$。重启后重置这两个 epoch 相关统计，$s$ 仍对应全局改进历史。长度差使用在线 global-best，而不是最优标签。$d_r$ 是采样估计，不写成精确全 tour 距离；$q$ 使用指纹时也应明确其近似性质，离线诊断可核对实际 tour。

$e_R$ 的局部尺度由当前实例的固定近邻距离摘要给出，尺度接近零时使用冻结的保护常数。$p_R$ 使用执行动作之前的信息素；对于 $b=1$，它仍是旧记忆的观测，而不是重置后的未来值。所有缺失或不可定义特征使用显式一致的默认规则。

### 终端设计的约束

不将最优 gap、最优边、未来改进、任意全图熵计算或一次完整 LS 的返回值封装成看似廉价的 terminal。每个特征的准备成本必须计入求解时间。全局与区域特征先计算一次，不能让树中每次引用都重新扫描 tour。

由于动作通过 argmax 比较，纯粹加在所有动作上的同一个状态项会相消。在没有触发输出裁剪或改变数值平局时，例如 $f(a)=s+h(a)$ 与 $h(a)$ 选择相同动作；若进入裁剪饱和区，不能无条件使用这一等价性。有效程序需要通过 $s\kappa$、$qb$ 等交互改变动作排序；MUL、MIN/MAX 和 AQ 为此提供表达能力。

## 4.6 一个示意程序

例如：

$$
f(s_t,a)=
\operatorname{MUL}(q,b)
+\operatorname{MUL}(e_R,\kappa)
-\operatorname{MUL}(c,\kappa)
-\operatorname{MUL}(b,g_r).
\tag{9}
$$

式 (9) 只展示表达能力：重复反馈可与重启相互作用，区域边长异常可与改动程度相互作用，修复成本和替代参考质量可以形成抑制。它不是提前指定的正确规则，也不能被当作进化已经发现的结果。

## 4.7 Fitness：只用完整运行后的平均最优差距

主训练使用 TSP500 和 TSP1K。每代在两个规模各抽取一个相同大小的训练 mini-batch，所有个体使用相同实例、种子和固定预算，完成同一精度的整段求解。

$$
J(f)=\frac{1}{2}\sum_{n\in\{500,1000\}}
\frac{1}{|D_n|S}
\sum_{I\in D_n}\sum_{\xi=1}^{S}
 g(f;I,\xi,B_n).
\tag{10}
$$

**最小化 $J(f)$。** 不增加 diversity reward、单步收益项、复杂度加权项或额外多目标排序。程序规模使用硬限制；fitness 相同或处于冻结的数值平局容差内时优先较短程序。

Anytime 曲线作为结果分析，不混入第一版 fitness。较慢的决策、更多修复和频繁重置会减少固定预算内可完成的搜索量，因此通过完整运行自然影响式 (10)。GPU 批量评价的预算语义见第 5.5 节；不能把多个程序争用 GPU 时的任意完成时间当作单实例求解时间。

## 4.8 标准代际 GP

建议初始配置为：种群 128、50 代、最大深度 5（根深度为 0）、最多 63 个节点、锦标赛大小 4、精英 4 个、子树交叉概率 0.8、变异概率 0.2。第一版使用 DEAP 的子树变异，新 ERC 随新子树生成；不增加额外常数优化器。交叉与变异采用 `varAnd` 的先交叉、再独立变异语义。超限后代用 `gp.staticLimit` 按父代回退，并在导出阶段再次检查，不执行不明确的修剪。

每代重新评价包括精英在内的全部个体，使用该代共同 mini-batch；不把上一代不同数据上的 fitness 与当前代直接比较。训练池足够大时轮换 mini-batch，但评价预算、终端、LS 和 fidelity 不改变。

最终程序仅由独立验证集选择。主实验使用至少五个独立 GP 演化重复，并报告训练过程的稳定性。**不进行多保真淘汰、不训练 surrogate、不动态扩大 grammar，也不嵌套额外的策略优化器。**

# 5. CUDA 实现：与 GP 表示共同设计，但不作为 RQ

本节定义实施方案，不报告尚未测得的加速比。CUDA 的 warp 执行与内存访问机制说明，应优先让同一 warp 执行相同指令，并使相邻线程访问相邻数据。[R4, R5]

## 5.1 并行层级

| 层级 | 并行对象 | 建议映射 | 不能混淆的事项 |
|---|---|---|---|
| 多 GPU / 集群 | 不同 GP 个体、演化重复或实验配置 | 每个 GPU worker 在一次计时任务内评价一个程序 | 不让竞争 GPU 资源改变个体 fitness 的预算语义 |
| 单 GPU 批量 | 同规模的不同实例与求解种子 | 多个相互独立的 colony | 静态实例数据可共享，信息素与运行历史不可串用 |
| Colony 内部 | 多只蚂蚁 | 一个 thread block 负责一只蚂蚁的 tour | 同批次只读参考解和信息素 |
| 程序评分 | 同一个 colony 的 32 个动作 | 一个 warp，一条 lane 对应一个动作 | 同一 warp 不混入不同 GP 树 |
| 单只蚂蚁内部 | 候选 move 检查、tour 数据移动 | block 内并行检查和归约 | 接受 move 后统一提交，避免冲突修改 |

32 个动作不是新的理论假设，而是有限控制空间与 GPU 执行形状的一个方便匹配。若以后需要更大动作集，可以分成多个 warp 评分再归约，但不应为了填满线程而加入没有研究意义的动作。

## 5.2 GP 树的表示与求值

Python＋DEAP 在 CPU 上负责 GP 的选择、交叉和变异；自有导出器检查表达式并转换为固定上限的后缀指令序列。C++/CUDA 只读取该程序格式，不读取 DEAP 对象；具体接口见第 11 节。例如：

```text
opcode[node]      : 终端 / 常数 / 七种函数
operand[node]     : terminal 索引或常数索引
constant[index]  : FP32 常数
program_length   : 不超过 63
```

GPU 对同一程序的不同动作执行相同指令序列。某一步若为 MUL，则整个 warp 都执行 MUL；差异只在各 lane 的输入特征。这样避免“一条 lane 一棵不同的树”造成的解释器控制流分化。

后缀求值的栈深度由树深度限制。根深度为 0、最大深度为 5、最多二元函数时，最多需要 6 个求值栈元素。实现仍需检查编译器是否把动态索引栈放入 local memory；小数组不自动保证寄存器驻留。可以使用固定栈槽或专门化访问消除部分动态索引。

**第一版训练与正式测试采用同一个 GPU 求值后端。** NVRTC 可以将固定程序编译成 CUDA 可执行代码，但这是可选的后续工程优化，不应在训练后悄悄更换执行器，再把两种后端的时间混在一起。[R6]

## 5.3 Terminal 的数据布局和准备

动作特征采用结构分离布局：

```text
action_feature[terminal_id][colony_id][action_id]
```

同一 warp 读取某个 terminal 时，32 个动作值连续，便于合并内存访问。[R5] $u,s,q,c$ 先按 colony 计算；参考特征按两个参考模式计算；区域特征按两个参考模式的四个区域计算。之后通过轻量广播或展开供 32 个候选使用。

终端准备不应随着树中引用次数增加。比如程序五次读取 $v_R$，仍只计算一次该区域的档案分歧，而不是五次访问档案。区域特征的主要读取量受区域大小 16 和档案大小 4 限制。档案须维护 `tour+position` 或邻接视图，保证一次边是否存在检查为常数次读取；不能在每个 terminal 内线性扫描整个档案 tour。稀疏信息素查询的候选行查找成本也应计入，或提前缓存区域边对应的条目索引。

静态数据与动态数据分开：

```text
静态：coordinates / distance metadata / candidate IDs / candidate distances
动态：pheromone / reference tours / archive tours / ant workspaces / feedback EMA
```

对同一实例的不同随机重复，可以共享只读候选与距离；信息素、参考解、档案和随机流必须独立。稀疏候选可使用定长行或 CSR，具体选择依赖对称化之后的度数分布。无向化的 k-nearest 图具有 $O(nk)$ 总边数，并不意味着每个节点的度数都至多为 $2k$。

不使用每个 colony 一份 $n\times n$ 动态信息素表。仅作为容量估算，$n=10,000$、每行 32 个 FP32 信息素值需要约 1.28 MB，不含候选索引、参考边例外、档案与 ant workspace；同规模稠密 FP32 表则为约 400 MB。这是存储计算，不是实测峰值显存。

## 5.4 FACO 内核的 CUDA 移植：借鉴 cuOpt，保留研究对象

### 构造阶段：候选并行，路径顺序保持

每只蚂蚁从批次只读参考解开始。对当前节点，warp 各 lane 读取候选条目的 $\tau\eta$ 和 visited mask，通过前缀和及同一次随机抽样执行 categorical roulette；候选不足、全部已访问、权重下溢或总和为零时，执行明确的原生/扩展回退规则。

不能因为 GPU 上 argmax 更方便，就用最大权重选择替换 roulette。cuRAND 的具体随机数端点约定也须显式处理，避免在抽样上界或累计误差时选到无效条目。shuffle、scan、reduce 优先基于 CUDA/CUB 的受维护原语实现。GPU-MMAS 源码提供了 warp scan/reduce/vote 的参考，但其授权尚未充分明确，暂不直接复制其文件。[S5]

构造路径仍按原生顺序推进；并行发生在候选评价、不同蚂蚁与不同 colony 间，不声称通过并行化消除了路径依赖。

### 局部搜索：evaluate → reduce → commit → reactivate

cuOpt 的 `two_opt.cu` 把 move 查找、归约、提交和受影响节点标记明确分开，值得借鉴。但它依赖完整 routing solution、route-node mapping、约束维度、反向片段和内部 move 数据结构；其核查版本的搜索也并非 FACO 的候选行 checklist 邻域。直接替换会同时改变 LS，因此本项目不把该文件当成可直接链接的纯 TSP 2-opt 库。[S6]

我们的移植保留 `Route::two_opt_nn` 的次序：一次处理一个 checklist 节点，对该节点的两类候选交换并行计算增益，选择一个合法改进，统一完成反转与 inverse-position 更新，再重新激活受影响端点。GPU tie-breaking 按固定候选顺序和交换类型处理；不把“每个 checklist 节点内选最好”悄悄改成“整条 tour 全局选最好”。[S2]

第一版不同时提交同一条 tour 上的多个未验证互不冲突的移动，也不引入 cuOpt 的 sliding-TSP、跨 route 交换或完整进化搜索。所有 Static/GP 对照使用同一个移植 LS。

### 候选排序是算法前提，不只是存储选择

原生 LS 在候选边距离不短于当前边时执行 `break`，这要求候选行按真实距离排序。alpha-nearness/POPMUSIC 决定**哪些边入选**后，应生成独立的距离有序 LS 视图；原始先验分数与顺序可另存用于候选截断。不能将按 alpha 分数排序的行直接交给原生提前退出逻辑。[S2]

也可以删除提前退出并扫描全部候选，但这会改变检查集合与成本，须作为明确且对所有方法一致的 LS 配置。主协议采用距离有序视图，保留原有距离过滤条件。

### Tour 表示、临时存储与 checklist

第一版采用原生对应的 `tour[]+position[]`，便于与 CPU 对照验证。重定位涉及的数组平移使用独立 scratch 或两阶段协作，避免并发原地移动造成读写覆盖；2-opt 反转采用固定的短段/补段语义。共享内存只承载局部候选、归约与小片段，不默认整条 10K tour 可放入单个 block 的 shared memory。[S2, S6]

cuOpt 的 `tsp_route.cuh` 展示了 predecessor/successor 视图和 shared-memory 大小计算，可作为后续表示优化参考；它并不证明替换后可无代价地兼容我们所有 LS 操作。应先验证正确性与剖析瓶颈，再决定是否采用链接表示。[S7]

checklist 可使用有界队列、pending bitmap 或 epoch 标记，但一个节点被处理后必须允许在后续 move 影响它时再次入队。不能把“已经处理过”永久当作“不再需要检查”。维持原生接受改动上限，并额外使用统一的 move-evaluation 上限及 deadline；新增截止条件对所有共同底座控制器一致并写入配置。

容量估算：两份 32 位 tour/position 数组在 $n=10,000$ 时为每只蚂蚁约 80 KB，128 只蚂蚁约 10.24 MB；尚不含 visited、scratch、checklist、参考解和档案。它们必须加入 colony 容量规划，不能只计算信息素表就估计最大 batch size。

### 同步、重启与启动开销

构造和 LS 期间参考解与信息素只读。批次结束后依次执行结果归约、档案/参考选择、信息素更新、缓存刷新和反馈更新。重启事务在下一批开始前完整完成；不把异步 CPU `source_sol_local_update` 原样带入 GPU 而破坏只读快照语义。[S2]

cuOpt 的 `cuda_graph.cuh` 展示了 stream capture、更新失败后重新实例化等组织方式，但其类明确不是线程安全的。可在基础实现稳定后，用自己最小的 CUDA Graph 封装减少重复 kernel launch；每个 worker 独立管理 graph 和 stream，不作为训练算法的额外层次，也不先将整套 cuOpt 设为强依赖。[S8]

线程块内同步必须一致到达；非法候选不能使部分线程提前退出后其他线程仍使用完整 collective。随机流按实例、重复、批次、蚂蚁和阶段划分。跨实现只在规定的运算和随机条件下比较逐步一致性，其余场景比较操作不变量与统计分布，不无依据要求所有 CPU/GPU 轨迹逐位相同。[R4, R5]

## 5.5 固定预算 fitness 的 GPU 评价语义

**同一计时评价任务内，GPU 只评价一个 GP 程序。** 该程序同时求解一个固定批大小、相同规模的实例—种子集合。不同程序使用相同的批大小、GPU 型号、输入面板和时间上限；跨 GPU 才并行不同程序。

此时 $B_n$ 表示这个固定批量工作负载的共同 wall-clock 预算，而不是每个实例独享整张 GPU 的预算。在式 (4)、(10) 中，固定运行配置包括 GPU、batch size 和并发 colony 数。不能把批量吞吐量换算成未经测量的单实例时延。

GPU 批内发生实例间同步等待也属于该批量执行策略的实际成本。第一版按规模分组，不在同一工作波中混放 TSP100 与 TSP10K。主方法与共同底座对照使用相同批处理逻辑。

时间从实例特定初始化、候选准备和必要传输开始计算。训练中可缓存只读候选以避免重复构图，但相应的实例预处理成本必须按一致规则占用预算或另设明确的“候选已给定”训练配置；正式端到端结果不可免费排除某个方法独有的预处理。最简单的主协议是将固定预处理费用保留在预算内，实际搜索使用扣除该费用后的预算，并在独立端到端运行中核实。

CUDA kernel 启动是异步的，计时需要正确的同步或事件语义；不同 stream 的交错可能污染计时。[R5] 使用小型完整提交批次与时间戳维护 best-so-far，只评价截止前完成的解。若最后一次 kernel 越过截止，不能计入它产生的迟到改进；同时记录实际超限及丢弃的末尾工作。计时粒度必须足够细，防止不同程度动作因超长 kernel 受到不可控偏差。

同一规格 GPU worker 可以顺序评价若干 GP 个体；CPU 集群负责图构建、原生求解器对照和实验调度。这里的并行化不改变任何个体的 fitness 精度，也不引入多保真筛选。

## 5.6 工程验收，而不是新增 RQ

正式训练前完成 tour 可行性、增量成本与完整重算、CPU/GPU 特征一致性、GP 求值误差、重启状态切换、候选图限制和截止行为测试。FP32 的 GP 分数存在数值平局时使用冻结规则；长度计算使用符合数据标签的精度。

源码移植还须通过三项专项测试：相同重定位序列产生相同 MNE 计数；信息素 reset 同时覆盖默认值与缓存；打乱候选先验排序后，经 LS 距离视图转换不改变预期候选检查集合。Hard 模式对重定位的全部重连边和 2-opt 的两条新边逐条验证。上述是正确性验收，不增设 RQ 或主实验。

正式结果附一张资源表即可：候选准备、特征与 GP、扰动、LS、信息素维护的时间占比，峰值显存和离线 GPU/CPU 资源。它们支撑实验可信性，不再扩张成独立的 GPU research question。

# 6. 实验公共协议

## 6.1 数据使用

| 数据 | 主用途 | 是否参与主方法训练/选择 |
|---|---|---|
| TSP100 | 正确性、小规模回迁与饱和效应检查 | 否 |
| TSP500 | 训练、独立验证、同规模测试 | 仅训练与验证划分可以 |
| TSP1K | 训练、独立验证、主要同规模测试 | 仅训练与验证划分可以 |
| TSP10K | 大规模零样本迁移 | 不参与训练、特征设计或程序选择 |
| TSPLIB | 外部实例结构与支持距离类型上的检验 | 不参与训练和选择 |

实例数量根据已有文件核实。建议从 500/1K 各不少于数千个训练实例建立训练池，每代每规模抽取 16 个实例、每实例 2 个种子；验证各 200–500 个；冻结测试各 200 个。TSP10K 的测试可以从 100 个独立实例起步。上述是起始采样建议，不是已经读取得到的数据规模。

所有派生于同一基础点集的置换、旋转或子采样实例按父实例分组切分，避免跨划分泄漏。最终五个独立进化重复分别由验证集选出程序；不得按测试成绩选训练 seed。

对于 TSPLIB，按原始距离类型计算目标；不能用未经规定舍入的欧氏距离替代官方距离定义。[R9] 只纳入已经正确实现且有匹配标签的对称实例。合成数据也须核对坐标缩放、舍入与最优标签的一致性。

## 6.2 预算与主要指标

使用短、中、长三档预算，数值通过不接触正式测试的统一底座预实验确定后冻结。主统计终点为中预算下的最优 gap；短、长预算用于判断结论是否只在一个时间点成立。

GPU 批量实验固定 GPU 型号和 batch size，所有共同底座变体使用相同 kernel 与资源。另报告固定单实例执行配置下的端到端结果，明确硬件与线程数。原生 CPU MMAS/FACO/LKH 与 GPU 方法的结果不能未经说明地称为硬件等价比较；主要机制结论来自同一 CUDA 底座的对照。

报告三个层面的指标：最终最优 gap；best-so-far 质量—时间曲线；解释行为的返回参考率、实际改边数、重启频率和 LS 工作量。行为指标不作为 fitness，也不因为它们增加或减少就自动认定性能更好。

最优标签只由 evaluator 读取。若在线求解器已恰好找到最优，它仍不知道标签，因此不能通过标签提前停止并把剩余时间重新分配。

## 6.3 比较对象的最小集合

**共同底座主对照：**

1. **Static-Control：** 充分调参的固定程度、固定区域选择规则及固定重启概率/周期；使用与 GP 相同档案、图和 LS。
2. **Rule-Control：** 简单的停滞阈值策略，配置预算与 GP 的离线资源一并披露。
3. **GP-NoFeedback：** 使用相同函数集与动作空间，删除 $s,q,c$ 后重新训练，仍可读取当前参考/区域结构、信息素特征与搜索时钟。
4. **GP-Full：** 本文完整程序。

已有自适应幅度 FACO 是相关对照。[R2] 核查的 `run_faco_apt` 使用五个 MNE 值 4–8，并可逐蚂蚁更新共享源解；它与本方案的四个 MNE 值及批次级控制不同。保留原生 CPU 版本作为外部对照；共同 CUDA 底座上的 bandit 采用与 GP 相同的程度集合和批次节奏，明确标为接口适配，不冒充原生复现。二者放入 E1 的结果或附表，不形成新实验章节。[S2]

**外部算法参照：** 原生 `run_mfaco`、`run_faco_apt`、ACOTSP 的 MMAS+LS，以及支持相同距离和规模的 LKH/DyNACO。cuOpt 以整个求解器作为可选 GPU 外部参照，而不是嵌入 GP 的 LS。外部参照用于衡量竞争力；它们的完整内核、硬件与训练条件可能不同，不承担单组件归因。优先保留最接近的自适应 FACO 与充分调参的共同底座；其他外部方法按代码和距离支持范围纳入，不为覆盖所有软件而扩张科学问题。[R2, R3]

E1 表中同时列出 FACO-Native、FACO-Control 的静态配置与 GP-FACO，使读者能分清原生→固定扩展和固定扩展→学习两段变化。涉及跨 CPU/GPU 的第一段不作单组件性能归因。cuOpt/ACOTSP/LKH 的许可与版本隔离见第 10 节。

## 6.4 统计推断

合成测试建议每实例使用 10 个求解种子，TSPLIB 使用 20 个；以实例为统计单位，先汇总同一实例的求解重复，再计算配对差异与 95% 置信区间。独立 GP 演化重复作为第二层变异来源报告，不把同一实例的多个 seed 当成独立实例。

E1 的主要比较预先限定为 GP-Full 对 Static-Control、Rule-Control 和 GP-NoFeedback，并进行多重比较校正。E2 直接报告析因效应与交互的区间。E3、E4 分条件和实例组报告，不用大量 TSP100 易实例稀释 10K 或 TSPLIB 上的负结果。

统计显著与实际有用分开讨论。TSP100 若普遍达到最优，说明该预算下发生饱和，不能由此证明各种控制策略在更难条件下等效。

# 7. 四组主实验及其 RQ 对应

## E1 → RQ1：反馈是否改变有效决策，而不只是增加程序复杂度？

### E1 的目的

同时建立方法的基本有效性和“LS 后反馈有用”的证据。E1 不研究既有 GP 状态转移规则为什么失效，也不比较旧项目的训练机制。

### 主要比较

在同一固定候选图和相同 LS 下，比较 Static-Control、Rule-Control、GP-NoFeedback 与 GP-Full；可在同一表增加同接口 bandit。以 TSP500/1K 的冻结同分布测试为主，大规模结果由 E4 系统报告。

GP-NoFeedback 必须在删减终端集下重新训练，不能只把一个已学程序的输入设为零。它仍然包含当前结构和信息素信息，因此结论应表述为“显式 LS 后反馈的额外价值”，而不是“有历史 vs 完全无历史”。

### 一个有针对性的状态分叉分析

为直接检查状态依赖性，从开发阶段的运行中，按预先规定的高/低终态返回率和高/低修复工作量采样快照。从同一快照出发，保持参考解和区域不变，只比较 $k=2$ 与 $k=16$；执行该动作后使用同一个固定继续策略，在相同额外时间内评价结果。

分叉不需要覆盖所有 32 个动作，也不需要单独训练价值网络。它只回答一个明确问题：**增加扰动程度的价值，是否会随 LS 后状态改变？** 使用多个分叉种子并按基础实例聚合；状态分层、动作与评价时长在看到结果前冻结。

若状态来自多个算法，应记录来源并分层，避免只从 GP 自己擅长的轨迹中选取快照。这里的状态—动作效果限定于样本状态与固定继续策略，不外推为全局最优策略。

### 输出与判断

**主输出：** 一张 gap/anytime 比较表或图；一张不同状态下“大扰动相对小扰动”的配对收益图。

**支持 RQ1 的证据：** GP-Full 在独立实例上优于去反馈重训，且动作收益随反馈状态出现可重复变化。

**不支持的情况：** GP-Full 与去反馈版本没有稳定差异；或者大/小扰动的排序在各状态基本一致。仅完整方法超过一个未充分调参的静态值，不足以证明反馈价值。

## E2 → RQ2：局部扰动与搜索中心重启如何相互作用？

### $2\times2$ 析因设计

定义两个因素：$L$ 表示是否学习局部区域与程度；$R$ 表示是否学习重启。所有变体仍使用相同动作接口和最多一棵 GP 树，仅通过动作 mask 限制哪些选择由程序决定。

| 变体 | 学习区域与程度 $L$ | 学习重启 $R$ | 未学习部分如何处理 |
|---|---:|---:|---|
| $M_{00}$ | 否 | 否 | 验证集充分配置的固定/手工规则 |
| $M_{10}$ | 是 | 否 | 重启由与 $M_{00}$ 相同的固定规则决定 |
| $M_{01}$ | 否 | 是 | 区域与程度由与 $M_{00}$ 相同的规则决定 |
| $M_{11}$ | 是 | 是 | 完整动作由一棵树联合评分 |

$M_{10}$ 与 $M_{01}$ 必须各自重新训练。不是冻结完整程序后关闭一个输出，也不为每个因素再引入一个独立策略网络。$M_{00}$ 的规则具有与其他方法相同的参考切换和档案接口，避免把新增底座能力误认为学习收益。

### 交互的数学定义

令 $\mathcal L_{lr}$ 为给定实例组和预算下的平均 gap，定义：

$$
\operatorname{Syn}=
\mathcal L_{10}+\mathcal L_{01}
-\mathcal L_{00}-\mathcal L_{11}.
\tag{11}
$$

$\operatorname{Syn}>0$ 表示该损失尺度上，联合收益大于两个单因素收益之和；$\operatorname{Syn}\approx0$ 表示近似可加；负值表示收益重叠或拮抗，但不自动说明完整方法绝对更差。

同时报告三个基本差异：$\mathcal L_{00}-\mathcal L_{10}$、$\mathcal L_{00}-\mathcal L_{01}$、以及完整方法相对最佳单因素版本的改善。不能只报告交互而忽略最终质量。

### 机制观察

记录各变体的重启频率、重启后的扰动程度、LS 终态返回率和后续改进。观察“学习局部扰动后是否减少重启需求”或“重启后是否更偏向小改动”等模式，但不把这些相关性独立当作因果结论；因果证据主要来自上述受控因素变化。

### 输出与判断

**主输出：** 四个变体的结果表、一张交互效应图、少量与交互对应的搜索行为摘要。

**结论边界：** 本实验针对局部修改与“切换参考解＋重置信息素”这一重启组合。它不单独证明信息素不可替代，也不拆成十余个重启子模块实验。若联合收益为零或负，接受两类机制存在替代关系这一结果。

## E3 → RQ3：高质量候选先验与探索自由是互补还是替代？

### 图与限制方式的四个条件

| 候选先验 | Hard：永久共享硬图 | Escape：保留有限探索出口 |
|---|---|---|
| alpha-nearness | alpha-Hard | alpha-Escape |
| POPMUSIC | POPMUSIC-Hard | POPMUSIC-Escape |

alpha-nearness 利用 1-tree 相关结构评价边；POPMUSIC 利用子问题改进和 tour 结构生成候选。[R7, R8] 本文不把这两种先验的采用本身视为创新。

**Hard 的定义：** 固定图 $E_0$ 包含候选边与共同初始可行 tour 的边；扰动与 LS 的每条新增边都必须属于 $E_0$。仅“用候选边枚举一个 2-opt 端点”不等于满足这个约束，另一条新增边也要检查。

节点重定位还必须检查移走节点后形成的补边，以及插入节点产生的两条边；只检查当前节点到选中节点的那一条边不满足 Hard 约束。保持可行 tour 不等于保持在固定图内。

**Escape 的定义：** 保持相同基础候选，并提供有界备用候选来源。在固定比例的机会中，用备用候选替换一部分常规候选进行扰动；LS 允许与该扰动相关的明确重连例外。每个节点每次检查的候选槽位总数保持相同，不简单通过多检查一倍候选获得优势。新增或例外边的来源、比例和缓存容量固定，不由 GP 学习。

备用候选使用候选先验的后续排名与少量均匀采样节点构造，完全不使用最优标签。Hard 与 Escape 使用同样的构建流程准备候选资料以隔离访问限制；端到端补充结果还需报告 Hard 不准备备用资料时的实际节省，不能只提供一个故意承担无用成本的 Hard 对照。

当候选机制发生变化时，所有需要重新检查的节点都要被激活；否则“增加了出口”可能没有真实进入 LS。Escape 不宣称所有生成 tour 都属于同一个固定稀疏图，其访问预算有限与永久硬约束是不同概念。

### 候选导入与目标语义

候选生成先在 CPU 侧用经过许可与版本核验的 LKH/POPMUSIC 工具完成，输出节点 ID、候选 ID、原问题距离及先验分数，再上传 GPU。主候选可按先验决定成员，但 LS 使用按原问题距离排序的独立视图，避免原生 `break` 提前终止。文件导出接口与候选工具版本在实施时锁定；本轮尚未验证 LKH 源码包的构建与导出链路。[S2, S12]

不将 cuOpt 的 `WaypointMatrix` 当作 Hard 候选图接口。其公开 API 使用 waypoint 图的最短路径生成目标点间成本并可展开中间路径；这与“原 TSP 的 tour 每条边必须属于候选集”不是同一约束，可能改变目标距离及中间节点经过方式。[S10]

### 比较控制与学习控制

四个图条件分别比较 Static-Control 和同样配置的 GP-Full。为研究图条件本身的作用，GP 在每个条件中使用相同训练数据、函数集、终端集和离线资源独立训练。不能仅把在 Escape 中学到的程序直接用于 Hard 后变差，就断言 Hard 没有可学习空间。

默认方法选用的候选条件在测试前冻结；E3 不用于事后挑选最有利于主结果的图。

### 控制初始化与候选容量

各条件使用相同初始 tour，避免把 POPMUSIC 可能提供的初始解优势算作候选图的作用。基础候选实际边数、访问槽位和内存预算需要匹配；无向化后的实际度数和边数必须记录。

在同时间结果之外，报告 move-evaluation 数作为成本解释。若 Escape 只因额外计算获得改进，不能直接称为更有效的结构探索。

### 用最优标签解释限制，而不是泄漏答案

对有最优 tour 标签的实例，离线计算：

$$
\operatorname{Recall}^*(E_0)=\frac{|E_0\cap E(x^*)|}{n}.
\tag{12}
$$

覆盖率为 1 说明该标注最优 tour 可行，但不保证局部搜索可从当前解到达它；覆盖率小于 1 也不能排除图中有其他等长最优 tour。只有最优长度标签的实例不计算最优边覆盖率。

### 输出与判断

**主输出：** 四个条件中 Static 与 GP 的成对结果；Hard/Escape 差异；覆盖率与性能的诊断图。

**支持 RQ3 的一种结果：** 在两类高质量先验下，有限出口仍带来收益，并改变 GP 相对静态控制的优势。

**另一种有意义结果：** 高覆盖、高质量候选使出口收益消失，表明在该规模与预算内，候选先验可部分替代额外探索。两种结果都回答科学问题，不预设 Escape 必须胜出。

## E4 → RQ4：冻结规则的规模与实例结构迁移

### 训练和选择严格停留在 500/1K

使用主配置下在 TSP500/1K 训练、经独立验证选出的程序，冻结其函数树、常数、归一化方式、动作集合与底座参数。TSP10K 和 TSPLIB 的解质量、状态轨迹或最优标签都不能用于修改这些内容。

10K 的运行预算只能根据预先定义的规模规则或独立底座校准确定，不依据 GP 在 10K 的成绩挑选。查看 10K 结果后再调整程度集合或 terminal 阈值，就不能继续称为本轮零样本结果。

### 两条迁移轴

**规模轴：** 同生成分布的 TSP100、500、1K、10K。500/1K 提供训练规模内参照；100 检验回迁及预算饱和；10K 是主要大规模外推。

**实例结构轴：** TSPLIB 的支持实例，按规模和距离类型分组逐实例报告。如果合成数据确实包含多个生成分布，再使用未参加训练的分布作为补充；若只有一个生成器，不虚构跨合成分布实验。

GP 规则不输入城市编号和原始 $n$，但这不构成泛化保证。固定程度集合和采样特征本身也可能在 10K 上失效，应通过结果判断。

### 对照与指标

在所有目标数据上比较同样冻结的 Static-Control、Rule-Control、GP-NoFeedback 和 GP-Full，并报告相关外部求解器。使用目标实例自己的最优值计算 gap，在线仍不读取标签。

主要看：GP 相对共同底座的改善是否保留；规模增大后动作分布与实际改边数如何变化；哪些 TSPLIB 结构或距离类型失去优势。跨规模对比不直接比较未经归一化的绝对秒数。

### 输出与判断

**主输出：** 规模—性能图、10K 的质量—时间曲线、TSPLIB 逐实例表和分组摘要。

若 1K 有效而 10K 失效，结论是该程序尚未获得足够的规模迁移能力，不以“需要更大的 GPU”替代解释。若在部分 TSPLIB 类型上失效，应明确适用范围，不通过删除不利实例维持平均优势。

# 8. 伪代码

以下为定义语义的算法级伪代码，不是已经实现并测得加速的 CUDA 求解器。

## Algorithm 1：以 FACO 为固定内核的一个 colony

```text
Input : instance I, frozen GP tree f, budget B, fixed FACO-Control configuration
Output: best feasible tour completed within B

1  Start timing; build a cheap feasible incumbent xGB
2  Prepare candidates, LS distance-sorted views and initial improvement
3  r ← best completed initial tour; epochBest ← r; archive ← {r}
4  Initialize stored pheromone, default pheromone, product cache and feedback

5  while budget remains:
6      rAlt ← fixed alternative-reference rule(archive, r)
7      Build four START-NODE regions for each legal reference mode
8      Build up to 32 actions a = (restart, region, MNE)
9      Compute twelve PRE-ACTION terminals and experiment masks
10     a* ← evaluate one GP tree; masked argmax with frozen tie rule

11     if a*.restart = 1:
12         r ← rAlt; epochBest ← r
13         Recompute epoch pheromone bounds
14         Reset stored trails AND the default nonstored-edge pheromone
15         Clear stale ant/epoch state; invalidate and rebuild product cache
16         Preserve xGB and archive

17     parent ← r; freeze parent and pheromone for this batch
18     parallel for each ant:
19         y ← copy(parent); curr ← sample start node in selected region
20         Clear visited; mark curr; MNEcount ← 0; visitedCount ← 1
21         checklist ← empty
22         while MNEcount < a*.MNE and visitedCount < n and budget remains:
23             Enumerate candidates; mask moves violating declared graph constraints
24             sel ← native FACO roulette / fixed fallback among eligible moves
25             if no legal next selection exists: stop this construction
26             oldPred ← predecessor of sel in y
27             Mark sel visited; relocate sel immediately after curr in y
28             if undirected edge {curr, sel} is absent from parent:
29                 MNEcount ← MNEcount + 1
30                 Add curr, sel and oldPred to pending LS checklist
31             curr ← sel; visitedCount ← visitedCount + 1
32         z ← fixed FACO checklist 2-opt(y, distance-sorted candidates, deadline)
33         Register completed feasible result, timestamp, MNE and actual edge changes

34     Reduce only completed current-batch ant results
35     iterationBest ← best result from this batch
36     Update xGB and epochBest without discarding earlier epochBest
37     Update archive with the same fixed rule for all controllers
38     zDep ← epochBest with fixed probability, otherwise iterationBest
39     r ← zDep; apply fixed evaporation/deposition and refresh product cache
40     Update feedback from actual LS outputs and proceed

41 return best incumbent registered no later than deadline
```

第 23 行的合法性过滤在抽样前检查重定位的所有新增边，而不是只检查当前到选中节点的那条连接。不能先提交非法移动再事后修复。原生回退与 E3 扩展回退的选择由固定图配置决定，GP 不选择另一套转移公式。

常规批次不复用旧个体最优；若另测原生 `keep_better_ant_sol=true`，其跨批次状态必须单独标记，且重启时按明确规则初始化。重启不丢失历史最优结果，但也不能通过陈旧 ant 或缓存把旧搜索中心隐式恢复。

任何迟到结果不进入指标。数据搬移、反馈、边界刷新、reset 和缓存构建均计入预算；候选耗尽时返回已有可行 tour，不调用未计时的外部求解器。

## Algorithm 2：Python/DEAP 管理的标准代际 GP

```text
Input : training pools D500, D1000; validation set V;
        function set F; terminal set T; fixed per-scale budgets Bn
Output: one validation-selected program per evolution seed

1  Initialize DEAP population P: 128 trees, depth ≤ 5 and size ≤ 63
2  Start persistent GPU workers; coordinator does not initialize CUDA
3  for generation = 1,...,50:
4      Sample common training panels and solver seeds for this generation
5      Invalidate ALL fitness values, including unchanged elites
6      Export each tree to validated postfix IR
7      for each program, dispatched to a homogeneous GPU worker:
8          Solve each fixed-shape per-scale batch in one C++/CUDA call
9          Return completed costs/status; online solver never receives labels
10     Compute J(f) outside the solver; assign fitness.values = (J(f),)
11     Save current generation summary and its winner's identity
12     if generation == 50: break       // final population is evaluated
13     Clone four elites under the declared tie rule
14     Select other parents using size-4 tournaments
15     Apply DEAP varAnd: pair crossover 0.8, then individual mutation 0.2
16     Replace over-limit offspring using the frozen parent-fallback rule
17     P ← cloned elites + offspring
18     Save next-generation checkpoint, RNGs, task manifest and config hashes

19 Evaluate the deduplicated generation winners and final population on V
20 Select using validation gap and the frozen size tie rule
21 Export IR, readable tree, constants, terminal/opcode definitions and config
22 Return the frozen program; never choose by test performance
```

每代所有个体在同一面板以相同精度评价。验证使用统一的固定协议；不同代随机面板上的原始 fitness 不直接用于跨代冠军排序。DEAP 负责遗传操作，完整求解与在线动作选择均在 C++/CUDA 内执行；第 11 节说明接口和调度。

## Algorithm 3：一个 warp 对同一程序的 32 个动作评分

```text
Input : one postfix program; feature[terminal][colony][action]; legality mask
Output: best legal action for this colony

1  lane ← lane index in the warp          // action 0,...,31
2  Every lane loads the SAME program instruction sequence
3  stack ← six bounded FP32 slots
4  for instruction in program:
5      if instruction is TERMINAL:
6          push feature[instruction.terminal][colony][lane]
7      else if instruction is CONSTANT:
8          push the same program constant
9      else:
10         Pop the declared number of operands
11         Apply ADD / SUB / MUL / MIN / MAX / ABS / AQ
12         Clip the result to [-8,8] and push
13 score ← the final stack value
14 if this action is illegal: score ← -infinity
15 (bestScore, tieKey, actionId) ← warp-wide masked reduction
16 lane 0 writes the selected action
```

这里由 opcode 决定的分支对整个 warp 一致；不能将同一 warp 的各 lane 分配给不同 GP 树。非法动作也保留其 lane 参与所需的 warp 操作，不能因不合法而提前退出后仍使用完整 warp mask。特征计算和合法性准备在评分之前完成，解释器内部不执行图搜索。

# 9. 研究执行顺序与论文组织

## 9.1 实施顺序

首先锁定作者 `run_mfaco` 版本并建立 CPU 语义参考，随后移植固定 FACO 内核，明确记录同步、个体记忆和重启的适配；再完成 CPU/GPU 校验与十二个 terminal 的低开销数据通路。随后实现单树评分与标准 GP，先完成 E1。若反馈没有可重复收益，优先检查 terminal 是否真正区分了相关状态，而不是增加多保真、模型层数或更多输出头。

E1 后实施 E2，确定是否有必要联合学习两个尺度；再实施 E3，确定结构先验与探索出口的关系。E4 使用提前冻结的主方法做严格迁移测试。可以并行运行已冻结配置的任务，但不能因 E4 的结果反过来修改本轮程序。

## 9.2 建议的论文证据链

**Introduction / Motivation：** 有限预算下结构保留与结构释放的矛盾。

**Method：** 节点重定位 FACO 的固定执行机制，加上起始区域、MNE 和重启三个明确接口；一棵 GP 树、七个函数、十二个终端、一个完整运行 fitness。

**Experiments：** E1 证明或否定反馈价值；E2 识别两尺度关系；E3 检验候选先验与探索自由；E4 确定规则可迁移范围。

**Implementation：** CUDA 映射、成本计入、数值一致性与资源表，不将工程问题包装成新的科学 RQ。

**Discussion：** 接受可能的替代关系、无额外反馈收益或迁移失败。结论依据具体证据，不预设 GP、restart 或图外探索在所有条件下都必需。

## 9.3 本轮有意不研究的内容

不回溯旧 GP 状态转移设计失效的原因；不学习常规信息素更新公式；不学习 LS 内部策略；不学习候选图生成器；不同时设计多个重启强度和参考排序器；不引入多保真、代理模型或额外强化学习训练器；不把 GPU 加速另立为 RQ。

这些取舍不是认为相关问题没有价值，而是让当前论文围绕一个明确对象形成可归因结论：**局部搜索之上的结构控制规则。**

# 10. 源码依据、复用边界与实施清单

本节是前述方法的工程落实，不增加第五个 RQ，也不增加第五组主实验。以下“可复用”表示具有明确的接口或实现参考价值，并不表示已经在本项目成功编译、集成或测得加速。

## 10.1 来源选择与许可状态

| 来源 | 本轮锁定版本或状态 | 在本项目中的角色 | 复用决定 |
|---|---|---|---|
| 作者 `ACO-TSP-Adaptive-Tuning` | `a904e6a8786d48593ef1cac975edef5ed8920af3`，2025-02-25 | `run_mfaco` 为算法语义参考；`run_faco_apt` 为自适应对照 | MIT；保留版权与许可，抽取最小 CPU 参考和必要数学语义，再实现 CUDA 内核 [S1–S4] |
| 作者 `FocusedACO` | 本轮确认原始仓库位置，未锁定并审查其全部文件 | 追溯 2022 FACO，不作为当前实现的模糊替代来源 | 不能将另一个仓库的 MIT 许可或某段逻辑无依据推广到此仓库 [R1] |
| NVIDIA/cuOpt | `0cccfd3e426f391feaadc2afd1f9ead1533ab2ac`，2026-09-03 | routing 源码的 GPU 设计参考与外部求解器 | 核查文件 Apache-2.0；局部 helper 可在拆解依赖并履行许可后适配，不直接绑定整个内部 LS [S6–S11] |
| 作者 `GPU-based-MMAS` | `8aac0556177fdc8e022c98c56886e9b8baf99869`，2020-01-09，`origin-master` | warp 级 ACO 选择、scan/reduce/vote 的历史实现参考 | 核查根目录树及文件头未找到明确的项目级许可；在确认授权前不复制进发布代码 [S5] |
| ACOTSP | 官方列出的 V1.03；本轮未构建源码包 | MMAS CPU 外部基线 | 官方列明 GPL；不默认按 MIT/Apache 导入核心库 [S13] |
| LKH / POPMUSIC | 官方来源已确认；源码包版本、hash 和导出链路待实施时锁定 | CPU 候选生成及 LKH 外部质量参照 | 官方标明学术和非商业用途；遵守原条款，保持独立来源记录 [S12] |

许可是逐来源、逐文件的事实，不能因为代码在 GitHub 上公开就认为可任意复制。保留 `LICENSE`、适用的 `NOTICE`、文件版权头和改动记录；第三方随附文件单独核查。ACOTSP/LKH 的独立进程调用用于保持工程边界，不作为自动消除许可义务的法律结论。公开发布前仍应由项目负责人完成许可证审查。

cuOpt 的官方安装文档在本轮列出稳定版 26.08 与 nightly 26.10。上表锁定的是**源码审查 commit**，不是声称已安装并验证同版本 wheel。若外部基线使用稳定包，应另记完整包版本、构建与依赖，不能把 main 源码事实不加核验地投射到任意旧包。[S11]

## 10.2 FACO：应读、应移植、应保留的具体位置

| 位置 | 已核查的作用 | 计划中的具体处理 |
|---|---|---|
| `src/faco.cpp::run_mfaco` | 完整源 tour 上的节点重定位；新连接计数；批次后参考解与强化 | 作为 CPU 行为参考，提取固定执行路径，不把主循环中所有选项都变成 GP 动作 |
| `select_next_node` 的无 `Ant` 重载 | 有 visited mask 的候选 roulette、备用表及全体未访问回退 | CUDA prefix-scan 抽样；保留概率与回退语义，不替换成贪心 argmax |
| `Route::relocate_node(target,node)` | 把 node 放到 target 之后；原实现数组移动与增量成本 | 保留操作含义，重写安全的 block 协作移动；与 CPU 对照合法性和成本 |
| `Route::two_opt_nn` | 实际用于该路径的 checklist LS、两类交换、距离提前退出、影响端点重激活 | 移植这个函数的语义，而不是误把另一处同名初始化 LS 当成主内核 |
| `Route::flip_route_section` | 反转给定片段或较短补片段，修复 inverse positions | 固定表示语义和 tie-breaking；不能仅验证长度却忽略后续方向依赖 |
| `ACOModel` / `calc_trail_limits_cl` | 保留率、LS 下的候选相关信息素边界、强化 | 提取为固定规则；epoch 替换单独声明 |
| `src/pheromone.h::CandListPheromone` | 候选边条目、默认值、蒸发与部分强化 | GPU 稀疏条目＋默认标量；reset 涵盖所有状态，不能只调 `set_all_trails` |
| `src/faco.cpp::run_faco_apt`、`src/mab.h` | 原生 bandit 对照入口；逐蚂蚁 MNE 选择 | 保留原生外部版本；同底座 bandit 另标记为批次节奏适配。`mab.h` 各策略数值实现需在真正复现时进一步逐函数测试 |
| `src/progargs.h` | 实际默认配置及可选行为 | 每次运行输出显式配置，尤其个体记忆和异步源更新开关 |

核查的 `ProgramOptions` 默认包括主候选 16、备用 64、LS 候选 20、MNE 8、`beta=1`、信息素保留率 0.5、最佳参考概率 0.01，以及启用个体最好解保留和局部源更新的选项。这只是该结构中的默认值，不表示论文每个实验或命令行解析最终一定采用这些数值。主实验显式设置，不能依赖默认主算法名恰好选中所需函数。[S3]

原生构造代码有候选长度上限 32。GPU 的“一次 32 lane”不是对所有图度数的数学限制；E3 若加初始 tour 边或使用可变度候选，需要按固定块宽分段处理、padding 与 mask，而不是无声截断造成候选丢失。用于合法性判断的图、用于主转移枚举的行、用于 LS 的距离排序行应分别定义。

## 10.3 cuOpt：有价值的部分与不采用的部分

### A. 最值得借鉴：局部搜索的任务分解

核查文件 `cpp/src/routing/local_search/two_opt.cu` 包含 `find_two_opt_moves`、`execute_two_opt_moves` 和 `mark_impacted_nodes`。它们展示了候选并行评价、归约、应用和受影响节点维护之间的边界。[S6]

本项目采用这一**组织模式**，但写面向对称 TSP、固定 FACO checklist 的轻量内核。cuOpt 对 route dimensions、约束片段、sampled node 到 route 的映射和共享内存的依赖，对我们的单 tour 问题不一定划算。第一版不移植这些额外状态，也不把它的完整邻域替换成新的默认 LS。

### B. 可借鉴或小范围适配：CUB top-k 与缓冲区组织

`cpp/src/routing/util_kernels/top_k.cuh` 使用 CUB 的 block load、radix sort、shuffle、store 等组件实现行内 top-k；模板约束、候选类型、RAFT span 和部分头文件仍与 cuOpt 内部结构有关。[S9]

因此首选直接使用 CUB 组成我们的候选归约和必要的 top-k，而不是复制整条依赖链。GP 的 32 动作只需要稳定 argmax，不需要为了复用该文件额外引入 top-64 排序。该 top-k 实现也不负责计算 alpha-nearness，不能把二者混为一谈。

### C. 后续可选：predecessor/successor 表示和 CUDA Graph

`tsp_route.cuh` 的 pred/succ 视图可参考数据组织；`cuda_graph.cuh` 的捕获与更新流程可参考重复启动组织。但前者不能直接替代所有 tour 操作，后者有非线程安全约束。采用前须在本问题上证明正确与必要，不作为第一版不可缺少的依赖。[S7, S8]

### D. 作为外部求解器：Solve / BatchSolve

官方文档公开了 routing `Solve` 与 `BatchSolve`，并给出了小规模 TSP 批量示例。可据此构建外部 GPU 对照，但该示例不构成 10K 批处理速度的证据。[S10, S11]

对原始对称 TSP 的 wrapper 必须验证单车辆、闭合起终点、所有要求节点恰好访问、无丢单、无额外目标或惩罚改变问题。输入矩阵严格采用标签对应的距离定义；若内部类型转换改变整数边权或连续距离，需单独披露，不以转换后的成本重新声称达到原标签。读取结果后用独立评价器重算原目标。

10K 稠密 FP32 成本矩阵仅原始边权就约 400 MB；这是容量下界计算，不是 cuOpt 实测用量，更不能推断其单 tour 规模上限。多个实例的矩阵、内部搜索工作区和数据搬运需实际测量。所有预处理、传输、建模和求解开销纳入所报告的端到端配置。

本轮核查的源码示例与 latest API 页对 `BatchSolve` 返回值的呈现并非完全一致，因此 wrapper 以实际锁定版本的实现和最小调用测试为准，不直接混用不同版本片段。C++ 内部 API 的稳定性风险也由官方 README 明示；不能将内部 `detail` 函数作为无版本约束的长期接口。[S10, S11]

### E. 明确不采用

不把每只蚂蚁的 LS 替换为完整 cuOpt Solve；不把 waypoint 最短路径图当作稀疏 tour 合法边图；不引入 LP/MIP 求解作为已有最优标签数据的必要环节；不把 sliding-TSP 等更强局部求解偷偷加到 GP 一侧。任何后续更换 LS 的研究需要重新定义底座，并让相同对照同步更换，不混入本轮核心结论。

## 10.4 实现模块与来源边界

建议模块如下。文件名为拟建项目的组织建议，**不是声称这些文件已实现**。

```text
gp-faco/
  cpu_reference/
    faco_relocation_reference.cpp    # 从锁定作者实现提取，保留许可与差异说明
  include/
    instance_view.hpp               # 原目标距离、候选行和标签隔离
    faco_config.hpp                 # 原生语义/固定适配的显式选项
    gp_program.hpp                 # 七算子、十二终端、后缀码与栈边界
  cuda/
    faco_construct.cu              # 起点接口、roulette、visited、原生 MNE
    checklist_two_opt.cu           # 相同 LS 的并行评估与安全提交
    pheromone.cu                   # 稀疏更新、默认值和重启事务
    features.cu                    # 可重用的 colony/reference/region 特征
    gp_score.cu                    # 同树、32 动作、warp 归约
    colony_step.cu                 # 批次边界与 deadline-safe 记录
  training/
    generational_gp.py             # 标准 GP，不增加多保真或代理模型
    evaluate_population.py         # 固定 GPU 批次配置和外部 fitness
  adapters/
    candidate_import.py            # 外部候选输出验证，不硬绑定 LKH 内部函数
    cuopt_baseline.py              # 可选外部基线，锁定包版本后测试
  provenance/
    sources.lock.json              # 仓库 commit、文件 blob、许可与复用状态
    THIRD_PARTY_NOTICES.md
  tests/
    test_native_mne.cpp
    test_restart_state.cpp
    test_candidate_order.cpp
    test_tour_and_gain.cpp
    test_cpu_gpu_policy.cpp
```

核心 CUDA 库只依赖必要的 CUDA/CUB/cuRAND 与项目自己的轻量数据结构；不把完整 cuOpt、LKH 或 ACOTSP 设为强制链接依赖。若之后实际复制 Apache helper，再为该文件保留许可、来源 commit 和改动记录。

## 10.5 工程验收顺序与完成状态

| 验收点 | 必须验证的行为 | 当前状态 |
|---|---|---|
| 来源锁定 | FACO/cuOpt/GPU-MMAS 的 commit、关键文件与许可事实 | 已静态核查；见来源索引和随附 manifest |
| 原生 CPU 参考 | 正确读取距离与标签格式，显式配置运行，重现核心操作 | 待在 CPU 集群构建与运行 |
| CUDA 基础内核 | 先移植无 GP 的固定 FACO；核查 tour、增量成本、MNE 与 LS 顺序 | 待实现与测试 |
| 参考与重启 | 默认信息素、缓存、ant 状态和 epoch 记录一致切换 | 已定位风险；待单元测试 |
| GPU GP | 有限值、六槽栈、合法动作 mask、CPU/GPU 分数与选择一致性 | 方案定义完成；待实现 |
| alpha/POPMUSIC | 官方工具版本/hash、候选导出、距离排序视图、原目标一致 | 官方来源已确认；源码构建/导出未验证 |
| cuOpt 外部基线 | 稳定版本 API、闭合 TSP 语义、距离类型、返回格式、总时间 | 源码和文档已审查；未安装试跑 |
| 性能剖析 | 完整成本与峰值显存，不只测 GP 评分 kernel | 尚无实测结果，不能承诺加速比 |

本环境的 GitHub 连接器可读取公开源码，但命令行仓库下载因网络解析失败未完成；也未验证到可用的 CUDA 编译与 GPU 执行环境。因此交付物是**有源码依据的研究与实现计划**，不是已经运行的 GPU-FACO。最终部署以项目实际 CPU/GPU 集群上的构建、数值测试和剖析为准。

## 10.6 本轮对研究结论的影响

四个 RQ 不变。源码审查主要让实验处理更精确：E1 比较相同 FACO-Control 上的反馈价值；E2 比较原生 MNE/起始位置与固定重启的学习交互；E3 在相同 LS 距离视图及原问题距离下比较候选先验与出口；E4 冻结相同程序与内核迁移。许可、CUDA 同步和 API 版本属于实施约束，不包装成额外科学贡献。


# 11. 实现决策：Python/DEAP 外层，C++/CUDA 内层

本节固定语言与模块边界，不增加 research question，不改变四组主实验，也不引入多保真、代理评价或另一套优化器。代码块中的项目接口与文件名是实现规格，不表示对应 CUDA 求解器已经实现或编译通过。

## 11.1 技术选型与职责

**确定采用混合实现：Python + DEAP 管理离线 GP；C++17 + CUDA 执行 FACO 及在线程序评分；pybind11 提供粗粒度调用边界。**

| 模块 | 语言/依赖 | 具体职责 | 不承担的职责 |
|---|---|---|---|
| GP 管理 | Python、DEAP | 树表示、初始化、选择、交叉、变异、日志 | 不在线遍历 TSP，不逐动作调用 Python 函数 |
| 训练编排 | Python、NumPy | 面板抽样、worker 调度、外部 fitness、验证选择 | 不控制每次 FACO 迭代或 LS move |
| 程序导出 | Python，项目自有小模块 | DEAP 前缀树转换为有限后缀指令，验证和序列化 | 不为每棵树调用 nvcc，不求解 TSP |
| 求解控制层 | C++17 | 状态生命周期、批次启动、截止、结果封装 | 不调用 Python policy 回调 |
| 在线计算层 | CUDA、CUB/cuRAND | 特征、32 动作评分、FACO、checklist LS、记忆更新 | 不负责 GP 的交叉变异 |
| 绑定层 | pybind11 | 小程序、数据句柄与整段运行结果的交换 | 不成为逐节点/逐边通信接口 |
| CPU 语义参考 | C++17 | 与 CUDA 使用相同目标、动作和程序格式的对照 | 不将 CPU 性能当作 GPU 单组件归因 |

纯 C++ 也能实现 GP，但本项目没有理由在第一版重新实现树操作、日志与实验管理；纯 Python 的在线 FACO 又不符合所需的执行边界。选择混合实现依据的是职责，而不是未经测量的加速倍数。预期瓶颈主要在完整求解评价，仍须通过第 5.6 节的剖析核实。

第一版不强制引入 PyTorch、CuPy、Triton、Ray 或完整 cuOpt。它们可以服务外部基线或已有基础设施，但不是使用 DEAP 或执行 CUDA 的前提。本项目的训练不用反向传播。GPU 资源首先投入批量 rollout，而不是投入种群选择和子树交换。

## 11.2 DEAP：使用算子库，不交出执行后端

DEAP 提供 `PrimitiveTree`、`PrimitiveSet`、树生成和遗传算子；`gp.compile()` 的公开源码将表达式转换为 Python 函数，并不生成 C++ 或 CUDA 代码。[P1, P2]

因此，使用边界是：

```text
DEAP individual
    -> 项目自己的后缀码导出器
    -> C++ 校验后的 Program
    -> CUDA 在线评分与 FACO 完整运行
    -> 原始结果返回 Python
    -> 外部 evaluator 计算 J(f)
    -> 写入 DEAP fitness
```

本方案全部终端和函数均为实值，第一版使用普通 `gp.PrimitiveSet` 即可，不需要为了一个数值类型引入多类型 grammar。终端 `b` 是值为 0/1 的实数，并不是在 GP 树中执行重启的命令。动作合法性由固定求解器检查，树只输出评分。

推荐 API 映射：

| 需求 | DEAP 组件 | 本项目约束 |
|---|---|---|
| 单目标最小化 | `base.Fitness`，`weights=(-1.0,)` | `fitness.values=(J,)`；不重复取负 |
| 个体 | `creator` + `gp.PrimitiveTree` | 每个个体恰好一棵树 |
| 初始树 | `gp.genHalfAndHalf` | 建议初始深度 1–3，演化后上限仍为 5 |
| 交叉 | `gp.cxOnePoint` | 子树交叉 |
| 变异 | `gp.mutUniform` | 第一版只用子树变异，新子树可生成新 ERC |
| 选择 | 锦标赛，大小 4 | 精确同分时按长度、稳定 hash 决胜，必要时用很小的选择封装 |
| 大小控制 | `gp.staticLimit` | depth≤5、nodes≤63，导出端再次验证 |
| 记录 | `tools.Statistics`、`tools.Logbook` | 记录本代同面板统计，不直接给跨面板原始 fitness 排名 |

DEAP 的常规 fitness 是 tuple，单目标也一样；选择通常返回对象引用，变异与交叉修改输入对象。必须克隆后再修改，不能把父代或精英意外改变。[P3]

`gp.staticLimit` 对超限后代的默认处理是随机选择一名父代来替换，而不是自动剪枝。第一版采用这一明确规则，并记录超限回退次数。初始化也单独检查合法性，不能只依赖算子装饰器。[P1]

额外的专用常数变异不是初版必需。若后续启用 `gp.mutEphemeral`，它只重新生成选定 ERC，不引入梯度常数优化或嵌套搜索。自写变异必须替换常数节点，不能随意原地修改可能被多个树引用的 terminal 对象。[P2]

## 11.3 Function/terminal 注册与稳定编号

函数和 terminal 必须使用统一的命名与编号，形成 Python 导出器和 CUDA 执行器共同遵守的程序接口。DEAP 在同一数值类型内负责组合合法性，C++ 负责实际执行格式的合法性。

| Terminal ID | Python 名称 | 对应论文符号 |
|---:|---|---|
| 0 | `elapsed` | $u$ |
| 1 | `stagnation` | $s$ |
| 2 | `return_rate` | $q$ |
| 3 | `ls_work` | $c$ |
| 4 | `restart` | $b$ |
| 5 | `mne_level` | $\kappa$ |
| 6 | `ref_gap` | $g_r$ |
| 7 | `ref_diff` | $d_r$ |
| 8 | `region_excess` | $e_R$ |
| 9 | `archive_disagreement` | $v_R$ |
| 10 | `pheromone_strength` | $p_R$ |
| 11 | `region_dispersion` | $\ell_R$ |

GP-NoFeedback 的 primitive set 可以只开放九个参数，但导出后仍使用上述全局 ID，例如删除 $s,q,c$ 后，`restart` 仍为 4，不能变成 1。否则消融程序会读取错误特征。候选动作的编码同样冻结，mask 只决定合法性，不更改编号含义。

Python 的七个 primitive 都注册为有名字的模块级函数。其主要用途是定义 grammar 和构建诊断 oracle；在线运行不调用它们。注册时不能直接用无裁剪的 `operator.add` 冒充本方案的 ADD：每一步输出裁剪到 $[-8,8]$ 是算法定义的一部分。

ERC 使用模块级的无参生成函数，在 $[-2,2]$ 采样后量化为 FP32，再将该实际值保存在 terminal 中。固定常数 $-1,0,1$ 也使用相同数值约定。常数在树生命周期内固定，不能在每次动作评分时重新抽样。不要用 lambda 定义需序列化的 ERC 工厂；DEAP 源码对此给出了可 pickling 限制的提示。[P2]

数值配置还应冻结 AQ 的 `rsqrtf` 近似语义、FP32 舍入、是否启用 FMA contraction 和 GPU 编译选项。Python/C++ 参考值与 GPU 可以按预定义误差验证，不把不同平台的 sqrt/rsqrt 实现无条件视为逐位等价。全局 `--use_fast_math` 不能悄悄影响距离与 LS 增益内核。

## 11.4 从 DEAP 前缀树到 GPU 后缀码

DEAP `PrimitiveTree` 按深度优先前缀顺序存储节点，节点元数允许恢复子树。[P2] 导出器依照固定函数表和参数表遍历，生成第 5.2 节的后缀码。第一版不需要 C++ parser、SymPy 简化、NVRTC 或逐个体编译。

例如：

```text
DEAP 可读树：SUB(MUL(return_rate, restart), MUL(ls_work, mne_level))

后缀码：
PUSH_FEATURE 2
PUSH_FEATURE 4
MUL
PUSH_FEATURE 3
PUSH_FEATURE 5
MUL
SUB
```

SUB 必须保持左右操作数顺序。后缀执行时先弹出右值，再弹出左值，执行 `left - right`；AQ 同样不能交换参数。每个算子后都按统一规则裁剪。由于逐步裁剪和浮点运算存在，导出时不交换子表达式、不做基于实数代数的任意重排。

建议程序数据结构为：

```text
Program:
    ir_version       : uint16
    numeric_spec_id  : uint16
    feature_spec_id  : uint16
    length           : uint16
    max_stack        : uint8
    opcode[63]       : uint8
    operand[63]      : uint8       # feature 或 constant 索引
    constants[<=63]  : float32
```

正式序列化规定字节序、数组长度和字段格式，不直接把带编译器 padding 的 C++ struct 原样写盘。可读 JSON 与紧凑二进制使用相同 IR；常数保存 FP32 位模式以保证恢复一致。

Python 导出端和 C++ 接收端都检查：函数白名单及元数、terminal ID、常数有限性、指令数、栈不下溢、峰值栈不超过 6，以及最终栈恰好留下一个值。未知版本或未知 opcode 直接拒绝，不作静默近似。纯常数树、单 terminal 树以及合法动作数量不足 32 的情况仍必须正确执行。

C++/CUDA 只依赖 `Program`，不依赖 DEAP 对象。将来更换外层 GP 库不会要求重写 FACO；训练结束也可以把同一个程序交给独立 C++ CLI 部署，而不再安装 DEAP。

## 11.5 Python↔C++：一次调用完成整段评价

采用持久的 `Engine` 和只读实例数据句柄。下面是接口契约示意，不是现有已实现 API：

```python
engine = gp_faco_ext.Engine(device_id, solver_config)

# 不包含最优长度、最优 tour 或相关标签路径。
data_handle = engine.register_instances(
    instance_data, distance_spec, candidate_views
)

# program_ir 已由 Python 编码为连续数值数组。
# task_spec 含实例 ID、求解种子、固定 batch shape、预算和日志级别。
result = engine.evaluate(program_ir, data_handle, task_spec)

# 只有外部 evaluator 读取标签；求解器只返回成本、状态和时间摘要。
J = evaluator.macro_mean_gap(result, optimal_labels)
individual.fitness.values = (float(J),)
```

`evaluate` 内部完成动态状态初始化、全部 FACO 迭代、特征与 GP 评分、LS、重启、deadline 检查及必要同步，再返回结果。不能将其拆成 Python 循环中的 `step()`、`score()`、`two_opt()`，也不能让 C++ 每个蚁群批次回调 Python 生成动作。

一个 GPU worker 可以持久保留只读实例与已分配缓冲区，但每个 `(program, instance, solve_seed)` 任务必须重新初始化信息素、参考/档案、visited、反馈、随机计数器与 best-so-far。**复用进程和显存分配不等于复用上一程序的算法状态。**

`register_instances` 和缓存只改变实际数据搬运组织，不改变第 5.5 节的预处理费用记账。该费用采用统一协议进入有效预算，或单列“候选已给定”的结果，不能因为缓存存在就免费给某个算法提供候选结构。

结果至少包含 `task_id`、实际成本、可行性/失败状态、有效时间、计时超限、少量统一检查点与工作量摘要。主要运行不复制每只蚂蚁的中间 tour 回 Python；完整轨迹只在诊断模式开启。整数距离目标和浮点目标使用明确的结果 dtype，不默认把所有长度压成 FP32。

## 11.6 pybind11 的 GIL、数组与生命周期规则

pybind11 不会在 Python 调用 C++ 时自动释放 GIL。长时间 C++ 求解应显式释放；释放期间不能访问 Python 对象或调用 Python 回调。[P4]

绑定层顺序固定为：

```text
持有 GIL：验证参数类型/shape/连续性，建立 C++ 自有 Program/Task。
释放 GIL：运行 Engine 的纯 C++/CUDA 逻辑并等待本次结果完成。
重新持有 GIL：构建 Python 结果对象，返回 NumPy 数组或摘要。
```

第一版采用同步返回接口；不先引入 Future、跨语言异步句柄或 Python 回调。`Engine` 每个 worker 只允许一个活动评价任务，避免共享缓冲区竞态。

数据交换使用明确的连续 NumPy 数组：例如候选 ID 为 int32、程序常数为 float32，距离数据按实例目标定义。pybind11 的 `forcecast` 可以隐式转换或复制输入；关键接口采用显式 dtype/shape 验证和不隐式转换的绑定约定，避免在计时内部隐藏大规模复制。[P5]

这里的“避免不必要的 Python↔C++ 复制”不等于“CPU 数据天然就在 GPU 上”。Host-to-device 传输仍存在并须计时。若 C++ 借用 NumPy 内存，必须保证所有者在同步操作完成前存活且数据不被其他线程修改；持久实例数据优先由 Engine 明确拥有或复制到设备端。

CUDA 异步错误在适当同步点转换为有任务 ID 的异常/状态。基础设施故障按固定规则重跑同一任务；非法程序或非法 tour 不可从 fitness 平均中丢弃。任一预定义样本失败则将该个体标为失败并按冻结规则赋罚值，例如 $+\infty$；不能因少做难例而得到更好平均值。

## 11.7 多 GPU：一个协调进程，每张卡一个常驻 worker

第一版采用静态资源分配：

```text
Coordinator：Python + DEAP，不初始化 CUDA
    |
    +-- GPU worker 0：一个常驻 Python 进程 + 一个 C++ Engine
    +-- GPU worker 1：一个常驻 Python 进程 + 一个 C++ Engine
    +-- ...
```

任务消息只发送 `program_ir + task_spec` 或其稳定 ID，不发送整个 DEAP toolbox、Python 可调用 policy 或活动 CUDA 指针。每个 worker 在一个计时工作波内只运行一个程序；该程序批量求解固定形状的独立实例/seed，符合第 5.5 节。下一程序复用缓冲区，但完整重置动态状态。

显式使用 `multiprocessing.get_context("spawn")`；在 worker 启动并绑定设备后才创建 Engine/CUDA context。主入口放在 `if __name__ == "__main__":` 下，不依赖平台默认启动方式。NVIDIA 官方说明指出 fork 已初始化 CUDA 的进程会带来上下文问题，spawn 需要可序列化的 worker 定义并建立独立解释器。[P6]

不要直接按 DEAP 的通用 CPU 示例创建很多任意 GPU 进程。该示例说明了可替换 map 与 pickle 限制，并不自动提供 GPU 设备归属、持久缓冲区或隔离计时。[P7]

同一代所有任务完成后再选择下一代；结果按 `program_id/task_id` 归集，而不是按到达顺序赋 fitness。运行较快或先完成的程序没有额外后代机会。主比较使用同型号、相同资源配置的 GPU；异构设备结果分层，不能直接混合 wall-clock fitness。

跨节点优先使用现有集群作业系统划分独立演化重复和实验配置。第一版不要求再建设 Ray/Dask 服务。如果单机 GPU 数量充足，再用上述轻量 worker 调度同一代的个体。

## 11.8 标准 GP 主循环：不能忽略训练面板变化

DEAP 的 `eaSimple` 等通用算法默认只重评无效 fitness，`varAnd` 负责克隆和对实际改变的个体作失效处理。[P8] 本研究每代更换共同训练面板，因此即使一棵精英树完全没变，其旧分数也不是新面板上的分数。

**保留标准代际 GP，自己写很短的主循环；不把 `eaSimple()` 原样当作完整训练器。**

```text
Initialize population P; initialize independent RNGs and GPU workers
for g = 0,...,G-1:
    panel ← sample training instances and solver seeds, common to every f in P
    invalidate ALL fitness values in P, including unchanged elites
    export every f to validated IR
    results ← evaluate full-budget tasks on persistent workers
    assign J(f) from the same-generation results
    save current population summary and the generation winner

    if g == G-1:
        break                         # do not end with unevaluated offspring

    elites ← clone 4 best individuals under the declared tie rule
    parents ← tournament-select len(P)-4 individuals from current P
    offspring ← DEAP varAnd(parents, cxpb=0.8, mutpb=0.2)
    enforce depth/size bounds; validate exports
    P ← elites + offspring
    write atomic checkpoint for the next generation

shortlist ← deduplicated generation winners plus the final evaluated population
evaluate shortlist on the SAME fixed validation protocol
select and export a frozen program; do not inspect test outcomes for selection
```

`varAnd` 中 0.8 是成对交叉概率，0.2 是随后逐个体变异概率，两者独立，一个后代可以先交叉再变异；不是“80% 交叉、20% 变异且互斥”。这是标准算子组合，不增加一个训练层次。[P8]

普通 `HallOfFame` 不应直接按来自不同随机面板的原始训练分数维护一个跨代冠军。我们只存每代获胜程序的身份和来源，最终统一在固定验证协议上比较。固定验证集较大是模型选择，不是训练阶段对候选进行低精度淘汰。

默认不开启跨代 fitness cache。若为完全重复任务去重，key 必须包含程序及数值版本、完整实例/seed 面板、预算、batch shape、底座配置与硬件执行协议；不能只按树字符串缓存。小树编译结果可以按完整 IR hash 缓存，因为它不涉及随机求解结果。

## 11.9 Checkpoint、程序导出和可复现性

DEAP 的 checkpoint 需要在自定义循环中显式保存；官方示例还提醒同时保存 Python 与 NumPy 的随机状态。[P9] 本项目至少保存：当前/下一代阶段标志、种群树、ERC 值、演化 RNG 状态、独立面板采样 RNG 状态、已确定的求解 seed、面板 manifest、已完成任务表、配置 hash、软件/源码版本和训练成本。

演化随机数和面板采样随机数使用不同的 generator；CUDA 的抽样流另按实例、seed、colony、批次和蚂蚁划分。GPU 任务完成顺序不能消耗协调进程的演化 RNG。重启训练恢复的是配置、任务与随机状态；基于 wall-clock 截止的 GPU 运行仍可能受系统调度和末尾工作粒度影响，不承诺所有恢复运行逐位一致。

pickle 仅用于受信任的内部训练 checkpoint，不把不可信 pickle 作为公共程序输入。对外冻结程序使用已校验的 JSON/二进制 IR、可读表达式、terminal/opcode 定义版本、数值语义、动作 mask、求解配置与数据划分记录。训练过程依赖 DEAP，不意味着最终 C++ 求解器需要 DEAP 运行时。

## 11.10 模块划分与实施顺序

```text
gp-faco/
  python/gp_faco/
    primitives.py          # 七函数、十二特征名、FP32 ERC 工厂
    evolution.py           # DEAP toolbox、边界检查与标准代际循环
    program_ir.py          # 前缀树 -> 后缀码；稳定编号和序列化
    evaluate.py            # 最优标签隔离；规模宏平均 fitness
    gpu_worker.py          # spawn、设备归属、常驻 Engine 和任务协议
  cpp/include/gp_faco/
    program.hpp            # 不依赖 Python/DEAP 的 IR 契约
    engine.hpp             # 实例句柄、评价任务和结果
    problem.hpp            # 距离与候选视图
  cpp/src/
    bindings.cpp           # pybind11：validate -> release GIL -> run -> return
    engine.cpp             # 主机批次调度、deadline、状态重置
    program_cpu.cpp        # 同一 IR 的 CPU 数值参考
    faco_reference.cpp     # 既有锁定作者实现的适配对照
  cuda/
    gp_score.cu            # 同树 32 动作的小栈执行器
    features.cu            # terminal 准备与有限归约
    faco_construct.cu      # 原生 MNE、roulette、节点重定位
    checklist_two_opt.cu   # 相同邻域与顺序下的并行评估和提交
    pheromone.cu           # 候选/default 信息素、reset、缓存
  tests/
    test_program_ir.py
    test_deap_operators.py
    test_feature_ids.py
    test_program_cpu_gpu.cpp
    test_native_semantics.cpp
    test_worker_isolation.py
  configs/
  provenance/
  CMakeLists.txt
  pyproject.toml
```

上述是对第 10.4 节模块建议的具体化，不要求同时保留两套重复目录。C++ 与 CUDA 标准、编译器、DEAP/NumPy/pybind11 版本依据目标集群的实际构建固定；本轮没有安装这些依赖，也不声称某一组合已验证。

实施顺序为：首先完成可独立验证的 CPU FACO 参考及无 GP 的 CUDA 内核；同时实现很小的 DEAP→IR→CPU/GPU 评分链路；再接入一次完整 `evaluate` 调用；最后启用多 GPU worker 和正式演化。DEAP 原型可以先使用假 evaluator 验证调度，但假分数不能用于性能结论。

最重要的验收包括：SUB/AQ 左右参数正确；裁剪和 ERC 一致；移除 terminal 后 ID 不错位；超限树不会进入 GPU；同一程序多次评价前状态完全重置；更换训练面板时全部 fitness 更新；CUDA 运行中无 Python 回调；最终一代已被完整评价。

## 11.11 对研究计划的影响

四个科学 RQ、四组主实验、函数集、终端语义和单一 fitness 均不改变。本节只是将“标准 GP”落实为 DEAP 的树与算子，并用自有轻量循环管理真实 GPU 评价。新增代码的主要价值是让学习接口、计时、标签隔离与程序部署边界可检查，而不是增加一种新的学习方法。

**最终定位：DEAP 是离线算法设计工具；C++/CUDA 是被设计的求解器和在线程序执行器；连接两者的是一个小而固定的程序格式与整段运行接口。**


# 参考文献与实现依据

文内 `[R编号]` 为研究文献引用，`[S编号]` 为本轮源码与官方接口引用。以下来源用于已有算法和 CUDA 事实；本文的动作空间、特征集合、GP 配置与实验假设是拟议设计。链接以代码形式保存，便于 Markdown 独立流转。

**[R1]** Skinderowicz, R. (2022). *Improving Ant Colony Optimization Efficiency for Solving Large TSP Instances*. Applied Soft Computing, 120, 108653. DOI: `10.1016/j.asoc.2022.108653`. 作者稿：`https://arxiv.org/html/2203.02228`。

**[R2]** Skinderowicz, R. (2024). *Enhancing Focused Ant Colony Optimization for Large-Scale Traveling Salesman Problems Through Adaptive Parameter Tuning*. ICCCI 2024, LNCS 14810, 41–54. DOI: `10.1007/978-3-031-70816-9_4`. 出版页面：`https://link.springer.com/chapter/10.1007/978-3-031-70816-9_4`。

**[R3]** Tran, D. T., Vu, V. K., & Ma, Y. (2026). *Beyond Static Priors: Dynamic Neural Guidance for Large-Scale Ant Colony Optimization*. 作者稿 arXiv:2606.04039v1；文中列出 KDD 2026 DOI: `10.1145/3770855.3817893`. 来源：`https://arxiv.org/html/2606.04039v1`。本文依据其状态感知边级指导与完整轨迹训练的描述定位相关工作，不直接采用未经独立复现的性能主张。

**[R4]** NVIDIA. *CUDA Programming Guide: Programming Model*. 官方文档，访问日期 2026-09-08。`https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html`。用于 warp 执行模型与线程协作背景。

**[R5]** NVIDIA. *CUDA C++ Best Practices Guide*. 官方文档，访问日期 2026-09-08。`https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html`。用于合并访存、计时与执行效率注意事项。

**[R6]** NVIDIA. *NVRTC Documentation*. 官方文档，访问日期 2026-09-08。`https://docs.nvidia.com/cuda/nvrtc/index.html`。仅作为可选运行时编译的工程依据，不是本方案的额外学习阶段。

**[R7]** Helsgaun, K. (2000). *An Effective Implementation of the Lin–Kernighan Traveling Salesman Heuristic*. European Journal of Operational Research, 126(1), 106–130. DOI: `10.1016/S0377-2217(99)00284-2`。用于 LKH 与 alpha-nearness 候选结构背景。

**[R8]** Taillard, É. D., & Helsgaun, K. (2019). *POPMUSIC for the Travelling Salesman Problem*. European Journal of Operational Research, 272(2), 420–429. DOI: `10.1016/j.ejor.2018.06.039`. 作者所在机构页面：`https://forskning.ruc.dk/en/publications/popmusic-for-the-travelling-salesman-problem/`。

**[R9]** Reinelt, G. *TSPLIB95: Questions and Answers*. 官方说明：`https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSPFAQ.html`。用于距离定义、最优值和最优 tour 可用性说明；项目已有标签仍需与实际距离实现逐实例匹配。


## 源码与官方接口索引

`[S编号]` 引用的是本轮核查的一手源码或官方接口。GitHub 链接使用固定 commit；路径中的函数名是定位符，不意味着其可作为稳定公共 API 单独调用。核查日期：2026-09-08。

**[S1]** RSkinderowicz. *ACO-TSP-Adaptive-Tuning*，commit `a904e6a8786d48593ef1cac975edef5ed8920af3`（2025-02-25）。仓库：`https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning/tree/a904e6a8786d48593ef1cac975edef5ed8920af3`。MIT 许可：`https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning/blob/a904e6a8786d48593ef1cac975edef5ed8920af3/LICENSE`。

**[S2]** 同上，`src/faco.cpp`。核查 `select_next_node`、`Route`、`run_mfaco` 及 `run_faco_apt` 的关键控制路径；`Route::two_opt_nn` 的候选提前退出见约 1180–1310 行，构造与 MNE 见约 1500–1640 行，源解与强化见约 1640–1720 行，adaptive 构造见约 1830–2050 行。`https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning/blob/a904e6a8786d48593ef1cac975edef5ed8920af3/src/faco.cpp`。核查文件 blob：`a446e48ef4e4f469bf67e4303325a287dca594f5`。

**[S3]** 同上，`src/progargs.h`。`https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning/blob/a904e6a8786d48593ef1cac975edef5ed8920af3/src/progargs.h`。核查默认候选数、保留率、参考概率、个体记忆和局部更新开关；不把这些默认值等同于已调优实验配置。

**[S4]** 同上，`src/pheromone.h`，`CandListPheromone`。`https://github.com/RSkinderowicz/ACO-TSP-Adaptive-Tuning/blob/a904e6a8786d48593ef1cac975edef5ed8920af3/src/pheromone.h`。核查文件 blob：`7ed00133a56ffdb3359cfc4bc664413e920ca3a8`。用于候选条目、默认值、蒸发、部分强化和 `set_all_trails` 的实际行为。

**[S5]** RSkinderowicz. *GPU-based-MMAS*，commit `8aac0556177fdc8e022c98c56886e9b8baf99869`（2020-01-09）。`src/mmas.cu`：`https://github.com/RSkinderowicz/GPU-based-MMAS/blob/8aac0556177fdc8e022c98c56886e9b8baf99869/src/mmas.cu`。README：`https://github.com/RSkinderowicz/GPU-based-MMAS/blob/8aac0556177fdc8e022c98c56886e9b8baf99869/readme.md`。本轮用于 warp scan/reduce/vote 和旧版实现布局的参考；在已检查根目录与文件头中未确认项目级授权，暂不复制。

**[S6]** NVIDIA. *cuOpt*，commit `0cccfd3e426f391feaadc2afd1f9ead1533ab2ac`（2026-09-03），`cpp/src/routing/local_search/two_opt.cu`。`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/cpp/src/routing/local_search/two_opt.cu`。核查文件 blob：`abe6e8a928e7fef8aeb927f542c98d3fe5cb45b6`。用于 move 查找/提交、受影响节点及 routing 内部依赖分析。

**[S7]** 同上，`cpp/src/routing/route/tsp_route.cuh`。`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/cpp/src/routing/route/tsp_route.cuh`。核查 pred/succ 视图、RMM/RAFT 依赖和共享内存大小计算。

**[S8]** 同上，`cpp/src/routing/cuda_graph.cuh`。`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/cpp/src/routing/cuda_graph.cuh`。核查 capture、graph update、重新实例化与非线程安全提示。

**[S9]** 同上，`cpp/src/routing/util_kernels/top_k.cuh`。`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/cpp/src/routing/util_kernels/top_k.cuh`。核查 CUB block 原语、模板约束与依赖；不作为 alpha-nearness 生成器。

**[S10]** NVIDIA. *cuOpt Routing API / Features / TSP Batch Example*。公开 API：`https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/routing/routing-api.html`；功能说明：`https://docs.nvidia.com/cuopt/user-guide/latest/routing-features.html`；固定源码示例：`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/docs/cuopt/source/cuopt-python/routing/examples/tsp_batch_example.py`。用于 BatchSolve、DataModel 和 WaypointMatrix 的接口语义，不构成 10K 性能证据。

**[S11]** NVIDIA. *cuOpt License, README and Installation*。固定许可：`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/LICENSE`；README：`https://github.com/NVIDIA/cuopt/blob/0cccfd3e426f391feaadc2afd1f9ead1533ab2ac/README.md`；官方安装：`https://docs.nvidia.com/cuopt/user-guide/latest/install.html`。核查 Apache-2.0、内部 C++ API 稳定性提示，以及访问时列明的稳定版 26.08 / nightly 26.10。包构建和源码 commit 分别锁定。

**[S12]** Helsgaun, K. *LKH / LKH-3 official pages*。`https://webhotel4.ruc.dk/~keld/research/LKH/`；`https://webhotel4.ruc.dk/~keld/research/LKH-3/`。官方搜索结果确认 POPMUSIC 候选功能和学术/非商业使用条款；本轮页面直接读取超时，未取得并构建官方源码包，因此不提供未经验证的内部导出函数名或 archive hash。

**[S13]** Dorigo / IRIDIA. *ACO Public Software — ACOTSP.V1.03.tgz*。`https://iridia.ulb.ac.be/~mdorigo/ACO/aco-code/public-software.html`。官方列明 AS、ACS、MMAS 等对称 TSP 实现及 GPL 许可；本轮未构建该包。

---

**方案概要：四个科学问题、四组主实验；一棵 GP 树、七个函数、十二个终端、一个 fitness；GPU 加速服务于大规模评价与求解，不扩张论文的研究问题。**


## Python/DEAP 与 C++/CUDA 实现来源

以下均为官方文档或公开源代码，核查日期为 2026-09-08。文档站展示的版本号不等于本项目已安装或通过构建的依赖版本；实际部署另行锁定版本与构建记录。

**[P1]** DEAP. *Genetic Programming API*. `https://deap.readthedocs.io/en/master/api/gp.html`。用于树表示、primitive set、交叉变异和 staticLimit 的接口定义。

**[P2]** DEAP. *deap.gp source*. `https://deap.readthedocs.io/en/master/_modules/deap/gp.html`。核对 PrimitiveTree、gp.compile、MetaEphemeral 和 mutEphemeral 的实际行为。

**[P3]** DEAP. *Operators and Algorithms*. `https://deap.readthedocs.io/en/master/tutorials/basic/part2.html`。用于 fitness tuple、选择返回引用与变异前克隆的约定。

**[P4]** pybind11. *Miscellaneous — Global Interpreter Lock*. `https://pybind11.readthedocs.io/en/stable/advanced/misc.html`。用于显式释放 GIL 和禁止无 GIL 访问 Python 对象的边界。

**[P5]** pybind11. *NumPy* 与 *Functions*. `https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html`；`https://pybind11.readthedocs.io/en/stable/advanced/functions.html`。用于 array dtype、连续性、forcecast 和 noconvert 约定。

**[P6]** NVIDIA DALI. *Parallel External Source — Spawn*. `https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/parallel_external_source.html`。NVIDIA nvImageCodec. *PyTorch DataLoader integration*. `https://docs.nvidia.com/cuda/nvimagecodec/samples/torch_dataloader.html`。仅引用官方关于 CUDA context 与进程启动的说明；本项目不因此依赖 DALI、nvImageCodec 或 PyTorch。

**[P7]** DEAP. *Using Multiple Processors*. `https://deap.readthedocs.io/en/master/tutorials/basic/part4.html`。用于区分通用 map/pickle 示例与本项目的持久 GPU worker。

**[P8]** DEAP. *Algorithms — eaSimple and varAnd*. `https://deap.readthedocs.io/en/master/api/algo.html`。用于默认 invalid-fitness 评价、varAnd 克隆与先交叉后独立变异的语义。

**[P9]** DEAP. *Checkpointing*. `https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html`。用于显式保存演化状态与随机状态；本文额外规定任务、数值语义和 wall-clock 可重复性边界。
