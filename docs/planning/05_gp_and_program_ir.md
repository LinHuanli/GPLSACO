# GP、稳定程序格式与演化协议

**最新主协议使用评价次数。** feature_spec_id=2将终端0命名为`progress`，取已完成search-tour evaluations/限额；其余11个终端、opcode和数值规范保持原编号。详见[次数协议](12_evaluation_count_protocol.md)与`configs/program_spec_v2.json`。以下v1的elapsed定义属于旧时间入口，两者不得静默混用。

## 1. 程序契约 v1

固定 global feature IDs：0 elapsed、1 stagnation、2 return_rate、3 ls_work、4 restart、5 mne_level、6 ref_gap、7 ref_diff、8 region_excess、9 archive_disagreement、10 pheromone_strength、11 region_dispersion。NoFeedback 删除 1/2/3 的注册入口，但其余编号不移动。城市号、区域号、原始 n、最优 gap 均不进入 grammar。

opcode 固定：0 PUSH_FEATURE、1 PUSH_CONSTANT、2 ADD、3 SUB、4 MUL、5 MIN、6 MAX、7 ABS、8 AQ。SUB/AQ 弹出右值再左值。函数每步裁剪到 [-8,8]；输入特征须有限且符合 feature spec；ERC 在 [-2,2] 抽取后量化 FP32，另允许固定 -1/0/1，执行期间不再采样。

IR 保存版本、numeric spec、feature spec、操作码、操作数、常数 FP32 位模式。可读 JSON 是第一版公共格式；二进制待需要时定义显式 little-endian 编码，不直接 dump 带 padding 的 struct。序列化后重新校验并稳定 hash。大小≤63、树深≤5（根为0）、栈峰值≤6、最终栈恰好一值；**栈≤6 不足以证明树深≤5**，接收端也重建深度核验。未知版本、非有限常数、越界 terminal、错元数均拒绝。

无代数化简和子树重排，因为中间裁剪/FP32 使实数恒等式未必成立。恒定树、单 terminal、裁剪饱和、非法动作和真实数值平局都进入验收。合法动作取最大分数，容差内按最小 ID；每次先计算真实最大再筛平局，避免 pairwise epsilon comparator 不传递。

## 2. DEAP 边界

模块级七个 primitive 实现有界 FP32 oracle；模块级 ERC 工厂支持 pickle。使用 PrimitiveSet/PrimitiveTree、genHalfAndHalf、cxOnePoint、mutUniform 和 staticLimit。`gp.compile` 只做 Python 诊断，不作为 CUDA 编译器。依据 [DEAP GP 官方 API](https://deap.readthedocs.io/en/master/api/gp.html)，安装版本另在 lock 文件登记并核对实际源码。

## 3. 主演化配置（v4 起始候选）

种群128、50代、初始深度1–3、最大深度5/节点63、锦标赛4、精英4。varAnd 先成对以0.8交叉，后逐个体以0.2独立变异；两者不互斥。选出的引用必须 clone 后操作。超限按照 staticLimit 父代回退，同时记录次数，导出端再检查。

每代抽取同一 500/1K 面板，每规模16实例×2 solve seeds。**包括精英和重复树的所有个体均失效并重评**；不沿用不同面板的旧 fitness，不默认缓存完整求解分数。每次评价都是相同固定精度与预算。

外部 fitness 为两规模 gap 均值的等权宏平均，每个实例先平均 solve seeds。一个预定义任务失败则该个体标记失败并赋 +infinity；基础设施暂态只按固定次数重跑同一 task，不能重抽难例。选择先按 fitness，冻结平局规则后按程序长度/hash；训练成本、性能统计分开。

## 4. 代际与验证

每代 complete-evaluation barrier 后才选择。保存当代冠军身份，不以跨代不同训练面板分数维护全局冠军。最后一代必须已评价，不能留下未评后代。验证 shortlist 为去重后的各代冠军＋最终种群，统一固定验证集/预算/种子，按验证 gap 和长度平局选一程序。

至少五个独立 evolution seeds，每个各选一程序，全部带入测试；不得按 test 成绩挑最佳 seed。NoFeedback、E2 两个单因素以及 E3 各图条件需独立重训，并使用相同资源配置。

## 5. Worker 与恢复

协调进程不初始化 CUDA；使用显式 spawn，worker 绑定 GPU UUID 后创建 Engine。静态数据可复用，动态状态每任务清零。任务 ID 包含 program IR hash、配置/实例/seed/预算/batch/hardware protocol；异构 GPU 结果不混入同一训练 fitness。

checkpoint 原子落盘：阶段、当前/下一代、种群和 ERC、演化 RNG、独立 panel RNG、已定 solve seeds、任务列表和完成表、所有配置/source/hash、GPU小时。公共程序用验证 JSON；pickle 仅用于本地可信 checkpoint。恢复保持任务身份，不承诺 wall-clock 截止轨迹逐位相同。

## 6. 验收样例

固定例 `SUB(MUL(return_rate,restart),MUL(ls_work,mne_level))` 的 postfix 要保序；随机合法深度≤5树跨 Python/C++/CUDA 比分与 action；删反馈后 restart 仍为4；未知 IR/超深 ABS 链/NaN/ERC越界被拒绝；final generation 已评价；更换 panel 后 elites 也实际运行；一次 worker 连续 A-B-A 证明动态状态隔离。

## 7. 首版worker与外部fitness实现

`worker.py` 已用显式spawn执行单GPU常驻Engine；协调进程禁止导入CUDA扩展。任务ID包含评价位置，因此相同IR的不同个体仍会分别执行；Future等待超时保留原任务，活动任务期间拒绝新提交。固定GPU/driver/host、二进制和Python源码hash、batch shape及设置进入协议身份。

输入只携带冻结的无标签Instance和Program。静态登记超容量时在任务边界重建该规模Engine，输出实际登记费、engine generation和worker时间。当前准备费仍为开发测量，正式表尚未冻结。完整有序坐标hash用于任务和实例key身份，split防泄漏继续由已发布点集分组负责。

`fitness.py` 独立核验返回身份、路线、成本与截止时间，拒绝遗漏/重复/混用程序或硬件；任一预定面板失败使个体fitness为+infinity，数据标签错误直接报错。聚合先实例内seed、再规模内实例、最后两规模等权。实际spawn、失败/容量边界、64条开发路线和73项GPU Python回归证据见 [worker报告](../reports/2026-09-08_worker.md)。

以上没有实现代际循环、精英实评、最终代/验证选择、调度重试或checkpoint恢复；这些仍属于G3下一项，不能以单次两规模fitness返回有限值代替。
