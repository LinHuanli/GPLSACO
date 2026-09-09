# E3 常驻worker、图身份及四条件开发训练验收

本次在独立E3工作树接通冻结图目录、常驻worker、外部评分及完整DEAP训练/验证/恢复。四条件工程运行全部完成；正式E3训练规模、Static选择及统计仍需冻结并执行。主目录十个E1训练和完整基线调参继续使用其原源码与原生二进制，没有迁移或缩减预算。

机器记录见[graph_worker_results.json](../../../../results/v1/graph_worker_results.json)，事前契约见[planning/19](../planning/19_e3_graph_worker_contract.md)。原生二进制仍为完整Escape验收的`fdd64ce88f8e58b144c95cc97805fe213fc96bffeae7674d51be981aa4d701bd`，本次没有修改C++/CUDA，也没有重新安装环境。

## 图身份与执行边界

worker协议版本5固定先验ALPHA/POPMUSIC及Hard/Escape模式，绑定图目录的路径和完整SHA256。目录包含同一32个开发实例的64份匹配v2图；摘要为`242b47dd677006de4d8a32925dc4166ddee8d37561a2e4ebd9d47e7a13a2e689`。每次准备和求解身份同时记录实例坐标、先验、完整图摘要及实际E0边数。原生实例键继续使用坐标摘要前64位，使四条件的原始随机流保持配对。

新实例注册前检查缓存文件完整字节和图摘要；原生仅接收坐标与六个图字段。已注册实例复用原准备资料，不在每个个体任务中传输完整候选图。外部评分独立重算路线成本，并检查实际模式、图版本/边数、Escape版本/容量、完整FE账目、Hard的E0可行性及零FE共同初解。图worker明确拒绝秒数预算。

所有工作树的spawn子进程使用主Git仓库下同一个UUID锁。真实验收中，同一GPU的第二个锁句柄均无法抢占活跃worker的锁。GPU启动前及任务边界核查占用；不同运行保存实际host、UUID、型号和driver。实际设备为`cuda05.ecs.vuw.ac.nz`上的两张RTX PRO 5000 Blackwell，driver `610.43.02`：

- `GPU-d2f06ac8-94bb-9744-3791-f0d636d29978`：worker矩阵、alpha-Hard/Escape开发训练。
- `GPU-f92178e7-9654-8706-adb7-bc338a8ea311`：Python GPU回归、POPMUSIC-Hard/Escape开发训练。

## 检查与保留的失败

首次真实worker矩阵完成alpha-Hard的10次求解后，下一条件的启动因GPU空闲读数未达到阈值退出；没有已启动求解的FE失败。原始日志、12份准备/求解记录和当时源码保留在`artifacts/graph-worker/matrix-v1`、对应log/resources及`source-v1.tar`，不能将其计为完整矩阵通过。

随后增加持UUID锁、尚未创建CUDA上下文时的空闲读数等待。每次仍重查计算进程、型号与driver；有外来进程立即拒绝启动。此等待不产生算法时间上限或FE扣费。针对性检查覆盖繁忙读数后转为空闲，以及等待时出现外来进程。修正后的矩阵实际exit 0；原Future观察超时仍等待同一次提交。

最终Python GPU回归为**156 passed、8 skipped**；8项是未重新启用的既有LKH原生导出检查，其此前证据仍保留。另以19项针对性检查覆盖图身份、外部可行性、缓存键、设备等待及容量重建；其中最后补充的Hard/Escape容量边界使用假Engine核验Python事务，真实32-colony运行阶段没有触发容量替换。C++/CUDA未改，沿用相同原生文件身份的17项CTest和三类零错误CUDA检查，不重复计作本次新增结果。

## 完整常驻worker矩阵

每个条件在N=500/1000上执行16实例×2个solve seed，固定32 colonies×32 ants。调用序列为GP零FE、GP 128 FE、Static 64 FE、GP 128 FE原生end_to_end、GP 128 FE cached，预算均指每colony的完整search-tour次数。

四条件共8个准备任务、40次求解、114,688 FE和1,280条返回路线，独立核验480,000条Hard路线边；最大成本重算误差`3.552713678800501e-14`。16组同PID重放的路线、成本、完整控制状态、工作计数及Escape计数精确一致，Hard/Escape的设备容量一致。真实矩阵CLI为27.28秒、最大CPU RSS 350,596 KiB；这是资源记录，不是时间预算。

累计观察到29,925条例外边登记及1,632次旧视图重激活；这些工作计数不能推出正式质量收益。完整逐移动许可仍由已有原生oracle证明，单凭最终tour不能证明整个Escape轨迹合法。

## 四条件独立开发训练和恢复

各条件单独初始化GP与训练运行，使用8个体×3代、每colony 256 FE。复用32个已核验开发实例，每规模前8个训练、后8个验证，4个solve seed组成完整32-colony形状；验证seed固定17/29/41/53。这个8实例×4seed面板用于工程通路检查，正式16实例×2seed面板须另行冻结。四条件演化seed均为1103、面板seed为73001，使用相同grammar与完整代际更新。

| 条件 | 训练＋验证调用 | 验证候选 | 总FE | 返回路线 | CLI秒数 |
|---|---:|---:|---:|---:|---:|
| alpha-Hard | 48＋12 | 6 | 491,520 | 1,920 | 41.22 |
| alpha-Escape | 48＋12 | 6 | 491,520 | 1,920 | 44.15 |
| POPMUSIC-Hard | 48＋12 | 6 | 491,520 | 1,920 | 40.60 |
| POPMUSIC-Escape | 48＋12 | 6 | 491,520 | 1,920 | 45.93 |

合计240次求解、1,966,080 FE及7,680条路线，全部独立审计通过，无求解或基础设施失败。每条件实际完成第5任务后暂停，恢复保留原7份记录及其准备资源；分别创建初始训练、恢复训练、验证三个不同PID的worker。原始返回先独立落盘，再查询GPU资源及评分。

审计从原始seed重放每代面板/solve seed、全部GP IR、fitness、冠军、4次交叉/4次变异、最终种群、6个候选的完整固定验证及选择结果。四运行全部准备/求解文件、独立原始返回、提交前后GPU观察均精确覆盖，无未知或已记录外来GPU进程。最大路线成本重算误差为`4.618527782440651e-14`。

## 剩余工作与复现

下一步先准备正式全部采样面板与完整验证成员的两种先验及匹配图，独立审计并冻结目录，再登记正式四条件GP训练、Static选择与配对统计。当前工程选出的程序不作为正式E3选定程序，不放行TEST。原生end_to_end仅重做Engine内部准备，外部LKH和图匹配成本另列；补充Hard省去无用备用准备的成本仍待完成。

入口分别为`prepare_graph_catalog.py`、`check_graph_worker.py`及`check_graph_training.py`，均要求显式项目内图目录/输出及实际GPU UUID。独立审计使用`audit_graph_worker.py`或`summarize_training.py`；后者新增显式数据库、数据根、入口源码与检查日志参数，仍保留旧开发审计默认入口。项目共用主目录`.venv`，各独立运行的完整参数、hash与原始记录见`artifacts/graph-worker`。
