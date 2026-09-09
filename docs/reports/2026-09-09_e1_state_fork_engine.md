# E1 状态分叉接口与真实面板验收

完整状态捕获、持久化、恢复及 MNE 2/16 配对干预已在独立 `work/e1-state-fork` 分支实现。快照保存 40 个持久算法缓冲和主机已提交 incumbent，保持源 FE progress 分母与批次编号；每个分支恢复原状态，第一批使用同一参考解和已固定区域，随后执行同一预定继续策略。预算按额外 FE 计，不设算法墙钟上限。

在空闲 Quadro RTX 6000 上，四切点/32 批完整状态对照、12 组配对与单 colony 提取、4 项 Python 检查、12 项既有 CTest，以及 memcheck/racecheck/synccheck/initcheck 均通过。真实 500/1K、各16开发实例×2seed的完整32×32工程矩阵共40调用、156,160 FE、1,156条返回路线，经原CLI exit 0后独立审计通过，最大成本重算误差`6.40e-14`，资源偏差与labels查询均为0。原完整轨迹与切点继续、A–B–A重放和单colony提取均一致。

真实矩阵使用工程预算256 FE/colony、切点128 FE、继续128 FE，整次CLI为25.79秒，含注册及快照I/O，不与正式求解速度合并。其用途是核验接口，正式q/c分层、pilot/采样成员、来源控制器、快照数量、分叉seed与效应统计仍须冻结和执行。E1–E4完整研究与正式TEST尚未完成。

代码固定在提交 `4206c55962904d7a68b6b73e9a85e33af29cf126`，native SHA256为`ca70d9c49435ec865255e9cd5f229daa6d6740504a33218b8b9a41dea8235f69`。实现见[分叉工程入口](https://github.com/LinHuanli/GPLSACO/blob/4206c55962904d7a68b6b73e9a85e33af29cf126/scripts/check_e1_state_fork.py)与[实现契约](https://github.com/LinHuanli/GPLSACO/blob/4206c55962904d7a68b6b73e9a85e33af29cf126/docs/planning/21_e1_state_fork_engine.md)。机器证据见[e1_state_fork_engine_results.json](e1_state_fork_engine_results.json)。完整原始返回和四份快照归档在`artifacts/runtimes/4206c55962904d7a68b6b73e9a85e33af29cf126/`，真实证据包SHA256为`5778480a401769579cc4f0631a34e8a0a2cae5c384b8e4c106f9c7b328c7748c`。main及E3正在运行的源码与原生底座保持各自冻结版本。

UTC 20:43:31 的[活动观察](formal_runs_observation_20260909_204331.json)为E1 27,740调用、基线2,780调用，E3 GP 14,183调用、Static3,961调用，均无已记账失败求解。34个既有工具句柄仍在运行，其中cuda10两个E3等待器尚在等待原GPU空闲，另一个GP由原Static队列接续。观察不替代完整CLI终态和选择审计。
