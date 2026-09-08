# E1 完整状态分叉接口的开发验收

独立 `work/e1-state-fork` 工作树已实现按完整批次捕获、持久化、读回与继续的原生次数接口。快照保留全部 40 个持久算法缓冲、主机已提交 incumbent、原批次编号、源 progress 分母、形状、配置和实例/seed。每个分支完整恢复，第一批固定快照区域和参考解，仅 MNE 2/16 不同，随后使用同一个预定继续策略。

在实时空闲的 cuda17 / Quadro RTX 6000 / `GPU-3d5546e7-cd11-ceef-40ce-6010c607db67` 上，四切点保存/重建与未切断轨迹共 32 批对照通过；12 组配对干预、A–B–A 重放及 12 组单 colony 提取通过。非法快照、配置/坐标不符、FE 越界与非整批输入被拒绝；100 ms 诊断延迟不改变结果。C++ CLI exit 0，1.18 秒，示例快照 299,052 字节。

既有 12 项 CTest 全部通过，CLI exit 0，20.28 秒；新增 Python 文件读回、重建、配对重放与非法绑定输入四项检查通过（0.87 秒）。内存、竞争和同步三类 compute-sanitizer 检查均 exit 0，0 错误/竞争；保留全部 `.tmp/state-fork-*`、`.tmp/pytest-state-fork-v1.log` 及构建日志。CUDA 使用独立 `build/state-fork`，sm75，原 main/E3 冻结二进制和源码未改。

快照容器有版本、完整长度与内容校验，设备缓冲与本 native ABI 绑定；正式持久化 manifest 还须固定 native/source SHA256，不宣称跨版本可移植。feature-major 特征提取按分面复制，不能直接按 colony 连续切片；主机 incumbent 单独保存，不能在恢复时依据设备近似成本重新选择。

[真实开发验收入口](../../scripts/check_e1_state_fork.py)已完成500/1K、各16实例×2seed的完整32×32形状检查。固定工程预算为256 FE/colony，切点128 FE，继续128 FE；四个来源/规模组合共40次实际调用、156,160 FE、1,156条返回路线。原CLI 29578与独立审计CLI 2124均exit 0，独立重算最大误差`6.40e-14`，资源偏差0、labels查询0。源轨迹前后缀的FE和工作量相加一致，四组完整恢复与A–B–A重放、单colony对应成员均一致。

实际运行设备为bordeaux / Quadro RTX 6000 / `GPU-cadcb7cc-937d-9f8e-6a29-efb7ea30bfd9`，启动前由gpu-free及具体UUID查询确认空闲。整个工程CLI为25.79秒、最大RSS736,988 KiB；时间包括注册、序列化与独立状态复制，不与主研究求解速度合并。另在cuda17完成initcheck，exit 0、0错误，四类CUDA工具均已有实际证据。脚本先单独保存原返回，再观察GPU资源，提交前等待空闲，不使用墙钟截止。

源提交`4206c55962904d7a68b6b73e9a85e33af29cf126`、native SHA256 `ca70d9c49435ec865255e9cd5f229daa6d6740504a33218b8b9a41dea8235f69`已封存。四个完整快照分别为33,880,428字节（500）和67,224,428字节（1K）；原始矩阵、快照、源代码、二进制和工具记录保留在main的`artifacts/runtimes/4206c55962904d7a68b6b73e9a85e33af29cf126/`。真实证据包SHA256为`5778480a401769579cc4f0631a34e8a0a2cae5c384b8e4c106f9c7b328c7748c`，机器报告见[e1_state_fork_engine_results.json](e1_state_fork_engine_results.json)。此为工程矩阵，不是正式机制采样或MNE收益结论。

正式机制实验仍须冻结开发pilot与采样成员、q/c阈值、来源控制器、每层快照数、分叉seeds、继续策略和缺层处理，再采样与做按基础实例聚合的统计。E1主训练、基线、E2–E4与正式TEST保持未完成。实现契约见[planning/21](../planning/21_e1_state_fork_engine.md)。
