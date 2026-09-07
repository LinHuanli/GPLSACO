# GPLSACO 开发约定

- 所有代码、构建、虚拟环境、临时文件和运行输出均位于本目录；外部 `../Datasets` 与 `../references` 只读。
- 以 `docs/design/GP_ACO_TSP_Research_Proposal_v4.md` 为科学范围；实现细化见 `docs/README.md`。不得把准备工作或 smoke test 写成 E1–E4 的完成证据。
- Python 管理离线 GP；C++17/CUDA 执行在线求解。新增核心逻辑写中文注释，保留第三方原有版权。
- 求解器输入不包含最优值、最优 tour 或标签文件路径；评分位于外部 evaluator。
- 使用项目 `.venv`；设置 `TMPDIR=$PWD/.tmp`、`PIP_CACHE_DIR=$PWD/.cache/pip`。只版本化源码、配置、manifest 和精简报告，不提交原始数据、二进制或完整运行日志。
- 正式 GPU 任务启动前核查目标 UUID 的实时 compute processes；固定同型号卡和 batch shape。繁忙时换空闲设备或保留任务等待，不中断他人任务。
- 数值语义、候选限制、重启事务、deadline 和数据划分改变需更新对应协议与有针对性的验证。正式测试揭盲后不改冻结方法。
- 完成一项可复核工作后更新 `docs/reports/progress.md`，注明证据、未完成项与下一步。
