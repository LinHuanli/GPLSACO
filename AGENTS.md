# GPLSACO 开发约定

- 所有代码、构建、虚拟环境、临时文件和运行输出均位于本目录；外部 `../Datasets` 与 `../references` 只读。
- 以 `docs/design/GP_ACO_TSP_Research_Proposal_v4.md` 为科学范围；实现细化见 `docs/README.md`。不得把准备工作或 smoke test 写成 E1–E4 的完成证据。
- 按用户2026-09-09最新指示，后续只使用实时空闲的RTX A5000；用gpu-free发现并核查目标设备。旧的“任意空闲型号”授权已被收紧。
- 不再进行文件、代码版本、检查点和实验管理记录的哈希计算或完整性哈希校验；训练、验证、测试、恢复均适用。采用普通编号、参数记录和必要的直接数据检查。算法内部若哈希适合性能与正确性则可使用，不为统一禁用而改变算法。
- 按用户2026-09-08补充，进化和ACO主实验按evaluation次数终止，尽量不设时间上限；最新协议见 `docs/experiments/protocol_v2.md`，优先于旧wall-clock主协议。旧计时记录保留为工程证据。
- 按用户2026-09-09补充，FACO基础参数须核对所继承版本的论文实验设置，不能以源码默认值或GPU工程预设代替论文依据。共同对照保持基础参数一致；偏离论文的设置注明目的和验证。蚂蚁数、初始化或LS变化另建实验版本，保留旧记录但不拼接不同配置的fitness；当前参数和依据见 `docs/design/implementation_v2.md`；旧 32 蚂蚁运行只读保留，不向新版本续接 fitness。
- Python 管理离线 GP；C++17/CUDA 执行在线求解。新增核心逻辑写中文注释，保留第三方原有版权。
- 求解器输入不包含最优值、最优 tour 或标签文件路径；评分位于外部 evaluator。
- 使用项目 `.venv`；设置 `TMPDIR=$PWD/.tmp`、`PIP_CACHE_DIR=$PWD/.cache/pip`。只版本化源码、配置、manifest 和精简报告，不提交原始数据、二进制或完整运行日志。
- 数值语义、候选限制、重启事务、deadline 和数据划分改变需更新对应协议与有针对性的验证。正式测试揭盲后不改冻结方法。
- 完成一项可复核工作后更新 `docs/reports/progress.md`，注明证据、未完成项与下一步。
- 用户2026-09-10批准的当前轮次为修复后的 `joint_single` 与 `conditional_three` 对照，各3个GP seed（1103、2207、3313），每次128个体×10个完整评价代。旧50代轮次已经暂停并保留检查点。新入口 `scripts/run_representation_pilot.py`，协议 `docs/experiments/representation_pilot_v3.md`。不得自动继续50代或进入正式val/test。
- 用户进一步明确所有服务器的空闲 A5000、所有独立实验任务都可并行，三个进化 seed 分卡同时运行。开发标定、baseline、验证和测试也应分片调度。CPU 原生对照可跨主机，每机最多一个8线程任务且必须保持24条原生初始路线，不因迁移改变参数。
- 当前只训练TSP500；每代16实例×1个ACO seed×128蚂蚁×1000迭代，六次进化共用冻结前10面板。g5/g10监控16开发实例×seed17×1000迭代；仅g10冠军在既有H校准的32开发实例×3seeds×5000迭代上比较。目录`artifacts/v3/gp-representation-pilot`，构建`build/v3-gp-representation-exact`。行为描述在64个冻结训练情境上直接调用同一CUDA评分器，避免CPU/GPU的AQ近并列舍入差异；繁殖在短GPU任务中执行，没有新增ACO FE。
