# 执行顺序、环境、GPU 与版本管理

## 1. 工作空间与环境

项目根固定为 `/vol/grid-solar/sgeusers/linbocheng/SparseACO/GPLSACO`。`.venv`、pip cache、临时文件、编译输出、远程共享目录和日志全部位于根下。外部参考目录不运行make，不写offsets；构建时指定项目内build，必要时复制到忽略的 `.deps`。

Python依赖限NumPy、DEAP、pybind11及测试/代码检查工具；实际版本写入 `provenance/python.lock.txt`。CUDA用可用系统toolkit，host compiler选择经实测兼容版本。C++17、编译选项、SM目标、driver、GPU与命令写环境报告；不根据“安装成功”宣称CUDA可运行。

```bash
cd /vol/grid-solar/sgeusers/linbocheng/SparseACO/GPLSACO
mkdir -p .tmp .cache/pip
export TMPDIR="$PWD/.tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python3 -m venv .venv
.venv/bin/python -m pip install -r provenance/python.lock.txt
```

## 2. GPU 使用

首选 RTX A5000，其次 RTX PRO 5000；已获用户授权使用空闲卡。用 `/home/linbocheng/bin/gpu-free` 检索（TMPDIR设项目内），再对目标host/UUID直接查询nvidia-smi的GPU与compute-apps。空闲快照不构成预约；启动前重检，运行中若被其他进程争用则该正式计时任务标记污染并按协议重跑。

2026-09-08启动检查：本机cuda-small1的三张RTX4000 Ada都忙；cuda02的A5000与若干Pro5000节点空闲。此表只是历史快照，不能作为以后启动依据。共享项目路径已在cuda02验证可见，无需把代码复制到外部目录。

调试只用一张卡。正式训练同一run固定同型号/批大小，不混用A5000与Pro5000的wall-clock fitness；额外设备可承担独立演化重复。优先按现有SSH/作业环境管理，无需先搭建Ray服务。不触碰其他项目或用户的运行进程。

## 3. 里程碑与产物

| 阶段 | 可交付成果 | 依赖 |
|---|---|---|
| P0 启动 | 本组文档、Git、数据/来源清单、环境与原生smoke | 无 |
| P1 语义 | 原生操作oracle、C++基础转移、IR三后端 | P0，允许彼此独立模块推进 |
| P2 完整底座 | 无GP CUDA、LS、重启、特征、deadline、Engine | G1/G2 |
| P3 训练闭环 | DEAP→IR→Engine→真实fitness→checkpoint→验证导出 | G3；先小规模同精度pilot |
| P4 E1 | 静态/规则调优、五重复完整/去反馈、分叉分析 | G4 |
| P5 E2/E3 | 各独立重训和固定条件矩阵，含负结果 | E1底座冻结 |
| P6 E4与整理 | 冻结10K/TSPLIB、统计与资源、证据到主张审计 | 原目标数据通过，模型已冻结 |

不预承诺几天内完成全部训练；P2/P3实测后再估计排期。发现阻塞时继续不依赖它的工作，如TSPLIB恢复不妨碍500/1K内核开发。

## 4. 计算预算估算

128×50=6,400个个体评价/演化重复，每个含500和1K两个规模工作波。如果每个规模的32个colony一次放下，主搜索时间下界为 `6400*(B500+B1000)`；五重复乘5，另加初始化/调度、验证shortlist、失败重跑、Static/Rule调参和消融。若每波容不下，按固定分波数再乘，不能把吞吐量当独占时延。

shortlist最多约50+128个程序（去重后减少），每个完整验证池可能远超训练panel成本；提前测量，避免只估训练不估选择。所有离线代价报告GPU小时与CPU小时，不用虚构加速比确定研究价值。

## 5. Git 与审计

远端为 `LinHuanli/GPLSACO`，初始为空；建立main并按可复核工作包提交。提交代码、中文文档、冻结配置、无私密信息的环境/来源hash和精简报告；artifacts、数据、环境和二进制忽略。第三方声明和许可证随实际复用同步。

每阶段报告列实际命令、结果、范围限制、剩余门槛。推送后核对远端commit；公开仓库不提交认证信息、其他用户GPU进程列表或未经许可的外部源包。实验任务manifest记录source commit，后续代码更改不会覆盖既有结果身份。
