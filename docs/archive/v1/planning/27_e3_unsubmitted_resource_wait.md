# E3提交前资源暂停的原地接续

2026-09-09。alpha-Hard/1103与POPMUSIC-Escape/4409原完整CLI因实际外来进程在提交前退出，分别保留2,736与1,196个已完成调用；下一待办attempts为空，无返回文件。先保存静止快照、原日志、资源及退出码，再用原独立训练审计重放前缀。已有worker字段是历史checkpoint记录，不把退出后的旧PID当作仍在运行。

`scripts/wait_e3_unsubmitted_resource.py`绑定原host/UUID、型号、driver、run ID、已审计checkpoint及源码SHA，沿用原`research_e3.py train --resume`和完整128×50及验证配置。在compute processes为0、显存≤1,024 MiB、util≤5%之前仅作CPU观察，每5秒采样，不创建CUDA上下文、不增加FE、不设算法墙钟上限。

原CLI若再次以同一明确的提交前资源错误退出，且新的待办仍为空attempt、无返回文件、最后资源观察为正整数外来进程数，等待器可回到等待。每轮记录日志、退出码、实际CPU/墙钟/RSS，核验所有原completed hash保持不变。未知占用、实际返回、running尝试和其他异常必须单独核查，不能通过此路径重算。原提交与求解代码保持冻结。

本次POPMUSIC-Hard/3313另发生了原GPU从空闲到worker初始化之间的竞争，采用已经冻结的`wait_e3_startup_resource.py`：独立核对已审计前缀的completed、费用、worker历史完全未变、无新增原生提交，绑定原明确初始化占用日志，再等待原UUID。所有恢复均保留既有资源偏差；完整终态及独立审计仍是放行门槛。
