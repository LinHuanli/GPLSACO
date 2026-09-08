# 主基线提交前资源暂停的原地接续

2026-09-09。完整776配置基线在第3,467个已完成调用之后因原cuda04 GPU出现外来进程退出。下一任务尚无attempt、无返回文件；原checkpoint、日志与退出码已保存。接续前独立重建完整6,208项搜索计划及已完成前缀，逐条核验任务身份、路线成本、外部fitness和FE/资源账目；本次前缀审计不产生候选选择。

独立等待器`scripts/wait_baseline_resource.py`绑定原host/UUID、型号、driver、run ID、配置、checkpoint、审计和源码SHA。在compute processes为0、显存≤1,024 MiB且util≤5%时运行原冻结`search_baselines.py --resume`，继续全部776配置及固定验证；无算法墙钟上限。等待阶段不创建CUDA上下文，不提交求解，不增加FE。

若原CLI再次以明确的提交前外来进程异常退出，且当前待办attempts为空、无返回文件，等待器可继续等待原设备。每轮单独保存日志、实际资源与退出码，核验所有已有completed hash保持不变。实际返回、running尝试、未知异常均须单独检查，不重复求解。资源观察和求解失败分列，受干扰运行的整体计时单列。

独立审计通过及完整终态审计是不同门槛；本协议不缩减搜索或验证，不提前放行TEST。E1 Full-1103同次cuda04占用事件使用已冻结的[原E1等待协议](25_e1_resource_wait.md)，保留原128×50训练和全部验证。
