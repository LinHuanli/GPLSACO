# E1：LS 后反馈的样本外价值

状态：主训练/比较配置进入[具体冻结协议](../planning/16_e1_freeze_and_execution.md)，完整FE校准已通过；正式测试仍须完整G4及全部训练/验证和基线选择封存。无测试结果。公共预算/统计见[公共协议](../planning/07_evaluation_and_statistics.md)。

## 1. 处理与主要终点

在同一FACO-Control、候选、LS、档案、动作空间和硬件上比较Static-Control、Rule-Control、GP-NoFeedback、GP-Full。NoFeedback删除stagnation/return_rate/ls_work后独立重训，评价次数`progress`和当前结构/信息素仍保留；结论限定为显式LS后反馈的额外价值。主预算按[次数协议](../planning/12_evaluation_count_protocol.md)，不设置求解秒数上限。

每个GP条件至少5个演化seed；500/1K共同面板、统一验证选择。Static固定MNE、区域规则、重启周期/概率的开发搜索需记录覆盖和成本；Rule只用预声明的停滞阈值族，不在测试后追加规则。

主终点是中预算同分布测试的实例平均gap，按500/1K分组及等权宏平均报告。Full对Static、Rule、NoFeedback三项主比较及Holm校正提前冻结。短/长预算和同接口bandit作为补充；FACO-Native/faco_apt另列原生硬件身份。

## 2. 状态分叉

只从开发实例采样：按照q和c的高/低四层，阈值取独立开发pilot的预定分位数后冻结；每实例/来源/层固定快照上限，不能只挑GP成功的轨迹。快照含参考、档案、信息素/default/cache、反馈、RNG和pending状态。

对同一快照、同一reference和region比较MNE2与16；各用配对分叉seeds，执行一个动作后使用同一固定继续策略、相同额外FE，不设置算法墙钟上限。每次恢复完整状态，不能让第二条分支继承第一条结果。源progress分母与批次编号保持不变；接口与验收见[状态分叉实现契约](../planning/21_e1_state_fork_engine.md)。按实例聚合大扰动相对小扰动的后续gap差及工作量，按来源分层。

正式运行前固定每层快照数、分叉seed数、继续策略、额外FE及缺层处理；缺少某状态层要报告覆盖，不能强行补造状态。

## 3. 产物与判断

产物：四控制器gap/anytime表；训练seed稳定性；分叉配对收益图；q/c分布、action/mask/tie饱和率诊断；全部任务和失败表。支持RQ1需要Full对去反馈重训的稳定优势，以及状态条件下动作收益可重复变化。只有Full超过未调优Static不充分；只有相关性轨迹也不充分。无差异时保留结果并按预定工程排查清单核查，不扩张grammar追逐测试集。
