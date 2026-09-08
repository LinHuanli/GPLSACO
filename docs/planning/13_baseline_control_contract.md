# Static/Rule共同底座控制契约

本文件细化v4中“充分调参的Static-Control”和“简单停滞阈值Rule-Control”的原生实现。主预算遵循[评价次数协议](12_evaluation_count_protocol.md)。这里定义参数族及行为，尚未完成开发调优，不将示例参数视为最终对照方法。

## 共同输入与原生入口

`FacoBatchEngine.evaluate_baseline_evaluations(keys, seeds, evaluation_limit_per_colony, policy, preparation_mode="cached", experiment_mask=UINT32_MAX)`在一次原生调用内完成全部搜索，无Python在线策略回调、无求解秒数上限。`policy`是[规格文件](../../configs/baseline_policy_v1.json)中示例所示的11字段字典，规范版本为1；返回值含实际使用的规范配置，避免释放GIL后输入字典变动造成身份歧义。

Static、Rule与GP共用同一个Engine、共同准备/初始tour、全部区域和特征构造、档案/替代参考、epoch重启事务、蚂蚁构造、checklist LS、信息素及来源选择。当前仅替换动作选择内核，未对基线省略十二特征等共同成本；资源测量须注明这一实现。每个方法使用同一per-colony FE限额，LS内部工作量另列。

C++诊断复用Scoring event槽；在基线入口中它表示基线动作选择，即使历史槽名为`gp_scoring`，也不能把该值解释为执行了GP程序。Python生产入口不导出该诊断profile，后续成本汇总必须按控制器身份命名。

控制动作仍为`16*restart + 4*region + level`，其中level 0/1/2/3对应原生MNE 2/4/8/16。固定区域指共享区域生成器的规则编号，不是固定城市ID集合；区域仍随实例、seed、批次与参考变化。档案无合法替代时，重启请求退回同区域/同MNE的保持动作。

实验mask必须保留该配置所有可能选择的保持动作，否则拒绝输入；不会悄悄切换到别的区域或程度。重启位可以被实验mask屏蔽。任何不适用字段必须规范为0，避免一个行为对应多个含无效参数的配置身份。

## Static参数族

Static固定区域和MNE，重启可关闭、按周期请求或按固定概率请求。周期P以已完成批次数为时钟：在batch索引P、2P、3P之前请求，batch 0不请求；如果该点重启被mask屏蔽则跳过，不积压待触发事件。概率模式使用`u<p`严格比较，允许p为0或1作为有明确行为的边界。

Bernoulli从同一colony solve key的独立Philox槽取得uniform53，位置为`(batch, ant=UINT32_MAX, step=1)`。现有来源选择使用同一保留ant的step 0，普通蚂蚁采用自身ant编号，因此基线重启不消耗或移动蚂蚁/来源的随机序列。该随机流不依赖GPU并发colony编号。

## Rule参数族

Rule固定区域，程度可固定，也可随全局停滞逐档增加：`base + min(max-base, stagnant_batches // step)`。base/max均为0..3；固定程度时step=0，升级时step为正。全局停滞使用完成批次中GB未严格改善的实际整数计数，不从FP32裁剪后的`stagnation`终端反推，不读取最优值或测试标签。

启用规则重启时，需要全局停滞达到阈值R，且当前epoch已完成批次数达到冷却C；R、C均为正。关闭时两者均为0。重启继续保留GB、档案和全局停滞，重置epoch与q/c反馈；由于全局停滞保留，冷却不能省略，否则达到阈值后可能逐批重启。严格GB改善使全局停滞归零，MNE升级自然恢复至base。重启本身不额外减少FE。

## 验收与后续调优

CPU以手算动作检查32动作映射、周期边界、概率严格比较、升级/饱和、跨epoch冷却及非法配置。CUDA以等动作GP作为共同底座对照，比较完整tour、档案、信息素、工作状态和来源随机数；检查规则/概率的真实动作、单/批量一致、主机延迟及插桩不改变轨迹。公共Python入口另覆盖31/500/1K合成点集和基线–GP–基线复用，正式数据集调参单列。

接下来须把基线纳入可恢复的离线配置搜索及外部fitness核验，预先登记development面板、配置族覆盖、共同FE限额与选择规则。Static需覆盖4个MNE、4个区域及充分的重启参数；Rule需覆盖程度固定/升级、重启阈值和冷却选择。调优全过程的配置数、FE、失败及实际CPU/GPU资源都必须保留。开发集选择结束后才冻结G4方法身份及五演化seed/正式测试manifest；不得把这里的少数示例当作“充分调参”的证据。
