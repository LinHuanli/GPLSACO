# 数据准备、距离语义和标签隔离

## 1. 权威边界与实际输入

外部 `Datasets` 只读。其 README/manifest 属于另一研究，限制 TSP1K 的语句不适用于本次用户授权；本研究仍按 v4 仅用 500/1K 训练和选择。先登记所有可用文件，再按用途选取，不继承旧 `excluded_files`，但应排除内容重复文件。

合成 `.txt` 每行是 `x1 y1 ... xn yn output v1 ... vn v1`，tour 使用 1-based 编号。解析允许最后闭合节点省略，但要求恰好 n 个互异合法节点；坐标必须有限，维数明确，闭合端点一致。内部统一 0-based，不把重复首节点保存两次。行号、字节偏移、文件 SHA-256 与原始行指纹共同定位实例。

本地 TSPLIB 文本同样是坐标＋tour 行，而非原始 TSPLIB 文件；例如 `berlin52.txt` 坐标在归一化范围内。不能通过文件名猜 `EDGE_WEIGHT_TYPE`，也不能无依据逆缩放恢复原整数坐标。

## 2. 两层核验

**文件层：** 全量流式 SHA-256、非空行数、字节数；检查逐字节重复；记录 first-record 的解析与 tour 可行性，明确抽样不代表全部记录通过。完整逐记录验证在建立正式训练/验证/测试面板时执行，失败样本不会静默丢弃。

**实例层：** 对所有实际纳入研究的记录核验坐标、tour 排列、闭环和目标长度。坐标规范化 hash 按点对排序检测节点置换重复；有父实例来源时按 parent ID 分组。不能声称点对排序能检测所有旋转、缩放和子采样；来源不明时记录此局限，并增加生成元数据检查。

合成首版目标用 FP64 连续欧氏距离，标签成本是给定 tour 在同一目标下的完整重算。此处“可行性＋成本一致”不等于“已独立证明最优”。标签登记：`declared_status=user_supplied_optimal`、`certificate_status`、`source_solver`（unknown/文档证实）、`distance_spec`、`label_tour_hash`、`recomputed_cost`。缺精确证书时保留标签并称 reference gap，绝不截断负 gap；负值触发距离/标签审计。

## 3. TSPLIB 正式恢复链路

1. 从原始数据或官方 TSPLIB 源恢复 `.tsp` 和距离类型，保存下载来源、版本和完整 SHA-256，放项目 `.deps` 或 `artifacts/data`。
2. 建立原始节点到现有标签编号的可验证映射。无法证明同一点集与顺序时使用官方 tour/length，不能将归一化坐标近似当原坐标。
3. 实现实际需要的 EUC_2D、CEIL_2D、ATT、GEO 或 EXPLICIT，按定义逐边重算；不支持的类型明确列入未支持表。
4. 每条标签路线独立算成本，与对应最优值比较；只有长度标签可计算 gap，但不能计算最优边覆盖率。
5. 校验后的原目标实例才进入 E4；归一化衍生问题可另列补充结果，不冠以官方 TSPLIB 目标。

距离规则依据 [TSPLIB 官方 FAQ](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSPFAQ.html)。文档说明不替代当前输入的逐实例校验。

## 4. 划分协议

| 池 | 预定来源与数量 | 用途 |
|---|---|---|
| 开发 | 500/1K training 各保留 128 个 parent groups | 工程调试、底座和特征数值校准，排除出正式测试 |
| 主训练 | 500/1K training，各至少数千；使用全量合格池 | 每代每规模 16 实例×2 solve seeds，共同面板 |
| 验证 | 优先独立 val 文件；不足 200 时由 training 的独立 parent groups 补到 256/规模 | 所有模型选择、静态调参；不借用 test 补足 |
| 同分布测试 | 原 test 文件全部合格独立实例；不足建议 200 时如实使用实际数目 | 正式 E1/E2；也可预留 training 独立组作补充并单独标注来源 |
| 100/10K/TSPLIB | test-only；10K 的 train 文件名不改变其在本研究中的迁移用途 | 冻结后 E4；不据其解质量设计特征或参数 |
| cluster/Gaussian | 检查实际独立来源，作为未训练分布 | E4 补充；不能从文件名推断生成机制 |

首先核实真实数量，以上不是现有实例计数。用固定 split seed、排序后的稳定实例 ID 和 parent groups 生成 `splits.v1.json`。研究初始化时登记哈希；配置冻结时再核验全量源文件 hash。测试访问分为格式/标签审计和性能揭盲，分别记录。

## 5. 在线与离线边界

`Instance` 仅含坐标/距离元数据、候选和不含答案的 ID；`Label` 单独存在于 Python evaluator。native adapter 导出连续距离 EXPLICIT 矩阵只适合小规模 CPU 语义检查，写盘成本单记，不成为 10K 内层表示。求解器运行目录不放 `best-known.json` 标签表；如原生入口要求文件，提供空对象且检查没有标签停止行为。

阶段产物：本研究文件 manifest、重复表、标签审计摘要、选定实例偏移索引、split manifest、TSPLIB 匹配表。原始数据、标签全集与矩阵不提交 Git。
