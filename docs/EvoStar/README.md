# 中文学术报告

主稿《利用Genetic Programming学习FACO的搜索控制策略》围绕“local search需要怎样的候选起点 → FACO的结构化扰动与搜索记忆 → GP学习外层控制 → 完整求解质量、规则行为及搜索成本”展开。Single-tree与multi-tree作为方法设计比较。正文使用中文，保留标准GP术语。正文完整定义GP方法与实验参数；补充材料整理历史实验、六个选定GP个体的完整树图、真实输入输出、逐seed结果与来源索引。结果截至2026-09-10，文稿整理于2026-09-14。

- [主稿 PDF](samplepaper.pdf)：当前22页，完整方法、GP与FACO设置、正式实验、控制机制图及代表性规则分析；正文不设页数上限。
- [补充材料 PDF](supplement.pdf)：当前19页，实验沿革、六个选定个体的完整树图与输入输出、实现对应和详细统计。
- [主稿 LaTeX](samplepaper.tex)、[补充材料 LaTeX](supplement.tex)、[参考文献](references.bib)。
- [编译记录](../../results/v3/evostar/build.json)：实际页数、版面检查和输出位置。

当前结论是：两种GP在TSP500上优于GPU MNE8对照，但相对原生FACO和MNE16强对照尚无可靠优势；TSP1000上，固定MNE16/区域0对照更好。原始FACO、连续距离适配版和节点重定位GPU底座在文中明确区分。合成数据使用“参考差距”，不将未独立认证的标签称为最优值。

## 阅读顺序与图表

主稿先说明在线搜索如何将参考路径与信息素转化为候选路径和初始LS checklist，再说明GP如何通过完整求解学习这些控制决策。正文完整给出representation、terminal的计算定义与角色可见性、function表、fitness、初始化与遗传算子，以及独立的GP和FACO参数表。实验围绕三个问题展开：相同FE下是否改善完整求解质量、学到了怎样的扰动与参考路径选择、迁移表现与计算代价如何。

- 图1：离线GP进化与在线搜索循环；分开显示参考路径／信息素、聚焦构造和local search，以及改进结果返回外层控制的反馈。
- 图2（第6页）：三棵树分别控制参考路径、扰动起点区域和MNE；用小规模路径展示构造与local search的关系。
- 图3：TSP500和TSP1000相对各对照的改善及冻结95%区间；显著性标记仅用于TSP500的九项Holm校正比较。
- 图4：三个GP seed与均值的训练和固定quick-validation曲线；quick validation在训练后执行。
- 图5：GP-J/3313的完整树及真实分数，展示priority score如何变成动作。
- 图6（第20页）：GP-C/3313的三棵完整树、关键输入、候选评分和选中动作；每行解释一个控制位置。
- 补充材料：全部六个individual的完整树图、代码字段对应、逐seed结果、在线monitor、全部配对统计及历史实验。

主稿图5和图6采用TSP500规则解释实例集首个实例、ACO seed 17、第100迭代的数据，两个GP策略各自运行至该状态。它们用于展示解码过程，不构成同状态消融。树图直接从冻结IR生成，保留每个节点、拓扑及ERC值；图表生成不调用求解器。

图2使用固定12城市的不规则二维散点，所有子图共用坐标、比例和1–12编号，路径图宽约26毫米。以小节点和外置编号避免遮住非关联边；示意起点集合为参考路径上连续的城市5、12、8、6，采样起点为城市5。灰色实线为保留边，橙色虚线为扰动新边，绿色点线为local search引入的边；最终路径中保留下来的扰动新边继续显示为橙色。

参考路径长度42.8424，已无改进2-opt移动，但不是全局最优。弱扰动将城市8移至5之后，长度增至43.2642，随后一次2-opt返回原路径。较强扰动将城市6移至5之后、城市9移至7之后，长度增至44.4579；一次2-opt反转片段12–8–7，得到长度39.9170的另一条局部最优路径，比参考路径缩短6.83%。所有长度从示意坐标计算，所有tour完整闭合；已枚举参考与最终路径各54个2-opt移动，核对局部最优性。这是扰动帮助local search到达更好解的成功机制示例，不是FACO实测轨迹，不对应指定MNE，也不表明扰动越强越好。

图6的真实动作是“保留当前参考路径＋区域1＋MNE16”。参考路径分数并列，区域1的archive disagreement与pheromone strength乘积超过progress，强度树的公共偏置不改变MNE排序。已有记录没有保存该步构造前后及LS后的路径，不能从中计算单步收益；正文将规则偏好、观测动作、整体质量与工作量分别解释。

## Local-search-centred研究视角

本文围绕local search设计候选起点生成与搜索中心选择。对复用参考路径的FACO，构造可以理解为带有信息素记忆的结构化扰动；这是一种设计视角，不是所有ACO与扰动算法等价的命题，也不是LS运行时间占比的测量。

依据[FACO 2022原文](https://arxiv.org/pdf/2203.02228)的§4.3–4.4、§5.3和§6，文稿明确区分主动构造新连接、复用剩余路径及通过checklist组织局部改进。每次仍生成完整tour，接受LS改进后检查表可以扩展；MNE不等于最终边差异，区域也不是独立子问题。原文已有构造与扰动的联系，本文贡献定位为GP联合学习外层决策，保留后续自适应FACO的相关工作及GPU底座版本说明。

质量与LS工作量共同用于解释结果，训练fitness仍只优化完整求解的终点质量。现有反馈干预只证明部分动作敏感；完整求解的反馈、区域和重启消融仍未完成。主稿的科学结论限于已有TSP500训练及TSP1000迁移数据。此次机制图与真实案例扩充没有新增ACO评价，已有结果表、曲线、冻结策略、实验参数与统计结果不变。

## GP设置与术语

正文的Terminal Set定义12个特征、归一化与平滑公式、常量和ERC；Function Set列出七种函数的元数及语义。Initialization and Genetic Operators完整说明修改后的full/grow、父本选择、单树与三树交叉、变异、复制、多样性约束、重试与随机移民。GP Parameter Settings和FACO Parameter Settings分别汇总实际参数与设置依据，不将预算选择称为最优配置。

GP个体指进化中的程序，进化搜索策略指其在FACO中执行的决策规则；表格以“进化运行”标识方法与GP seed，以“GP策略”标识选定规则。迭代预算预实验使用24种人工设计搜索策略：16种固定配置、4种带随机重启的配置、4种基于停滞反馈的人工规则。它们用于选择fitness evaluation中的ACO迭代数，不是本轮进化产生的个体。

正文保留代表性规则的解释；六个选定GP个体的完整树图、输入输出与逐项统计仍在补充材料。论文只保留版面与引用检查，构建脚本记录实际页数，不再因正文长度拒绝编译。

## ECRG相关论文的写作依据

本次重写阅读了以下论文的方法、实验和结果分析，并查看了相应图表。这里记录组织方式及其在本稿中的具体应用，正式算法引用保存在参考文献中。

| 论文 | 阅读位置 | 本稿采用的组织方式 |
|---|---|---|
| [Zhang等，AI 2018：Multi-tree GP for DFJSS](https://fangfang-zhang.github.io/files/2018-AI-Multitree.pdf) | §3–5，Fig. 2的tree-swapping，terminal表与对照设计 | 先解释多个决策的作用，再说明individual与genetic operators；补充相关算子引用，将两种representation按完整设计比较 |
| [Sun等，IEEE TSC 2024：Multi-cloud GP](https://zaixing-sun.github.io/publications/Sun2024-TSC.pdf) | §IV，Fig. 2–4和Table II；§V.D–E，Fig. 7–8 | 连接overview、tree角色、terminal可见性与fitness；用具体规则及解码示例解释控制偏好 |
| [MacLachlan等，Evolutionary Computation 2020：UCARP Vehicle Collaboration](https://arxiv.org/pdf/1911.08650) | §4的fitness与GP框架；§5实验；§6.3–6.4的规则语义和路线案例 | 先定义求解过程中的决策接口；结果由质量比较进入表达式、候选偏好与实际行为分析 |
| [Mei和Zhang，CEC 2018：Orienteering GP](https://homepages.ecs.vuw.ac.nz/~yimei/papers/CEC18-Orienteering.pdf) | §III.B的fitness公式，§IV.B–C的性能和look-ahead分析 | 清楚连接priority function与最终解；如实报告增强设计未获得明确收益的结果 |

写作采用“问题和作用 → 方法定义 → 实验证据 → 解释”的顺序。每个结果段先给出具体发现，再引用图表与数值；限制说明紧邻相关推论。影响GP后代分布的重试、回退和随机移民规则放在正文；工程历史、数值精度预实验、完整个体图与详细统计放在补充材料。实验参数与统计沿用本项目协议，文献中的30 runs、不同种群规模和检验方法不替换已有设置。

三个GP seed在学习曲线中分别呈现；性能区间沿用原分层配对bootstrap。反馈terminal的出现、分数变化、动作变化与最终质量收益分别描述。现有反馈干预属于固定状态的评分敏感性诊断，本文未将其写为完整求解消融。

## 直接编译论文

在 `docs/EvoStar` 目录执行：

```sh
latexmk -xelatex samplepaper.tex supplement.tex
```

PDF生成在同目录。编辑器或Overleaf应选择 **XeLaTeX**，主文档选择 `samplepaper.tex`；补充材料单独编译 `supplement.tex`。两份入口已添加编译器提示，各章节已标记主文档。目录中的 `.latexmkrc` 默认使用XeLaTeX，也兼容编辑器自带的 `latexmk -pdf` 配方。

原先报错 `Font unihei73 ... not found` 是误用pdfLaTeX导致的；不要用pdfLaTeX直接编译这份中文稿。现在误选引擎会立即提示需要XeTeX。

论文目录包含全部表格 `.tex` 和矢量图 `.pgf` 源码，复制这个目录即可编译，无须项目Python环境、实验数据或 `.tmp` 文件。需要安装XeLaTeX、latexmk、BibTeX，以及ctex/Fandol、TikZ、algorithm/algpseudocode、fvextra等LaTeX包。没有latexmk时，可按 `xelatex → bibtex → xelatex → xelatex` 的顺序编译同一主文件。

## 从冻结结果更新图表

只有需要重新生成图表时，才在GPLSACO根目录执行：

```sh
TMPDIR="$PWD/.tmp" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
TEXMFVAR="$PWD/.cache/texmf-var" PYTHONPATH=python \
.venv/bin/python scripts/build_evostar_report.py
```

脚本读取版本化精简结果，更新 `docs/EvoStar/generated/` 中的表格和PGF矢量源码，然后编译两份文档。LaTeX缓存和日志写入 `.tmp/evostar/`，PDF副本写入 `artifacts/papers/evostar/`。缓存及最终PDF不提交；图表源码作为论文的一部分提交。生成过程使用项目Python环境中的Matplotlib/NumPy及XeLaTeX，不启动实验，也不重新计算统计检验。

仅更新图表可加 `--figures-only`。历史预实验快照已保存在[精简证据](../../results/v3/evostar/evidence.json)，默认构建不读取历史原始输出。`--refresh-evidence`仅用于从已有历史汇总更新该快照，不执行预实验或求解。

模板保留LNCS字体大小和版心；`orivec`选项仅采用原始向量符号定义，避免与amsmath冲突。作者暂为匿名。

## 目录与数据对应

| 内容 | 文件／主要来源 |
|---|---|
| 引言、动机与相关工作 | `sections/introduction.tex`；FACO与四篇GP规则学习论文 |
| 方法 | `sections/method.tex`与`sections/gp_details.tex`；当前GP繁殖、CUDA评分和蚁群实现 |
| 实验设计 | `sections/design.tex`；`results/v3/representation-50gen/campaign.json`、`panels.json` |
| 正式结果、曲线、讨论 | `sections/results.tex`、`sections/discussion.tex`；同目录`summary.json`、`analysis.json` |
| 控制机制示意 | 图表脚本中的固定示意城市与完整tour；生成`control_mechanism.tex`，不代表实验轨迹 |
| 树图与输入输出 | `summary.json`中的冻结IR、`decision_examples.json`；生成`structure_*.tex`、`decision_figure.tex`与`conditional_decision_figure.tex` |
| 反馈诊断 | `selected_feedback_diagnostic.json`，只作固定状态评分干预 |
| 原生FACO复现 | `results/v2/faco_2022_tsplib.json` |
| 数值精度／迭代预算预实验 | `results/v3/evostar/evidence.json`，附原始来源路径 |
| 10代预实验、旧繁殖回放、加速 | `results/v3/pilot_analysis.json`、`historical_variation_replay.json`、`representation_performance.json` |
| 历史32蚂蚁实验 | `docs/archive/v1/`与`results/v1/`；与当前主结果分开记录 |

表格与曲线由[构建脚本](../../scripts/build_evostar_report.py)直接生成。bootstrap区间和Holm校正读取冻结统计；脚本不以图表显示精度重新做检验。全文没有把工程检查或部分训练记录写成E1–E4的完整科学结论。

`llncs.cls`和`splncs04.bst`来自用户提供的模板，保留原文件；`readme.txt`、`history.txt`保留其说明。原示例`fig1.eps`、`llncsdoc.pdf`未用于本稿，留在本地且不提交。
