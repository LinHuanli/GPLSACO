# 中文学术报告

主稿《利用Genetic Programming学习FACO的搜索控制策略》围绕“FACO的决策问题 → GP规则学习 → 与静态控制的质量比较 → 规则行为、迁移及成本”展开。Single-tree与multi-tree作为方法设计比较。正文使用中文，保留标准GP术语。补充材料整理历史实验、完整参数、全部六个冻结individual的树图、真实输入输出、逐seed结果与来源索引。结果截至2026-09-10，文稿整理于2026-09-13。

- [主稿 PDF](samplepaper.pdf)：13页，含参考文献；编译时要求不超过14页。
- [补充材料 PDF](supplement.pdf)：21页，实验沿革、完整树图、参数及统计。
- [主稿 LaTeX](samplepaper.tex)、[补充材料 LaTeX](supplement.tex)、[参考文献](references.bib)。
- [编译记录](../../results/v3/evostar/build.json)：实际页数、版面检查和输出位置。

当前结论是：两种GP在TSP500上优于GPU MNE8对照，但相对原生FACO和MNE16强对照尚无可靠优势；TSP1000上，固定MNE16/区域0对照更好。原始FACO、连续距离适配版和节点重定位GPU底座在文中明确区分。合成数据使用“参考差距”，不将未独立认证的标签称为最优值。

## 阅读顺序与图表

主稿先用overview和两段伪代码说明GP如何参与FACO求解，再给出representation、按含义分组的terminal可见性表与fitness公式。实验围绕三个问题展开：是否改善质量、学到了什么控制行为、迁移表现与计算代价如何。

- 图1：离线GP进化与在线FACO决策循环。
- 图2：TSP500和TSP1000相对各对照的改善及冻结95%区间；显著性标记仅用于TSP500的九项Holm校正比较。
- 图3：三个GP seed与均值的训练和固定quick-validation曲线；quick validation在训练后执行。
- 图4：两个seed 3313控制器的树／表达式及真实分数，展示priority score如何变成动作。
- 补充材料：全部六个individual的完整树图、代码字段对应、逐seed结果、在线monitor、全部配对统计及历史实验。

主稿图4采用TSP500解释面板首个实例、ACO seed 17、第100迭代的数据，两个控制器各自运行至该状态。它们用于展示解码过程，不构成同状态消融。树图直接从冻结IR生成，保留每个节点、拓扑及ERC值；图表生成不调用求解器。

## ECRG相关论文的写作依据

本次重写阅读了以下论文的方法、实验和结果分析，并查看了相应图表。这里记录组织方式及其在本稿中的具体应用，正式算法引用保存在参考文献中。

| 论文 | 阅读位置 | 本稿采用的组织方式 |
|---|---|---|
| [Zhang等，AI 2018：Multi-tree GP for DFJSS](https://fangfang-zhang.github.io/files/2018-AI-Multitree.pdf) | §3–5，Fig. 2的tree-swapping，terminal表与对照设计 | 先解释多个决策的作用，再说明individual与genetic operators；补充相关算子引用，将两种representation按完整设计比较 |
| [Sun等，IEEE TSC 2024：Multi-cloud GP](https://zaixing-sun.github.io/publications/Sun2024-TSC.pdf) | §IV，Fig. 2–4和Table II；§V.D–E，Fig. 7–8 | 连接overview、tree角色、terminal可见性与fitness；用具体规则及解码示例解释控制偏好 |
| [MacLachlan等，Evolutionary Computation 2020：UCARP Vehicle Collaboration](https://arxiv.org/pdf/1911.08650) | §4的fitness与GP框架；§5实验；§6.3–6.4的规则语义和路线案例 | 先定义求解过程中的决策接口；结果由质量比较进入表达式、候选偏好与实际行为分析 |
| [Mei和Zhang，CEC 2018：Orienteering GP](https://homepages.ecs.vuw.ac.nz/~yimei/papers/CEC18-Orienteering.pdf) | §III.B的fitness公式，§IV.B–C的性能和look-ahead分析 | 清楚连接priority function与最终解；如实报告增强设计未获得明确收益的结果 |

写作采用“问题和作用 → 方法定义 → 实验证据 → 解释”的顺序。每个结果段先给出具体发现，再引用图表与数值；限制说明紧邻相关推论。工程调度、精度标定、繁殖重试和历史记录放在补充材料。实验参数与统计沿用本项目协议，文献中的30 runs、不同种群规模和检验方法不替换已有设置。

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

仅更新图表可加 `--figures-only`。历史标定快照已保存在[精简证据](../../results/v3/evostar/evidence.json)，默认构建不读取历史原始输出。`--refresh-evidence`仅用于从已有历史汇总更新该快照，不执行标定或求解。

模板保留LNCS字体大小和版心；`orivec`选项仅采用原始向量符号定义，避免与amsmath冲突。作者暂为匿名。

## 目录与数据对应

| 内容 | 文件／主要来源 |
|---|---|
| 引言、动机与相关工作 | `sections/introduction.tex`；FACO与四篇GP规则学习论文 |
| 方法 | `sections/method.tex`；当前GP繁殖、CUDA控制和蚁群实现 |
| 实验设计 | `sections/design.tex`；`results/v3/representation-50gen/campaign.json`、`panels.json` |
| 正式结果、曲线、讨论 | `sections/results.tex`、`sections/discussion.tex`；同目录`summary.json`、`analysis.json` |
| 树图与输入输出 | `summary.json`中的冻结IR、`decision_examples.json`；生成`structure_*.tex`和`decision_figure.tex` |
| 反馈诊断 | `selected_feedback_diagnostic.json`，只作固定状态评分干预 |
| 原生FACO复现 | `results/v2/faco_2022_tsplib.json` |
| 精度／预算标定 | `results/v3/evostar/evidence.json`，附原始来源路径 |
| 10代预实验、旧繁殖回放、加速 | `results/v3/pilot_analysis.json`、`historical_variation_replay.json`、`representation_performance.json` |
| 历史32蚂蚁实验 | `docs/archive/v1/`与`results/v1/`；与当前主结果分开记录 |

表格与曲线由[构建脚本](../../scripts/build_evostar_report.py)直接生成。bootstrap区间和Holm校正读取冻结统计；脚本不以图表显示精度重新做检验。全文没有把工程检查或部分训练记录写成E1–E4的完整科学结论。

`llncs.cls`和`splncs04.bst`来自用户提供的模板，保留原文件；`readme.txt`、`history.txt`保留其说明。原示例`fig1.eps`、`llncsdoc.pdf`未用于本稿，留在本地且不提交。
