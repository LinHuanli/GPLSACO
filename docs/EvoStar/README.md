# 中文学术报告

主稿围绕“结构复用中的控制问题 → 联合单树与条件三树 → 强静态对照 → 行为与成本”展开。补充材料按实验阶段整理已完成工作，并提供完整参数、全部选中树、真实输入输出、逐种子结果与来源索引。结果截至2026-09-10，文稿整理于2026-09-13。

- [主稿 PDF](../../artifacts/papers/evostar/samplepaper.pdf)：含参考文献，编译时要求不超过14页。
- [补充材料 PDF](../../artifacts/papers/evostar/supplement.pdf)：实验沿革、完整记录与解释。
- [主稿 LaTeX](samplepaper.tex)、[补充材料 LaTeX](supplement.tex)、[参考文献](references.bib)。
- [编译记录](../../results/v3/evostar/build.json)：实际页数、版面检查和输出位置。

当前结论是：两种GP在TSP500上优于GPU MNE8对照，但相对原生FACO和MNE16强对照尚无可靠优势；TSP1000上，固定MNE16/区域0对照更好。原始FACO、连续距离适配版和节点重定位GPU底座在文中明确区分。合成数据使用“参考差距”，不将未独立认证的标签称为最优值。

## 一条命令构建

在GPLSACO根目录执行：

```sh
TMPDIR="$PWD/.tmp" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
TEXMFVAR="$PWD/.cache/texmf-var" PYTHONPATH=python \
.venv/bin/python scripts/build_evostar_report.py
```

环境需具备项目Python环境中的Matplotlib/NumPy，以及XeLaTeX、latexmk、BibTeX、pdfinfo。LaTeX使用ctex、Fandol、TikZ、algorithm/algpseudocode、fvextra等包；图表通过fontconfig的`fc-match`读取Droid Sans Fallback中文TrueType字体，嵌入矢量PDF。模板保留LNCS字体大小和版心；`orivec`选项仅采用原始向量符号定义，避免与amsmath冲突。作者暂为匿名。

构建仅读取版本化的精简结果，不启动实验，也不重新计算统计检验。图、表、LaTeX缓存和日志写入`.tmp/evostar/`，最终PDF写入`artifacts/papers/evostar/`。这些生成物不提交仓库。源文件和图表脚本足以重新生成PDF；首次拉取仓库后先运行上述命令，PDF链接才会存在。

只生成图表可用`--figures-only`。历史标定快照已经保存在[精简证据](../../results/v3/evostar/evidence.json)，默认构建不需要历史原始输出。只有明确要更新这两份已有历史汇总时才使用`--refresh-evidence`，它读取快照记录的原始JSON路径，不执行标定或求解。

## 目录与数据对应

| 内容 | 文件／主要来源 |
|---|---|
| 引言、动机 | `sections/introduction.tex`；2022 FACO、2024自适应FACO等一手文献 |
| 方法 | `sections/method.tex`；当前GP繁殖、CUDA控制和蚁群实现 |
| 实验设计 | `sections/design.tex`；`results/v3/representation-50gen/campaign.json`、`panels.json` |
| 正式结果、曲线、讨论 | `sections/results.tex`、`sections/discussion.tex`；同目录`summary.json`、`analysis.json` |
| 树与输入输出 | `summary.json`中的冻结IR、`decision_examples.json` |
| 反馈诊断 | `selected_feedback_diagnostic.json`，只作固定状态评分干预 |
| 原生FACO复现 | `results/v2/faco_2022_tsplib.json` |
| 精度／预算标定 | `results/v3/evostar/evidence.json`，附原始来源路径 |
| 10代预实验、旧繁殖回放、加速 | `results/v3/pilot_analysis.json`、`historical_variation_replay.json`、`representation_performance.json` |
| 历史32蚂蚁实验 | `docs/archive/v1/`与`results/v1/`；与当前主结果分开记录 |

表格与曲线由[构建脚本](../../scripts/build_evostar_report.py)直接生成。bootstrap区间和Holm校正读取冻结统计；脚本不以图表显示精度重新做检验。全文没有把工程检查或部分训练记录写成E1–E4的完整科学结论。

`llncs.cls`和`splncs04.bst`来自用户提供的模板，保留原文件；`readme.txt`、`history.txt`保留其说明。原示例`fig1.eps`、`llncsdoc.pdf`未用于本稿，留在本地且不提交。
