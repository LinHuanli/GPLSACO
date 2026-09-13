#!/usr/bin/env python3
"""从冻结精简结果生成中文论文图表和 PDF；不启动实验、不计算文件摘要。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/EvoStar"
GEN = SOURCE / "generated"
EVIDENCE = ROOT / "results/v3/evostar/evidence.json"
BASE = ROOT / "results/v3/representation-50gen"
LABELS = {
    "joint_single": "GP-J",
    "conditional_three": "GP-C",
    "faco_2022_continuous_identity_guard": "FACO-N",
    "gpu_faco_mne8_uniform_no_restart": "GPU-U8",
    "gpu_fixed_mne16_region0_no_restart": "GPU-R16",
    "gpu_faco_mne16_uniform_no_restart": "GPU-U16",
}
FEATURES = [
    "progress",
    "stagnation",
    "return_rate",
    "ls_work",
    "restart",
    "mne_level",
    "ref_gap",
    "ref_diff",
    "region_excess",
    "archive_disagreement",
    "pheromone_strength",
    "region_dispersion",
]
SYMBOLS = [
    "p",
    "s",
    r"\rho",
    "w",
    "r",
    "u",
    r"\delta_q",
    "d_q",
    "e_R",
    "a_R",
    r"\tau_R",
    "d_R",
]


def read(path):
    return json.loads(Path(path).read_text())


def esc(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def label(run):
    for name, short in LABELS.items():
        if run == name or run.startswith(name + "-s"):
            return run.replace(name, short).replace("-s", "/")
    return esc(run)


def write_table(name, headers, rows, columns=None):
    """所有数值表直接由记录产生，表中不做新的统计推断。"""
    columns = columns or "l" + "r" * (len(headers) - 1)
    text = [
        r"\begin{tabular}{@{}" + columns + r"@{}}",
        r"\toprule",
        " & ".join(headers) + r" \\ \midrule",
    ]
    text.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    text.extend([r"\bottomrule", r"\end{tabular}"])
    (GEN / f"{name}.tex").write_text("\n".join(text) + "\n")


def expression(program):
    """按存档后缀 IR 显示原始树，不化简，不重新评分。"""
    stack = []
    names = {2: "ADD", 3: "SUB", 4: "MUL", 5: "MIN", 6: "MAX", 8: "AQ"}
    for op, arg in zip(program["opcode"], program["operand"], strict=True):
        if op == 0:
            stack.append(FEATURES[arg])
        elif op == 1:
            value = struct.unpack("<f", struct.pack("<I", program["constant_bits"][arg]))[0]
            stack.append(repr(value))
        elif op == 7:
            stack.append(f"ABS({stack.pop()})")
        else:
            b, a = stack.pop(), stack.pop()
            stack.append(f"{names[op]}({a}, {b})")
    if len(stack) != 1:
        raise ValueError("存档树指令不能归约为一个表达式")
    return stack[0]


def tree_figure(program):
    """从冻结IR绘制真实树结构；短符号与论文terminal表一致。"""
    stack, nodes = [], []
    names = {2: "+", 3: "-", 4: r"\times", 5: r"\min", 6: r"\max", 8: r"\AQ"}
    for op, arg in zip(program["opcode"], program["operand"], strict=True):
        children = []
        if op == 0:
            symbol = SYMBOLS[arg]
        elif op == 1:
            value = struct.unpack("<f", struct.pack("<I", program["constant_bits"][arg]))[0]
            symbol = str(int(value)) if value in [-1, 0, 1] else f"c_{{{arg}}}"
        elif op == 7:
            children, symbol = [stack.pop()], r"\mathrm{abs}"
        else:
            right, left = stack.pop(), stack.pop()
            children, symbol = [left, right], names[op]
        nodes.append({"symbol": symbol, "children": children})
        stack.append(len(nodes) - 1)
    if len(stack) != 1:
        raise ValueError("树图需要一个完整的冻结表达式")
    leaf = 0

    def place(idx, depth):
        nonlocal leaf
        node = nodes[idx]
        if node["children"]:
            for child in node["children"]:
                place(child, depth + 1)
            node["x"] = sum(nodes[c]["x"] for c in node["children"]) / len(node["children"])
        else:
            node["x"] = leaf
            leaf += 1
        node["y"] = -depth

    place(stack[0], 0)
    out = [
        r"\begin{tikzpicture}[x=8mm,y=8mm,every node/.style={font=\small,inner sep=1.5pt}]",
    ]
    # 先画边，再用有底色的节点遮住连接端点，避免线段穿过字符。
    for node in nodes:
        for child in node["children"]:
            c = nodes[child]
            out.append(rf"\draw[gray] ({node['x']},{node['y']}) -- ({c['x']},{c['y']});")
    for node in nodes:
        style = "draw,rounded corners=1pt,fill=white" if node["children"] else "fill=blue!7"
        out.append(rf"\node[{style}] at ({node['x']},{node['y']}) {{$" + node["symbol"] + "$};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def refresh_evidence():
    """仅在整理历史记录时手动使用；之后构建只需版本化结果。"""
    sources = {
        "numeric_selection": "artifacts/v2/protocol/numeric_selection.json",
        "horizon_selection": "artifacts/v2/round-3seed/calibration/summary.json",
    }
    payload = {
        "scope": "Historical development calibration; no new ACO evaluations",
        "source_paths": sources,
        "new_aco_evaluations": 0,
    }
    payload["numeric_selection"] = read(ROOT / sources["numeric_selection"])
    payload["horizon_selection"] = read(ROOT / sources["horizon_selection"])["selection"]
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def figures(summary, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 保存可随论文提交的矢量源码，直接编译不依赖 Python、额外系统字体或临时 PDF。
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.unicode_minus": False,
            "pgf.texsystem": "xelatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[UTF8,scheme=plain,fontset=fandol]{ctex}",
            "savefig.bbox": "tight",
        }
    )
    colors = ["#2166ac", "#b35806"]
    fig, axes = plt.subplots(
        2, 2, figsize=(6.2, 3.9), sharex="col", sharey="row", constrained_layout=True
    )
    for col, rep in enumerate(["joint_single", "conditional_three"]):
        selected = [r for r in summary["runs"] if r["run_id"].startswith(rep)]
        series = [[], []]
        for r in selected:
            train = [
                b["gpu_fixed_mne16_region0_no_restart"] - min(h["population_fitness"])
                for b, h in zip(r["baseline_curve"], r["history"], strict=True)
            ]
            val = [p["gap_percent"] for p in r["validation_curve"]]
            for row, y in enumerate([train, val]):
                x = np.arange(1, 51)
                axes[row, col].plot(x, y, color=colors[col], alpha=0.28, lw=0.7)
                series[row].append(y)
        for row in range(2):
            x = np.arange(1, 51)
            axes[row, col].plot(x, np.mean(series[row], axis=0), color=colors[col], lw=1.4)
            axes[row, col].grid(alpha=0.15)
            axes[row, col].spines[["top", "right"]].set_visible(False)
        axes[0, col].axhline(0, color="gray", lw=0.7, ls="--")
        axes[0, col].set_title(LABELS[rep])
        axes[1, col].set_xlabel("Generation")
    axes[0, 0].set_ylabel("训练改善（百分点）")
    axes[1, 0].set_ylabel("快速验证差距（%）")
    fig.savefig(GEN / "curves.pgf", backend="pgf")
    plt.close(fig)

    # 全部 50 代的绝对训练冠军与种群中位数，供补充材料辨识训练信号。
    fig, axes = plt.subplots(
        2, 3, figsize=(6.2, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, r in zip(axes.flat, summary["runs"], strict=True):
        fitness = [h["population_fitness"] for h in r["history"]]
        ax.plot(range(1, 51), np.median(fitness, axis=1), lw=1, label="种群中位数")
        ax.plot(range(1, 51), np.min(fitness, axis=1), lw=1, label="本代冠军")
        ax.set_title(label(r["run_id"]), fontsize=9)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("进化代数")
    for ax in axes[:, 0]:
        ax.set_ylabel("训练参考差距（%）")
    fig.savefig(GEN / "training_absolute.pgf", backend="pgf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.1), sharey=True, constrained_layout=True)
    for col, rep in enumerate(["joint_single", "conditional_three"]):
        curves = [
            [p["gap_percent"] for p in r["monitor_curve"]]
            for r in analysis["runs"]
            if r["run_id"].startswith(rep)
        ]
        for y in curves:
            axes[col].plot(range(5, 51, 5), y, color=colors[col], alpha=0.3, lw=0.8)
        axes[col].plot(range(5, 51, 5), np.mean(curves, axis=0), color=colors[col], lw=1.4)
        axes[col].set(title=LABELS[rep], xlabel="Generation")
        axes[col].grid(alpha=0.15)
        axes[col].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("开发监控差距（%）")
    fig.savefig(GEN / "monitor.pgf", backend="pgf")
    plt.close(fig)

    # 区间和显著性均读取既有统计，不重新估计或按图中小数重新检验。
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.5), sharey=True, constrained_layout=True)
    for ax, dimension in zip(axes, [500, 1000], strict=True):
        rows = [r for r in analysis["comparisons"] if r["dimension"] == dimension]
        for y, row in enumerate(rows):
            center = row["improvement_percentage_points"]
            lo, hi = row["improvement_ci95"]
            color = colors[0 if row["method"] == "joint_single" else 1]
            ax.errorbar(
                center,
                y,
                xerr=[[center - lo], [hi - center]],
                fmt="o",
                color=color,
                markersize=3.5,
                capsize=2,
                elinewidth=1,
            )
            if dimension == 500 and row["holm_p"] < 0.05:
                ax.annotate(
                    "*",
                    (hi, y),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    color=color,
                    fontsize=11,
                )
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        ax.axhline(3.5, color="0.85", lw=0.6)
        ax.axhline(7.5, color="0.85", lw=0.6)
        ax.set_yticks(
            range(len(rows)), [LABELS[r["method"]] + " / " + LABELS[r["comparator"]] for r in rows]
        )
        ax.set(title=f"TSP{dimension}", xlabel="改善（百分点）", xlim=(-0.043, 0.045))
        ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.grid(axis="x", alpha=0.12)
    axes[0].invert_yaxis()
    fig.savefig(GEN / "effects.pgf", backend="pgf")
    plt.close(fig)


def tables(summary, analysis, evidence):
    comparisons = analysis["comparisons"]
    methods = list(LABELS)
    means = {(c["dimension"], c["method"]): c["method_gap_percent"] for c in comparisons}
    means.update(
        {(c["dimension"], c["comparator"]): c["comparator_gap_percent"] for c in comparisons}
    )
    write_table(
        "quality",
        ["方法", "TSP500", "TSP1000"],
        [[label(m), f"{means[500, m]:.5f}", f"{means[1000, m]:.5f}"] for m in methods],
    )
    for n in [500, 1000]:
        rows = []
        for c in comparisons:
            if c["dimension"] != n:
                continue
            lo, hi = c["improvement_ci95"]
            p = c.get("holm_p")
            ptext = "--" if p is None else (f"{p:.5f}" if p < 0.001 else f"{p:.3f}")
            rows.append(
                [
                    label(c["method"]),
                    label(c["comparator"]),
                    f"{c['improvement_percentage_points']:+.5f}",
                    f"$[{lo:+.5f},{hi:+.5f}]$",
                    ptext,
                ]
            )
        write_table(
            "primary" if n == 500 else "transfer",
            ["方法", "对照", r"$\Delta$", r"95\%区间", r"$p_H$"],
            rows,
            "llrrr",
        )
        wins = [
            [
                label(c["method"]),
                label(c["comparator"]),
                "/".join(map(str, c["wins_ties_losses"])),
                f"{c['relative_route_improvement_percent']:+.5f}",
            ]
            for c in comparisons
            if c["dimension"] == n
        ]
        write_table(f"wins{n}", ["方法", "对照", "胜/平/负", r"路径长度改善（\%）"], wins, "llrr")

    diagnostic = read(BASE / "selected_feedback_diagnostic.json")
    rows = []
    full = []
    for d in diagnostic["runs"]:
        if d["contexts"] == "training_contexts":
            rows.append(
                [label(d["run_id"])]
                + [
                    d["feedback_interventions"][f]["0"]["any"]
                    for f in ["stagnation", "return_rate", "ls_work"]
                ]
            )
        full.append(
            [
                label(d["run_id"]),
                {
                    "training_contexts": "训练64",
                    "explain_n500": "500回放12",
                    "explain_n1000": "1000回放12",
                }[d["contexts"]],
            ]
            + [
                f"{d['feedback_interventions'][f]['0']['any']}/{d['feedback_interventions'][f]['1']['any']}"
                for f in ["stagnation", "return_rate", "ls_work"]
            ]
        )
    write_table("feedback", ["控制器", "停滞", "回归率", "工作量"], rows)
    write_table("feedback_full", ["控制器", "情境", "停滞0/1", "回归0/1", "工作0/1"], full, "llrrr")

    test = {(r["method"], r["dimension"]): r for r in analysis["test"]}
    rows, costs, diversity = [], [], []
    tree_text = []
    for r, ar in zip(summary["runs"], analysis["runs"], strict=True):
        if r["run_id"] != ar["run_id"]:
            raise ValueError("结果与分析运行顺序不一致")
        rid = r["run_id"]
        selected = r["selected"]["controller"]
        rows.append(
            [
                label(rid),
                selected["controller_id"].split("-s")[1].split("-", 1)[1],
                ar["selected_nodes"],
                ar["selected_quick_rank"],
                f"{r['selected']['selected']['gap_percent']:.5f}",
                f"{test[rid, 500]['gap_percent']['mean']:.5f}",
                f"{test[rid, 1000]['gap_percent']['mean']:.5f}",
            ]
        )
        costs.append(
            [
                label(rid),
                f"{ar['training_gpu_minutes']:.2f}",
                f"{ar['monitor_gpu_minutes']:.2f}",
                f"{ar['validation_gpu_minutes']:.2f}",
                f"{ar['training_wall_hours']:.2f}",
                f"{ar['generation_evaluation_wait_minutes']['mean']:.2f}",
            ]
        )
        diversity.append(
            [
                label(rid),
                ar["g50_unique_ir"],
                ar["g50_unique_behavior"],
                f"{ar['g50_max_height_per_individual']['median']:.0f}",
                f"{ar['g50_nodes_per_individual']['median']:.1f}",
                ar["fitness_unique_g1_g10_g50"][-1],
            ]
        )
        tree_text.append(r"\par\noindent\begin{minipage}{\linewidth}")
        tree_text.append(
            r"\subsubsection{"
            + label(rid)
            + "："
            + esc(selected["controller_id"].split("-s")[1].split("-", 1)[1])
            + r"}\leavevmode\par"
        )
        for role, tree in enumerate(selected["trees"]):
            role_name = (
                "联合评分树"
                if len(selected["trees"]) == 1
                else ["参考路径树", "区域树", "强度树"][role]
            )
            tree_name = f"structure_{rid}_{role}"
            (GEN / f"{tree_name}.tex").write_text(tree_figure(tree))
            tree_text.extend(
                [
                    r"\noindent\begin{minipage}{\linewidth}",
                    role_name + r"\par",
                    r"\begin{center}\input{generated/" + tree_name + r".tex}\end{center}",
                ]
            )
            constants = []
            for i, bits in enumerate(tree["constant_bits"]):
                value = struct.unpack("<f", struct.pack("<I", bits))[0]
                if value not in [-1, 0, 1]:
                    constants.append(f"$c_{{{i}}}={value!r}$")
            if constants:
                tree_text.append("ERC：" + "，".join(constants) + r"。\par")
            tree_text.append(r"\end{minipage}\par\medskip")
        tree_text.append(r"\end{minipage}\par\bigskip")
    write_table(
        "selected",
        ["控制器", "个体", "节点", "快排", "完整验证", "测试500", "测试1000"],
        rows,
        "llrrrrr",
    )
    write_table("costs", ["控制器", "训练", "监控", "选模", "墙钟/时", "代评价等待/分"], costs)
    write_table(
        "diversity", ["控制器", "程序", "行为", "高度中位", "节点中位", "适应度"], diversity
    )
    (GEN / "trees.tex").write_text("\n".join(tree_text) + "\n")

    # 时间先在每次运行内累计，再对三个进化seed取均值。
    maincost = []
    for method in ["joint_single", "conditional_three", *list(LABELS)[3:]]:
        gp_runs = [r for r in analysis["runs"] if r["run_id"].startswith(method)]
        test_rows = [
            r
            for r in analysis["test"]
            if r["dimension"] == 500
            and (r["method"] == method or r["method"].startswith(method + "-s"))
        ]
        count = len(test_rows)
        train = f"{sum(r['training_gpu_minutes'] for r in gp_runs) / 3:.2f}" if gp_runs else "--"
        validation_minutes = sum(
            r["monitor_gpu_minutes"] + r["validation_gpu_minutes"] for r in gp_runs
        )
        val = f"{validation_minutes / 3:.2f}" if gp_runs else "--"
        maincost.append(
            [
                LABELS[method],
                train,
                val,
                f"{sum(r['solve_seconds'] for r in test_rows) / count:.1f}",
                f"{sum(r['ls_evaluations_per_tour'] for r in test_rows) / count:.1f}",
            ]
        )
    write_table("maincost", ["方法", "训练/分", "监控与选模/分", "测试/秒", "LS评价/FE"], maincost)

    for n in [500, 1000]:
        rows = []
        for r in analysis["test"]:
            if r["dimension"] != n or "restarts" not in r:
                continue
            v = r["restarts"]
            rows.append(
                [
                    label(r["method"]),
                    f"{v['mean']:.1f}",
                    f"{v['median']:.1f}",
                    f"{v['p90']:.1f}",
                    f"{v['max']:.0f}",
                    r["no_restart_solves"],
                ]
            )
        write_table(f"restarts{n}", ["控制器", "均值", "中位", "P90", "最大", "零重启次数"], rows)
        rows = [
            [
                label(r["method"]),
                f"{r['solve_seconds']:.2f}",
                f"{r.get('construction_steps_per_tour', 0):.2f}",
                f"{r.get('ls_evaluations_per_tour', 0):.2f}",
            ]
            for r in analysis["test"]
            if r["dimension"] == n and r["method"] != "faco_2022_continuous_identity_guard"
        ]
        write_table(f"testcost{n}", ["方法", "总摊销秒", "构造步/FE", "LS评价/FE"], rows)

    tsplib = read(ROOT / "results/v2/faco_2022_tsplib.json")["summary"]
    write_table(
        "tsplib",
        ["实例", "蚂蚁", r"平均差距/\%", r"最佳差距/\%", "平均秒"],
        [
            [
                r["instance"],
                r["ants"],
                f"{r['mean_gap_percent']:.5f}",
                f"{r['best_gap_percent']:.5f}",
                f"{r['mean_seconds']:.3f}",
            ]
            for r in tsplib
        ],
    )
    rows = []
    for h in evidence["horizon_selection"]["rows"]:
        for s in h["scales"]:
            rows.append(
                [
                    s["dimension"],
                    h["iterations"],
                    f"{s['mean_gap_percent']:.5f}",
                    f"{100 * s['retained_improvement']:.2f}",
                    f"{s['spearman']:.4f}",
                    "是" if s["passed"] else "否",
                ]
            )
    write_table("horizon", ["规模", "迭代", r"参考差距/\%", r"保留率/\%", "秩相关", "通过"], rows)
    rows = []
    for variant, d in evidence["numeric_selection"]["candidates"].items():
        for r in d["rows"]:
            rows.append(
                [
                    esc(variant),
                    r["dimension"],
                    "静态区域3" if r["controller"].startswith("static") else "GP代表",
                    f"{r['speedup']:.4f}",
                ]
            )
    write_table("precision", ["数值方案", "规模", "控制器", "相对exact加速比"], rows, "lr lr")
    perf = read(ROOT / "results/v3/representation_performance.json")
    write_table(
        "speed",
        ["样本", "优化前/秒", "优化后/秒", r"时间减少/\%"],
        [
            [
                label(r["case"]),
                f"{r['original_seconds']:.3f}",
                f"{r['optimized_seconds']:.3f}",
                f"{r['time_reduction_percent']:.2f}",
            ]
            for r in perf["same_card_h100"]
        ],
    )
    pilot = read(ROOT / "results/v3/pilot_analysis.json")
    write_table(
        "pilot",
        ["运行", r"开发差距/\%", "行为数", "训练GPU分钟"],
        [
            [
                label(r["run_id"]),
                f"{r['end_gap_percent']:.5f}",
                r["unique_behaviors"],
                f"{r['training_gpu_minutes']:.2f}",
            ]
            for r in pilot["runs"]
        ],
    )
    old = read(ROOT / "results/v3/historical_variation_replay.json")
    write_table(
        "old_variation",
        ["种子", "代际", "新程序/124", "交叉未变/尝试"],
        [
            [
                r["seed"],
                r["transition"].replace("->", r"$\to$"),
                r["novel_IR_children_of_124"],
                f"{r['unchanged_crossovers']}/{r['crossover_attempts']}",
            ]
            for r in old["rows"]
        ],
    )


def decision_examples():
    examples = read(BASE / "decision_examples.json")
    # 主图使用两个已冻结个体各自的真实状态；不把两条轨迹当成同状态消融。
    joint = next(
        v for v in examples["joint_single-s3313-n500"] if v["iteration"] == 100 and v["colony"] == 0
    )
    conditional = next(
        v
        for v in examples["conditional_three-s3313-n500"]
        if v["iteration"] == 100 and v["colony"] == 0
    )
    figure = [
        r"\begin{minipage}[t]{0.39\textwidth}\centering\small",
        r"\textbf{GP-J/3313}\par\medskip",
        r"\input{generated/structure_joint_single-s3313_0.tex}\par\medskip",
        rf"$\rho={joint['features'][2][joint['action']]:.8f}$，$u=1$\par",
        r"保留参考路径时：\par\smallskip",
        r"\begin{tabular}{cr}\toprule 区域$j$ & MNE16得分\\\midrule",
    ]
    for region in range(4):
        figure.append(f"{region} & {joint['scores'][4 * region + 3]:.9f}" + r"\\")
    figure.extend(
        [
            r"\bottomrule\end{tabular}\par\smallskip",
            r"输出：$(r,j,\mathrm{MNE})=(0,3,16)$",
            r"\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.59\textwidth}\centering\small",
            r"\textbf{GP-C/3313}\par\medskip",
            r"$\begin{aligned}",
            r" f_r&=(ws)(s\delta_q)\AQ(r,-1),\\",
            r" f_j&=p+\max(p,a_R\tau_R),\\",
            r" f_k&=u+(p+\rho).",
            r"\end{aligned}$\par\medskip",
            r"$p=0.0198$；依次比较以下候选：\par\smallskip",
        ]
    )
    for role, stage in enumerate(conditional["stages"]):
        heading = [
            r"Reference：$r\in\{0,1\}$",
            r"Region：$j\in\{0,1,2,3\}$",
            r"MNE：$(2,4,8,16)$",
        ][role]
        scores = ",\\;".join(f"{v:.6f}" for v in stage["scores"])
        choice = [r"$r=0$", r"$j=1$", r"$\mathrm{MNE}=16$"][role]
        figure.extend(
            [
                heading + r"\par",
                "$(" + scores + r")$\par",
                r"$\longrightarrow$ " + choice + r"\par\smallskip",
            ]
        )
    figure.extend(
        [
            r"输出：$(r,j,\mathrm{MNE})=(0,1,16)$",
            r"\end{minipage}",
        ]
    )
    (GEN / "decision_figure.tex").write_text("\n".join(figure) + "\n")
    parts = []
    for run in ["joint_single-s3313", "conditional_three-s3313"]:
        d = next(v for v in examples[run + "-n500"] if v["iteration"] == 100 and v["colony"] == 0)
        action = d["action"]
        parts.append(r"\subsection{" + label(run) + "的真实输入与输出}")
        parts.append(
            f"取TSP500解释面板首个实例、ACO种子17、迭代100。所选动作编号为{action}，"
            f"对应$r={action // 16}$、$j={(action % 16) // 4}$、MNE={2 ** (action % 4 + 1)}。"
            "下表为所选完整动作的12个输入；全动作输入矩阵保留在机器记录中。"
        )
        write_table(
            "input_" + run,
            ["输入", "数值"],
            [
                [r"\texttt{" + esc(f) + "}", f"{d['features'][i][action]:.9g}"]
                for i, f in enumerate(FEATURES)
            ],
        )
        parts.append(r"\begin{center}\small\gentable{input_" + run + r"}\end{center}")
        if "stages" in d:
            for role, stage in enumerate(d["stages"]):
                parts.append(
                    ["参考路径", "区域", "强度"][role]
                    + "阶段的分数为 "
                    + ", ".join(f"{v:.9g}" for v in stage["scores"])
                    + f"；选择编号{stage['selected']}。\\par"
                )
        else:
            rows = []
            for idx in range(0, 32, 4):
                rows.append(
                    [idx // 16, (idx % 16) // 4]
                    + [
                        f"{d['scores'][j]:.9g}" if (d["legal_mask"] >> j) & 1 else "--"
                        for j in range(idx, idx + 4)
                    ]
                )
            write_table("scores_" + run, ["$r$", "$j$", "MNE2", "MNE4", "MNE8", "MNE16"], rows)
            parts.append(r"\begin{center}\small\gentable{scores_" + run + r"}\end{center}")
    (GEN / "examples.tex").write_text("\n".join(parts) + "\n")


def compile_papers():
    destination = ROOT / "artifacts/papers/evostar"
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"new_aco_evaluations": 0, "documents": []}
    for name in ["samplepaper", "supplement"]:
        out = ROOT / ".tmp/evostar" / name
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "build.log"
        with log_path.open("w") as log:
            subprocess.run(
                [
                    "latexmk",
                    "-xelatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    f"-outdir={out}",
                    f"{name}.tex",
                ],
                cwd=SOURCE,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        pdf = out / f"{name}.pdf"
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        latex_log = (out / f"{name}.log").read_text()
        problems = [
            line
            for line in latex_log.splitlines()
            if any(
                v in line
                for v in [
                    "Overfull \\hbox",
                    "Overfull \\vbox",
                    "Missing character:",
                    "undefined references",
                    "undefined citations",
                ]
            )
        ]
        if name == "samplepaper" and pages > 14:
            problems.append(f"主稿{pages}页，超过14页（含参考文献）")
        shutil.copy2(pdf, destination / pdf.name)
        receipt["documents"].append(
            {
                "name": name,
                "pages": pages,
                "problems": problems,
                "pdf": str((destination / pdf.name).relative_to(ROOT)),
            }
        )
    (ROOT / "results/v3/evostar/build.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if any(d["problems"] for d in receipt["documents"]):
        raise SystemExit("版面检查未通过；详细日志在 .tmp/evostar。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-evidence", action="store_true", help="从已有本地历史汇总更新精简标定快照"
    )
    parser.add_argument("--figures-only", action="store_true", help="仅生成图表，不编译LaTeX")
    args = parser.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(ROOT / ".tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("TEXMFVAR", str(ROOT / ".cache/texmf-var"))
    if args.refresh_evidence:
        refresh_evidence()
    summary, analysis = read(BASE / "summary.json"), read(BASE / "analysis.json")
    if len(summary["runs"]) != 6 or any(r["completed_generations"] != 50 for r in summary["runs"]):
        raise ValueError("主稿仅适用于冻结的六次完整50代结果")
    tables(summary, analysis, read(EVIDENCE))
    decision_examples()
    figures(summary, analysis)
    if not args.figures_only:
        compile_papers()


if __name__ == "__main__":
    main()
