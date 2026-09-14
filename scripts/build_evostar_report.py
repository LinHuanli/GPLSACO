#!/usr/bin/env python3
"""从冻结精简结果生成中文论文图表和 PDF；不启动实验、不计算文件摘要。"""

from __future__ import annotations

import argparse
import json
import math
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

    # 全部 50 代的训练最优值与种群中位数，供补充材料辨识训练信号。
    fig, axes = plt.subplots(
        2, 3, figsize=(6.2, 4.0), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, r in zip(axes.flat, summary["runs"], strict=True):
        fitness = [h["population_fitness"] for h in r["history"]]
        ax.plot(range(1, 51), np.median(fitness, axis=1), lw=1, label="种群中位数")
        ax.plot(range(1, 51), np.min(fitness, axis=1), lw=1, label="本代最优")
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
    axes[0].set_ylabel("监控验证差距（%）")
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
    write_table("feedback", ["GP策略", "停滞", "回归率", "工作量"], rows)
    write_table("feedback_full", ["GP策略", "情境", "停滞0/1", "回归0/1", "工作0/1"], full, "llrrr")

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
        ["进化运行", "个体", "节点", "Quick排名", "Full验证", "测试500", "测试1000"],
        rows,
        "llrrrrr",
    )
    write_table("costs", ["进化运行", "训练", "监控", "选模", "墙钟/时", "代评价等待/分"], costs)
    write_table(
        "diversity", ["进化运行", "程序", "行为", "高度中位", "节点中位", "适应度"], diversity
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
        write_table(f"restarts{n}", ["GP策略", "均值", "中位", "P90", "最大", "零重启次数"], rows)
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
    write_table("precision", ["数值方案", "规模", "搜索策略", "相对exact加速比"], rows, "lr lr")
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
        ["运行", r"预实验差距/\%", "行为数", "训练GPU分钟"],
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


def control_mechanism_figure():
    """用固定小图说明控制位置；示意路径不冒充FACO运行轨迹。"""
    # 不规则散点同时包含内部与边界城市；所有子图保持坐标、比例和编号一致。
    cities = [
        (0.7, 0.8),
        (3.5, 1.2),
        (6.0, 0.5),
        (9.2, 1.8),
        (7.6, 4.3),
        (9.0, 8.7),
        (5.3, 7.8),
        (5.0, 4.9),
        (2.6, 6.2),
        (0.8, 9.0),
        (0.4, 4.7),
        (3.0, 3.5),
    ]
    # 已核对的示意路径：参考路径是非全局最优的2-opt局部最优。
    # 弱扰动将城市8移至城市5之后；较强扰动依次将城市6移至5之后、9移至7之后。
    # 两种候选分别经一次改进2-opt返回reference和到达improved。
    reference = [0, 1, 2, 3, 4, 11, 7, 5, 6, 9, 8, 10]
    alternative = [0, 1, 11, 8, 7, 2, 3, 4, 5, 6, 9, 10]
    weak = [0, 1, 2, 3, 4, 7, 11, 5, 6, 9, 8, 10]
    strong = [0, 1, 2, 3, 4, 5, 11, 7, 6, 8, 9, 10]
    improved = [0, 1, 2, 3, 4, 5, 6, 7, 11, 8, 9, 10]
    region = [4, 11, 7, 5]
    start_city = 4
    # 编号置于小节点旁，避免候选边6--12被非端点城市8的大圆标记遮住。
    label_offsets = [
        (-1.8, -1.6),
        (0, -2),
        (0.8, -2),
        (2, -0.6),
        (2, 0.7),
        (1.8, 1.6),
        (0, 2),
        (2, 0),
        (-2, 0.6),
        (-1.8, 1.6),
        (-2, 0),
        (-1.2, -1.9),
    ]

    def length(order):
        """仅计算示意图的欧氏长度，不调用求解器或实验评价器。"""
        return sum(
            math.dist(cities[a], cities[b])
            for a, b in zip(order, order[1:] + order[:1], strict=True)
        )

    reference_cost, weak_cost, strong_cost, improved_cost = map(
        length, [reference, weak, strong, improved]
    )
    improvement_percent = 100 * (reference_cost - improved_cost) / reference_cost
    out = [
        r"\begin{tikzpicture}[x=1mm,y=1mm,>=Stealth,",
        r" every node/.style={font=\scriptsize,inner sep=1pt},",
        r" role/.style={draw,fill=blue!5,align=center,text width=25mm,minimum height=19mm},",
        r" city/.style={fill=white,inner sep=0.2pt,",
        r" font=\fontsize{6}{6}\selectfont}]",
    ]

    def tour(order, x, y, *, previous=None, base=None, color="orange!85!black", marked=False):
        # 逐边直接比较，不计算路径摘要；所有示意图共用城市及坐标。
        previous_edges = (
            [sorted((a, b)) for a, b in zip(previous, previous[1:] + previous[:1], strict=True)]
            if previous is not None
            else []
        )
        base_edges = (
            [sorted((a, b)) for a, b in zip(base, base[1:] + base[:1], strict=True)]
            if base is not None
            else []
        )
        coords = [(x + 2.9 * (cx - 4.8), y + 2.9 * (cy - 4.75)) for cx, cy in cities]
        for a, b in zip(order, order[1:] + order[:1], strict=True):
            changed = previous is not None and sorted((a, b)) not in previous_edges
            style = f"{color},thick,{'densely dashed' if color.startswith('orange') else 'dotted'}"
            style = style if changed else "gray!65,thin"
            if not changed and base is not None and sorted((a, b)) not in base_edges:
                # 最终路径中仍保留的扰动新边继续显示为橙色，便于与参考路径比较。
                style = "orange!85!black,thick,densely dashed"
            xa, ya = coords[a]
            xb, yb = coords[b]
            out.append(rf"\draw[{style}] ({xa:.3f},{ya:.3f})--({xb:.3f},{yb:.3f});")
        for i, (nx, ny) in enumerate(coords):
            style, text_color = "draw=black,fill=black", "black"
            if marked and i in region:
                style = "draw=blue!75!black,fill=blue!25"
                text_color = "blue!75!black"
            out.append(rf"\draw[{style}] ({nx:.3f},{ny:.3f}) circle (0.4);")
            lx, ly = label_offsets[i]
            out.append(
                rf"\node[city,text={text_color}] at ({nx + lx:.3f},{ny + ly:.3f}) {{{i + 1}}};"
            )
        if marked:
            nx, ny = coords[start_city]
            out.append(rf"\draw[blue!75!black,thick] ({nx:.3f},{ny:.3f}) circle (1.0);")
            out.append(
                rf"\draw[->,blue!75!black] ({nx + 6:.3f},{ny + 4:.3f})"
                rf"--({nx + 1.6:.3f},{ny + 1:.3f});"
            )

    out.extend(
        [
            r"\node[role] (r) at (13,0) {$f_r$：参考路径\\$r=0$保留；$r=1$切换};",
            r"\draw[->] (r.east)--(33,0);",
            r"\draw[blue!75!black,rounded corners] (34,-16) rectangle (66,15);",
        ]
    )
    tour(reference, 50, 0)
    tour(alternative, 96, 0)
    out.extend(
        [
            r"\node at (50,-18) {活动路径$q_0$（保留）};",
            r"\node at (96,-18) {备选路径$q_1$};",
            r"\draw[gray!35] (0,-22)--(120,-22);",
            r"\node[role] (j) at (13,-40) {$f_j$：起点区域\\比较$j=0,1,2,3$};",
            r"\draw[->] (j.east)--(33,-40);",
            r"\node[align=left,text width=37mm] at (97,-40) "
            r"{选中$R_1=\{5,12,8,6\}$\\采样起点：城市5（蓝圈）\\后续修改可超出该集合。};",
        ]
    )
    tour(reference, 50, -40, marked=True)
    out.extend(
        [
            r"\node at (50,-58) {蓝色节点：起点集合};",
            r"\draw[gray!35] (0,-61)--(120,-61);",
            r"\node[role] (k) at (13,-78) {$f_k$：扰动强度\\MNE：2、4、8、16\\构造新边的阈值};",
            r"\draw[->] (k.east)--(33,-78);",
        ]
    )
    tour(weak, 50, -78, previous=reference)
    tour(strong, 96, -78, previous=reference)
    out.extend(
        [
            rf"\node[align=center] at (50,-97) {{较弱扰动：{weak_cost:.4f}\\"
            rf"LS后：{reference_cost:.4f}（返回$q$）}};",
            rf"\node[align=center] at (96,-97) {{较强扰动：{strong_cost:.4f}\\"
            rf"LS后：{improved_cost:.4f}（得到更短路径）}};",
            r"\draw[gray!35] (0,-103)--(120,-103);",
            r"\node[anchor=west] at (0,-108) "
            r"{成功扰动示例：先生成较长候选，再由local search找到更短路径};",
        ]
    )
    tour(reference, 16, -126, marked=True)
    tour(strong, 60, -126, previous=reference)
    tour(improved, 104, -126, previous=strong, base=reference, color="green!45!black")
    out.extend(
        [
            r"\draw[->] (31,-131)--node[above]{示意扰动}(44,-131);",
            r"\draw[->] (75,-131)--node[above,align=center]{改进\\2-opt}(88,-131);",
            rf"\node[align=center] at (16,-146) {{2-opt局部最优$q$\\$C(q)={reference_cost:.4f}$}};",
            rf"\node[align=center] at (60,-146) {{扰动后$y$（暂时变差）\\"
            rf"$C(y)={strong_cost:.4f}$}};",
            rf"\node[align=center] at (104,-146) {{LS后$z$：$C(z)={improved_cost:.4f}$\\"
            rf"\textbf{{较$q$缩短{improvement_percent:.2f}\%}}}};",
            r"\draw[gray!65] (5,-154)--(12,-154);\node[anchor=west] at (14,-154) {保留边};",
            r"\draw[orange!85!black,thick,densely dashed] (38,-154)--(45,-154);"
            r"\node[anchor=west] at (47,-154) {扰动新边};",
            r"\draw[green!45!black,thick,dotted] (80,-154)--(87,-154);"
            r"\node[anchor=west] at (89,-154) {LS新边};",
            r"\end{tikzpicture}",
        ]
    )
    (GEN / "control_mechanism.tex").write_text("\n".join(out) + "\n")


def conditional_decision_figure(record):
    """逐角色展示冻结树、已保存评分及动作含义，不重新求值。"""
    features = record["features"]
    action = record["action"]
    r, j, k = [stage["selected"] for stage in record["stages"]]
    out = [r"\small"]
    headings = [
        r"$f_r$：Reference selection",
        r"$f_j$：Starting-region selection",
        r"$f_k$：Perturbation-intensity selection",
    ]
    for role, stage in enumerate(record["stages"]):
        out.extend(
            [
                r"\noindent\textbf{" + headings[role] + r"}\par\smallskip",
                r"\begin{minipage}[c]{0.42\textwidth}\centering",
                rf"\input{{generated/structure_conditional_three-s3313_{role}.tex}}",
                r"\end{minipage}\hfill",
                r"\begin{minipage}[c]{0.55\textwidth}\small",
            ]
        )
        if role == 0:
            out.extend(
                [
                    rf"$w={features[3][action]:.6f}$，$s={features[1][action]:.5f}$。\par",
                    "备选参考路径的相对差距 " + rf"$\delta_q={features[6][16]:.6f}$。\par",
                    r"\begin{tabular}{@{}lr@{}}\toprule 候选 & Priority score\\\midrule",
                ]
            )
            choices = ["保留活动路径", "切换备选路径"]
        elif role == 1:
            out.extend(
                [
                    rf"$p={features[0][action]:.4f}$；"
                    rf"各区域$\tau_R\approx {features[10][action]:.5f}$。\par",
                    r"\begin{tabular}{@{}crr@{}}\toprule 区域 & $a_R$ & Priority score\\\midrule",
                ]
            )
            choices = [f"{idx} & {features[9][16 * r + 4 * idx]:.5f}" for idx in range(4)]
        else:
            out.extend(
                [
                    rf"$p={features[0][action]:.4f}$，$\rho={features[2][action]:.6f}$。\par",
                    r"\begin{tabular}{@{}crr@{}}\toprule MNE & $u$ & Priority score\\\midrule",
                ]
            )
            choices = [f"{2 ** (idx + 1)} & {(idx + 1) / 4:.2f}" for idx in range(4)]
        for idx, score in enumerate(stage["scores"]):
            shown = f"{score:.6f}" if stage["legal"][idx] else "--"
            if idx == stage["selected"]:
                shown = r"\mathbf{" + shown + "}"
            out.append(choices[idx] + " & $" + shown + r"$\\")
        out.extend([r"\bottomrule\end{tabular}\par\smallskip"])
        explanation = [
            rf"$\Rightarrow r={r}$：并列时保留当前参考路径。",
            rf"$\Rightarrow j={j}$：$a_R\tau_R$超过$p$，区域1获得最高分。",
            rf"$\Rightarrow\mathrm{{MNE}}={2 ** (k + 1)}$：公共项$p+\rho$不改变$u$的排序。",
        ][role]
        out.extend([explanation, r"\end{minipage}\par\medskip"])
        if role < 2:
            out.append(r"\noindent\rule{\textwidth}{0.2pt}\par\medskip")
    out.append(rf"\centering\textbf{{最终动作：保留当前参考路径＋区域{j}＋MNE{2 ** (k + 1)}}}\par")
    (GEN / "conditional_decision_figure.tex").write_text("\n".join(out) + "\n")


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
        r"\begin{minipage}[c]{0.38\textwidth}\centering\small",
        r"\textbf{GP-J/3313}\par\medskip",
        r"\input{generated/structure_joint_single-s3313_0.tex}\par\medskip",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[c]{0.59\textwidth}\centering\small",
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
            r"\end{minipage}",
        ]
    )
    (GEN / "decision_figure.tex").write_text("\n".join(figure) + "\n")
    conditional_decision_figure(conditional)
    parts = []
    for run in ["joint_single-s3313", "conditional_three-s3313"]:
        d = next(v for v in examples[run + "-n500"] if v["iteration"] == 100 and v["colony"] == 0)
        action = d["action"]
        parts.append(r"\subsection{" + label(run) + "的真实输入与输出}")
        parts.append(
            f"取TSP500规则解释实例集中的首个实例、ACO种子17、迭代100。所选动作编号为{action}，"
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
                    "Float too large",
                    "Missing character:",
                    "undefined references",
                    "undefined citations",
                ]
            )
        ]
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
        "--refresh-evidence", action="store_true", help="从已有本地历史汇总更新精简预实验快照"
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
    control_mechanism_figure()
    decision_examples()
    figures(summary, analysis)
    if not args.figures_only:
        compile_papers()


if __name__ == "__main__":
    main()
