#!/usr/bin/env python3
# ruff: noqa: E402
"""汇总已经完成的10代预实验；仅分析development，不访问正式test。"""

import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.representation_pilot import BASELINES, DEFAULT_DIRECTORY, REPRESENTATIONS, SEEDS
from gp_faco.representation_statistics import comparison_statistics


def main():
    root = DEFAULT_DIRECTORY
    status = read(root / "status.json")
    if status["stage"] != "complete" or status["failed"]:
        raise ValueError("预实验尚未全部完成")
    baselines = read(root / "baseline_report_cache.json")["groups"]["end-dev"]
    rows, runs = [], []
    for method, values in baselines.items():
        rows.extend({**r, "method": method, "dimension": 500, "iterations": 5000} for r in values)
    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            run = f"{representation}-s{seed}"
            summary = read(root / "runs" / run / "summary.json")
            history = summary["history"]
            rows.extend({**r, "method": run} for r in summary["end_development"]["rows"])
            last = history[-1]
            raw = read(root / "jobs" / last["costs"]["jobs"][0] / "result.json")
            work = {
                key: sum(m["native_result"][key] for m in raw["members"])
                for key in (
                    "completed_construction_steps",
                    "completed_ls_evaluations",
                    "total_tour_evaluations",
                )
            }
            runs.append(
                {
                    "run_id": run,
                    "representation": representation,
                    "seed": seed,
                    "end_gap_percent": summary["end_development"]["gap_percent"],
                    "training_gpu_minutes": sum(h["costs"]["gpu_seconds"] for h in history) / 60,
                    "generation_seconds": [h["costs"]["gpu_seconds"] for h in history],
                    "work": work,
                    "monitor_curve": [
                        {"generation": h["generation"], "gap_percent": h["monitor"]["gap_percent"]}
                        for h in history
                        if "monitor" in h
                    ],
                    "expressions": last["expressions"],
                    "feedback_changes_of_64": last["feedback_intervention_changed_actions_of_64"],
                    "unique_behaviors": last["diagnostics"]["unique_behavior"],
                    "variation": last["diagnostics"]["variation"],
                }
            )
    panel = read(root / "panels.json")["end_development"]
    stats = comparison_statistics(
        rows,
        {"test_iterations": 5000, "statistics_seed": 91001, "bootstrap_replicates": 10000},
        panel,
    )
    stats.update(scope="exploratory_development_only", test_performed=False)
    # 开发分析只展示区间，不把探索性p值当成预登记正式测试。
    for r in stats["comparisons"]:
        r.pop("wilcoxon_p", None)
        r.pop("holm_p", None)
    stats.pop("primary_family_size", None)
    result = {"runs": runs, "development": stats, "test_performed": False}
    atomic_json(PROJECT / "results/v3/pilot_analysis.json", result)
    names = dict(
        zip(
            BASELINES,
            ["原始FACO2022", "GPU FACO MNE8均匀", "固定MNE16/region0", "GPU MNE16均匀"],
            strict=True,
        )
    )
    lines = [
        "# 10代预实验：结论与后续加速",
        "",
        "六次进化全部完成，无失败。这个实验检验繁殖修复和单树/条件三树表示，尚未进行正式测试。",
        "",
        "## 先看结果",
        "",
        "评价条件完全相同：32个既有development实例×3个ACO seeds×128蚂蚁×5000迭代。",
        "",
        "| 方法 | seed1103 gap% | seed2207 gap% | seed3313 gap% | 均值gap% |",
        "|---|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        values = [r["end_gap_percent"] for r in runs if r["representation"] == representation]
        lines.append(
            f"| {representation} | "
            + " | ".join(f"{v:.5f}" for v in values)
            + f" | {statistics.mean(values):.5f} |"
        )
    for method, values in baselines.items():
        mean = statistics.mean(r["gap_percent"] for r in values)
        lines.append(f"| {names[method]} | — | — | — | {mean:.5f} |")
    lines += [
        "",
        "单树三个seed的均值优于四类对照；三树只优于原始FACO，尚未超过GPU固定策略。这个结果支持把单树作为有竞争力的方案，但不能据此断言50代后的排序或正式测试提升。",
        "",
        "这里的gap下降比例不是路线长度下降比例。例如单树相对原始FACO约少0.06875个百分点gap，不能写成路线缩短34%。",
        "",
        "## 是否真的恢复探索",
        "",
        "末代六个种群的128个体仍有98–112种不同决策行为（在64个固定训练情境上），交叉能够持续改变结构和行为。",
        "",
        "树变复杂本身不是学习成功的证据。对停滞、回报、LS工作这三类反馈分别置1，观察64情境中决策改变次数：",
        "",
        "| 运行 | 停滞干预 | 回报干预 | LS工作干预 | 不同行为数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in runs:
        f = r["feedback_changes_of_64"]
        lines.append(
            f"| {r['run_id']} | {f['1']} | {f['2']} | {f['3']} | {r['unique_behaviors']} |"
        )
    lines += [
        "",
        "单树有冠军对回报干预较敏感；三树冠军在这些情境中改变动作较少。64情境是有限诊断，不能把零变化解释成所有搜索状态都无反馈作用，也不据此强制树使用某个输入。",
        "",
        "## 为什么三树慢",
        "",
        "| 运行 | 10代GPU分钟 | 末代GPU秒 | 末代构造步数 | 末代LS检查量 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in runs:
        lines.append(
            f"| {r['run_id']} | {r['training_gpu_minutes']:.2f} | "
            f"{r['generation_seconds'][-1]:.2f} | {r['work']['completed_construction_steps']:,} | "
            f"{r['work']['completed_ls_evaluations']:,} |"
        )
    lines += [
        "",
        "末代三树平均构造步数比单树多约64%，LS检查量多约30%。GP策略改变了搜索工作量，三树的全部时间差不能归因于表达式评分。",
        "",
        "本次生产布局event回放显示，构造与LS占主要时间；GP评分仅约1–2毫秒/100迭代整种群。后续优化集中在LS前缀扫描、蚂蚁私有标记的共享内存和多卡分片。",
        "",
        "## 最终冠军是什么",
        "",
    ]
    for r in runs:
        lines += [f"### {r['run_id']}", "", "```text", *r["expressions"], "```", ""]
    performance = read(PROJECT / "results/v3/representation_performance.json")
    if performance:
        lines += [
            "## 本轮加速测量",
            "",
            "使用同一张A5000，回放保存的初代/末代128个体×16个TSP500实例×1 seed；"
            "H100热身后测三次取中位数。原构建与新构建的最终路线、外部评分、控制状态和工作量一致。",
            "",
            "| 种群 | 原版秒 | 优化后秒 | 耗时降低 |",
            "|---|---:|---:|---:|",
        ]
        for row in performance["same_card_h100"]:
            lines.append(
                f"| {row['case']} | {row['original_seconds']:.2f} | "
                f"{row['optimized_seconds']:.2f} | {row['time_reduction_percent']:.1f}% |"
            )
        lines += [
            "",
            "保留无约束LS候选前缀并行、visited/queued标记共享内存，选择每block两只蚂蚁。"
            "2/4/8 warp布局已比较；2 warp在四组负载上均最快。"
            "这里是整组执行耗时，不能把它除以个体数称为单个求解延迟。",
            "",
            "六张A5000共享六个真实末代种群，H100下各分片热身后三次的总完成时间：",
            "",
            "| 每片个体数 | 六个种群完成秒数 |",
            "|---:|---:|",
        ]
        for chunk, seconds in sorted(
            performance["six_gpu_shard_h100"]["median_seconds"].items(),
            key=lambda item: int(item[0]),
        ):
            lines.append(f"| {chunk} | {seconds:.2f} |")
        full = performance["six_gpu_full_h1000"]["measurements"][0]
        lines += [
            "",
            f"据此选择{performance['shard_size']}个体一片。随后用完整H1000核对全部六个末代种群，"
            f"共{full['total_fe']:,} FE，结果和工作量与原预实验完全一致。累计GPU求解"
            f"{full['gpu_seconds'] / 60:.2f}分钟，"
            f"六个种群总完成时间{full['wall_seconds'] / 60:.2f}分钟。"
            "这次总时间含worker启动与资源发现；原预实验在不同主机运行，"
            "不据此宣称精确的完整预算加速比例。",
            "",
            "通过20项C++/CUDA检查、303项Python测试；另有8项按现有条件跳过。"
            "小规模连续与暂停恢复两次完整流程均完成131个任务，种群fitness、冠军、选模和GP测试结果一致，GPU分片没有重复求解。",
            "长任务另外暴露了共享文件系统的运行记录可见性问题：已改为派单前由协调器写入进程记录，"
            "缺少远端记录时查常驻worker租约，并直接打开消息文件以避免旧的exists结果。"
            "第一次长任务检查中两个分片被重复求解，该轮不用于耗时或无重复执行结论。"
            "修复后重新完成一轮131任务的小流程，以及六个种群的完整H1000回放；新长任务的12个分片都仅实际求解一次。",
            "",
            "[精简测量](../../results/v3/representation_performance.json) · "
            "[完整检查记录](../../results/v3/representation_full_verification.json)",
            "",
        ]
    lines += [
        "单树给32个动作评分。三树的三个表达式依次负责restart、region、MNE；"
        "实际12个输入和条件输出见[预实验完整报告](单树与三树预实验.md)。",
        "",
        "## 下一轮怎样做",
        "",
        "两种表示各3seed，重新初始化训练50代；保留相同GP/ACO参数和evaluation预算。"
        "使用测量选定的构建与分片，通过新旧结果对比及小规模完整流程后，自动发现全部空闲A5000。",
        "",
        "[50代协议](../experiments/representation_50gen_v3.md) · "
        "[50代主报告](单树与三树50代实验.md)",
        "",
    ]
    (PROJECT / "docs/reports/预实验结论与后续加速.md").write_text("\n".join(lines))
    print(json.dumps({"status": "complete", "runs": len(runs), "test_performed": False}))


if __name__ == "__main__":
    main()
