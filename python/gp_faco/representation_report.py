"""单一中文入口：六次进化曲线、对照和树解释；原始tour只留在artifacts。"""

import os
from datetime import datetime

import numpy as np

from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.primitives import feature_names
from gp_faco.remote import PROJECT
from gp_faco.representation_pilot import BASELINES, REPRESENTATIONS, SEEDS


def publish_report(directory, summaries):
    output = PROJECT / "results/v3/representation-pilot"
    output.mkdir(parents=True, exist_ok=True)
    status = read(directory / "status.json", {})
    cache_path = directory / "baseline_report_cache.json"
    baselines = read(cache_path, {"completed_jobs": [], "groups": {}})
    for prefix in ("baseline_gpu-", "baseline_native-"):
        for path in (directory / "jobs").glob(prefix + "*/result.json"):
            if path.parent.name in baselines["completed_jobs"]:
                continue
            job = read(path.parent / "job.json")
            if job.get("role") != "baseline":
                continue
            result = read(path)
            group = baselines["groups"].setdefault(job["group"], {})
            rows = group.setdefault(job["method"], [])
            rows.extend(
                {k: r[k] for k in ("instance_id", "seed", "gap_percent", "cost")}
                for r in result["rows"]
            )
            baselines["completed_jobs"].append(path.parent.name)
    atomic_json(cache_path, baselines)
    records = []
    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            run_id = f"{representation}-s{seed}"
            summary = next((s for s in summaries if s.get("run_id") == run_id), {})
            history = summary.get("history", [])
            records.append(
                {
                    "run_id": run_id,
                    "representation": representation,
                    "seed": seed,
                    "status": summary.get("status", "waiting"),
                    "completed_generations": len(history),
                    "training_curve": [
                        {"generation": m["generation"], "gap_percent": m["training_gap_percent"]}
                        for m in history
                    ],
                    "monitor_curve": [
                        {"generation": m["generation"], "gap_percent": m["monitor"]["gap_percent"]}
                        for m in history
                        if "monitor" in m
                    ],
                    "end_gap_percent": summary.get("end_development", {}).get("gap_percent"),
                    "latest": history[-1] if history else None,
                }
            )
    means = {
        group: {
            method: {
                "count": len(rows),
                "mean_gap_percent": float(np.mean([r["gap_percent"] for r in rows])),
            }
            for method, rows in methods.items()
        }
        for group, methods in baselines["groups"].items()
    }
    atomic_json(
        output / "summary.json",
        {
            "status": status,
            "runs": records,
            "baselines": means,
            "test_performed": False,
            "updated": datetime.now().astimezone().isoformat(),
        },
    )
    # 普通计数判断是否有新曲线点，不计算数据摘要或遍历训练原始结果。
    plot_state = [
        [r["completed_generations"], len(r["monitor_curve"]), r["end_gap_percent"]] for r in records
    ]
    plot_state.append(len(baselines["completed_jobs"]))
    if plot_state != read(output / "plot_state.json"):
        os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for i, record in enumerate(records):
            kind = "Single" if record["representation"] == "joint_single" else "Three"
            label = f"{kind} / {record['seed']}"
            for axis, key in zip(axes, ("training_curve", "monitor_curve"), strict=True):
                values = record[key]
                if values:
                    axis.plot(
                        [p["generation"] for p in values],
                        [p["gap_percent"] for p in values],
                        marker="o",
                        markersize=3,
                        label=label,
                        color=f"C{i}",
                    )
        for axis, title in zip(
            axes,
            ("Training: changing shared panels", "Development monitor: fixed panel"),
            strict=True,
        ):
            axis.set(
                title=title,
                xlabel="Evaluated generation",
                ylabel="Reference gap (%)",
                xlim=(0.7, 10.3),
            )
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=8)
        fig.savefig(output / "curves.svg")
        plt.close(fig)
        atomic_json(output / "plot_state.json", plot_state)

    def fmt(x):
        return "—" if x is None else f"{x:.5f}%"

    lines = [
        "# 单树与三树预实验",
        "",
        f"更新于 {datetime.now().astimezone().isoformat(timespec='seconds')}。"
        f"当前阶段：`{status.get('stage', 'preparing')}`。",
        "",
        "本轮比较共同修复繁殖后的联合单树和条件三树，各3 seed×10代，只训练TSP500。"
        "每代128个体，各跑16实例×1个ACO seed×128蚂蚁×1000迭代。"
        "旧50代实验已暂停；本轮结束后不会自动进入50代、正式val或test。",
        "",
        "g5/g10在16个固定开发实例上监控；每次只把g10冠军放到32开发实例×3 ACO seeds×H5000上比较。"
        "下表gap越小越好；训练面板逐代变化，因此不能把训练曲线的每次下降都解释为进化改进。",
        "",
        "| 表示 | GP seed | 已评价代数 | 最近训练gap | 最近开发监控gap | 结束开发gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['representation']} | {r['seed']} | {r['completed_generations']}/10 | "
            f"{fmt(r['training_curve'][-1]['gap_percent'] if r['training_curve'] else None)} | "
            f"{fmt(r['monitor_curve'][-1]['gap_percent'] if r['monitor_curve'] else None)} | "
            f"{fmt(r['end_gap_percent'])} |"
        )
    lines += [
        "",
        "![六条训练与开发曲线](../../results/v3/representation-pilot/curves.svg)",
        "",
        "## 与FACO和固定MNE16比较",
        "",
        "原生FACO2022与GPU均匀起点MNE8是主要对照。固定region0 MNE16对应旧冠军`mne_level`；"
        "均匀起点MNE16用于分开MNE增大和起始区域限制的影响。结果未齐时不计算部分面板的提升。",
        "",
        "| 结束开发对照 | 已返回实例/seed数 | 平均gap |",
        "|---|---:|---:|",
    ]
    for method in BASELINES:
        baseline = means.get("end-dev", {}).get(method, {})
        count = baseline.get("count", 0)
        lines.append(
            f"| {method} | {count}/96 | "
            f"{fmt(baseline.get('mean_gap_percent') if count == 96 else None)} |"
        )
    for r in records:
        if r["end_gap_percent"] is not None:
            comparisons = []
            for method in BASELINES:
                baseline = means.get("end-dev", {}).get(method, {})
                if baseline.get("count") == 96:
                    delta = r["end_gap_percent"] - baseline["mean_gap_percent"]
                    comparisons.append(f"相对`{method}`：gap差{delta:+.5f}个百分点")
            if comparisons:
                lines += ["", f"`{r['run_id']}`：" + "；".join(comparisons) + "。负数表示更好。"]
    lines += [
        "",
        "## 繁殖是否真的改变了个体",
        "",
        "旧第9→10代的48/49/50次交叉全部未改变结构；根节点不可交叉、单叶父代不参与交叉、"
        "同分时优先短树共同放大了复制。当前根节点可选，同分父代随机；采用互斥80%交叉、15%变异、5%复制。"
        "同代IR全部不同，相同行为最多2个。以下行为统计来自固定64个训练情境，不是全部ACO轨迹。",
        "",
        "| run | 唯一行为/128 | 最大同行为组 | 交叉尝试/改结构/改行为 | 繁殖秒数 |",
        "|---|---:|---:|---|---:|",
    ]
    for r in records:
        if not r["latest"]:
            continue
        d = r["latest"]["diagnostics"]
        v = d["variation"]
        lines.append(
            f"| {r['run_id']} | {d['unique_behavior']} | {d['max_behavior_group']} | "
            f"{v.get('crossover_attempted', 0)}/{v.get('crossover_structure_changed', 0)}/"
            f"{v.get('crossover_behavior_changed', 0)} | {d['breeding_seconds']:.3f} |"
        )
    lines += [
        "",
        "初始化高度按边数为2/3/4；进化后单叶高度0仍合法。每树高度≤5，整个个体总节点≤63。"
        "更深的树不自动意味着更好，需同时观察探索和最终性能。",
        "",
        "## 每次进化当前的冠军",
        "",
        "单树输出32个完整动作的相对分数。三树按restart→region→MNE输出2/4/4个条件分数，"
        "使用同一个动作前状态，最后才执行一次动作。分数不是概率。",
    ]
    examples_path = output / "decision_examples.json"
    examples = read(examples_path, {})
    for r in records:
        m = r["latest"]
        if not m:
            continue
        lines += ["", f"### {r['run_id']}，第{m['generation']}代", "", "```text"]
        roles = (
            ("joint",) if r["representation"] == "joint_single" else ("restart", "region", "mne")
        )
        lines += [
            f"{role}: {expression}"
            for role, expression in zip(roles, m["expressions"], strict=True)
        ]
        lines += [
            "```",
            "",
            f"反馈分别替换为1后，64个情境中的动作改变数（1=stagnation，2=return_rate，3=ls_work）："
            f"`{m.get('feedback_intervention_changed_actions_of_64')}`。共同加上的反馈量可能不改变排序。",
            "",
            f"全种群在固定情境上的动作频数：`{m['diagnostics']['context_action_counts']}`；"
            "restart为保持/重启，region为0/1/2/3，mne依次为2/4/8/16。",
        ]
        if r["end_gap_percent"] is not None and r["run_id"] not in examples:
            path = (
                directory
                / "jobs"
                / (f"population-{r['run_id']}-end_development-g10-p000-c000")
                / "result.json"
            )
            raw = read(path, {})
            members = raw.get("members", [])
            if members:
                rows = members[0]["native_result"].get("decisions", [])
                rows = [row for row in rows if row["colony"] == 0]
                if rows:
                    examples[r["run_id"]] = rows[-1]
        example = examples.get(r["run_id"])
        if example:
            action = example["action"]
            features = {
                name: example["features"][i][action] for i, name in enumerate(feature_names(2))
            }
            lines += [
                "",
                f"真实输入输出示例：结束开发面板第1个colony、"
                f"第{example['iteration']}次动作前，合法mask为`{example['legal_mask']:#010x}`。"
                f"最后选择动作{action}：restart={action // 16}，region={action % 16 // 4}，"
                f"MNE={2 << (action % 4)}。",
                "",
                "所选完整动作对应的12个输入值：",
                "",
                "```text",
                *[f"{name}: {value:.7g}" for name, value in features.items()],
                "```",
                "",
            ]
            if "stages" in example:
                for role, stage in zip(
                    ("restart", "region", "mne"), example["stages"], strict=True
                ):
                    lines += [
                        f"- {role}分数：`{[round(v, 7) for v in stage['scores']]}`；"
                        f"合法：`{stage['legal']}`；选择索引：{stage['selected']}。"
                    ]
            else:
                lines += [f"32个联合分数：`{[round(v, 7) for v in example['scores']]}`。"]
    atomic_json(examples_path, examples)
    lines += [
        "",
        "## 运行和证据",
        "",
        "完整参数、FE账目、特征含义和论文来源见[预实验协议](../experiments/representation_pilot_v3.md)。"
        "工程检查、近并列数值差异和测速见`results/v3/representation_cuda.json`；"
        "训练与结束开发比较的原始输入输出在`artifacts/v3/gp-representation-pilot/jobs/`。"
        "结束开发任务保存真实决策轨迹：单树32分、三树2/4/4分，附12个特征和合法mask。",
        "",
        "行为描述使用与训练完全相同的CUDA评分器，不做内容哈希，不缓存替代任何真实ACO评价。"
        "参考最优值只用于外部评分，不进入GP输入。"
        "3 seed结果用于判断是否值得扩大实验，不能保证统计显著。",
        "",
    ]
    (PROJECT / "docs/reports/单树与三树预实验.md").write_text("\n".join(lines))
    (PROJECT / "docs/reports/progress.md").write_text(
        "# 当前进度\n\n" + lines[2] + "\n\n"
        "正在执行单树/条件三树各3 seed×10代的预实验。详见[可读报告](单树与三树预实验.md)与"
        "[参数协议](../experiments/representation_pilot_v3.md)。旧50代实验已暂停；正式val/test未开展。\n"
    )
