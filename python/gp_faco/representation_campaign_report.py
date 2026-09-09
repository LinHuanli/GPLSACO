"""完整轮次唯一中文阅读入口；从精简代际记录生成曲线，不重复读取训练tour。"""

import os
import statistics
from datetime import datetime

from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.controller import Controller
from gp_faco.evolution import individual_from_program
from gp_faco.primitives import feature_names, make_primitive_set
from gp_faco.remote import PROJECT
from gp_faco.representation_campaign import RUN_IDS
from gp_faco.representation_pilot import BASELINES

NAMES = {
    BASELINES[0]: "原始FACO2022",
    BASELINES[1]: "GPU FACO MNE8均匀",
    BASELINES[2]: "固定MNE16/region0",
    BASELINES[3]: "GPU MNE16均匀",
}


def publish_report(directory, *, engineering_preview=False):
    config = read(directory / "campaign.json")
    if config["engineering"] and not engineering_preview:
        return
    preview = config["engineering"]
    target = directory / "report" if preview else PROJECT / "results/v3/representation-50gen"
    target.mkdir(parents=True, exist_ok=True)
    status = read(directory / "status.json", {})
    panels = read(directory / "panels.json")
    base_rows = read(directory / "baseline_scores.json", {}).get("rows", [])
    base = {
        (r["method"], r["dimension"], r["instance_id"], r["seed"], r["iterations"]): r
        for r in base_rows
    }
    records = []
    for run in RUN_IDS:
        root = directory / "runs" / run
        progress = read(root / "progress.json", {})
        selected = read(root / "selected_controller.json")
        screen = read(root / "validation_screen.json", {}).get("scores", {})
        history = progress.get("history", [])
        baseline_curve = []
        for m in history:
            panel = panels["training"][m["generation"] - 1]["panels"][0]
            row = {"generation": m["generation"]}
            for method in BASELINES:
                values = [
                    base.get((method, 500, name, seed, config["iterations"]))
                    for name in panel["ids"]
                    for seed in panel["seeds"]
                ]
                row[method] = (
                    statistics.mean(v["gap_percent"] for v in values)
                    if values and all(values)
                    else None
                )
            baseline_curve.append(row)
        # 同IR的冠军可能有不同编号，选模复用后按直接结构映射回代数。
        controller_keys = []
        for m in history:
            key = Controller.from_dict(m["winner"]).key
            if not any(k == key for k, _ in controller_keys):
                controller_keys.append((key, m["winner"]["controller_id"]))
        validation_curve = []
        for m in history:
            identifier = next(
                v for k, v in controller_keys if k == Controller.from_dict(m["winner"]).key
            )
            if identifier in screen:
                validation_curve.append({"generation": m["generation"], **screen[identifier]})
        records.append(
            {
                "run_id": run,
                "status": "selected" if selected else progress.get("status", "waiting"),
                "completed_generations": len(history),
                "history": history,
                "selected": selected,
                "validation_curve": validation_curve,
                "baseline_curve": baseline_curve,
            }
        )
    stats = read(directory / "statistics.json")
    atomic_json(
        target / "summary.json",
        {
            "updated": datetime.now().astimezone().isoformat(),
            "status": status,
            "runs": records,
            "statistics": stats,
            "engineering": preview,
            "test_performed": not preview and stats is not None,
        },
    )
    plot_state = [
        [r["completed_generations"], len(r["validation_curve"]), bool(r["selected"])]
        for r in records
    ]
    if plot_state != read(target / "plot_state.json"):
        os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for i, r in enumerate(records):
            h = r["history"]
            label = (
                r["run_id"].replace("joint_single", "Single").replace("conditional_three", "Three")
            )
            axes[0, 0].plot(
                [m["generation"] for m in h],
                [m["training_gap_percent"] for m in h],
                label=label,
                color=f"C{i}",
            )
            monitor = [m for m in h if "monitor" in m]
            axes[0, 1].plot(
                [m["generation"] for m in monitor],
                [m["monitor"]["gap_percent"] for m in monitor],
                marker="o",
                color=f"C{i}",
            )
            v = r["validation_curve"]
            axes[1, 0].plot(
                [m["generation"] for m in v], [m["gap_percent"] for m in v], color=f"C{i}"
            )
            axes[1, 1].plot(
                [m["generation"] for m in h],
                [m["evaluation_wait_wall_seconds"] / 60 for m in h],
                color=f"C{i}",
            )
        for axis, title in zip(
            axes.flat,
            [
                "Training champion (changing panels)",
                "Fixed development monitor",
                "Fixed validation screen (after training)",
                "Generation evaluation wall minutes",
            ],
            strict=True,
        ):
            axis.set_title(title)
            axis.set_xlabel("Evaluated generation")
            axis.grid(alpha=0.2)
        axes[0, 0].set_ylabel("Reference gap % (lower is better)")
        axes[0, 1].set_ylabel("Reference gap %")
        axes[1, 0].set_ylabel("Reference gap %")
        axes[1, 1].set_ylabel("Minutes")
        axes[0, 0].legend(fontsize=8)
        plot = target / "curves.svg"
        fig.savefig(plot)
        plot.write_text("\n".join(line.rstrip() for line in plot.read_text().splitlines()) + "\n")
        plt.close(fig)
        atomic_json(target / "plot_state.json", plot_state)
    done = sum(r["completed_generations"] for r in records)
    active_gpu = [a for a in status.get("active", {}).values() if a.get("gpu_uuid")]
    estimates = [
        statistics.median(m["costs"]["gpu_seconds"] for m in r["history"][-3:])
        * (config["generations"] - len(r["history"]))
        for r in records
        if r["history"]
    ]
    eta = (
        sum(estimates) / max(1, len(active_gpu)) / 3600
        if estimates and len(estimates) == 6
        else None
    )
    examples_path = target / "decision_examples.json"
    examples = read(examples_path, {})
    for run in RUN_IDS:
        for n in (500, 1000):
            key = f"{run}-n{n}"
            if key not in examples:
                path = directory / "jobs" / f"population-{run}-explain-n{n}-p0000-c000/result.json"
                raw = read(path, {})
                if raw.get("members"):
                    examples[key] = raw["members"][0]["native_result"].get("decisions", [])
    if examples:
        atomic_json(examples_path, examples)
    lines = [
        "# 工程流程报告（不作为科学结果）" if preview else "# 单树与三树：六次50代完整实验",
        "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        f"当前阶段：**{status.get('stage', 'waiting')}**；已完成 "
        f"**{done}/{len(RUN_IDS) * config['generations']}个评价代**；"
        f"正在执行GPU分片：{len(active_gpu)}。",
        "",
        "本报告说明实验做了什么、参数如何设置、现在有什么结果。训练只用TSP500；六个程序选定并冻结后才测试TSP500和TSP1000。",
        "",
        "- [完整参数和评价流程](../experiments/representation_50gen_v3.md)",
        "- [10代预实验结论与加速分析](预实验结论与后续加速.md)",
        "",
        "## 实验怎么设置",
        "",
        "| 项目 | 本轮设置 |",
        "|---|---|",
        "| 训练规模与重复 | TSP500；单树、条件三树各3个GP seed：1103、2207、3313 |",
        f"| GP规模 | 每次{config['population']}个体，重新初始化，"
        f"完整评价{config['generations']}代 |",
        "| GP繁殖 | 4个精英；锦标赛大小4；交叉80%、变异15%、复制5% |",
        "| GP结构 | 初始高度2/3/4分层；每树最高5，总计最多63节点；限制行为重复 |",
        f"| 个体fitness | 同代{config['instances_per_panel']}实例×1 ACO seed，最终路线平均gap% |",
        f"| ACO预算 | 128蚂蚁×{config['iterations']}迭代；按evaluation次数结束 |",
        "| 训练监控 | 每5代，冠军在固定16个开发实例、1 seed上运行H1000 |",
        "| 训练后选模 | 先32验证实例×1 seed×H1000筛选，再前4名用64实例×3 seeds×H5000 |",
        "| 冻结后测试 | TSP500和TSP1000，各128实例×10 ACO seeds×H5000；"
        "比较全部六次进化和四个FACO对照 |",
        "",
        "ACO蚂蚁数沿用FACO2022的规模公式；GP参数保留修复后的预实验设置。"
        "这轮只调整执行效率，详细依据与共用参数见上方协议。",
        "",
        "## 当前进度",
        "",
        "| 运行 | 完成代数 | 最近一代评价分钟 | 最近训练冠军gap% | 最终验证gap% |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in records:
        h = r["history"]
        last = h[-1] if h else None
        elapsed = f"{last['evaluation_wait_wall_seconds'] / 60:.2f}" if last else "待运行"
        gap = f"{last['training_gap_percent']:.5f}" if last else "—"
        selected_gap = (
            f"{r['selected']['selected']['gap_percent']:.5f}" if r["selected"] else "尚未选模"
        )
        lines.append(
            f"| {r['run_id']} | {len(h)}/{config['generations']} | "
            f"{elapsed} | {gap} | {selected_gap} |"
        )
    lines += [
        "",
        "最近训练gap来自不同代的不同面板，不能直接横向比较。开发监控和验证使用固定面板。",
        "",
        "## 曲线",
        "",
        "![训练、开发监控、验证和代时](curves.svg)"
        if preview
        else "![训练、开发监控、验证和代时](../../results/v3/representation-50gen/curves.svg)",
        "",
        "验证曲线在该次50代训练结束后补齐；训练期间每5代仅观察一次开发集冠军。",
        "",
        "## 当前个体与实际决策",
        "",
        "树的输出是候选动作的分数，分数最大者被执行。分数本身不是路线长度，也不是改进百分比。",
        "",
        "| 输入 | 含义 |",
        "|---|---|",
        "| progress | 已用ACO评价次数占本次预算的比例 |",
        "| stagnation | 连续未改进的迭代数，除以32并截断到1 |",
        "| return_rate | 局部搜索后回到来源路线的比例，使用滑动平均 |",
        "| ls_work | 局部搜索工作量的归一化滑动平均 |",
        "| restart | 当前候选动作是否重启，0或1 |",
        "| mne_level | 候选MNE等级：0.25/0.5/0.75/1对应2/4/8/16 |",
        "| ref_gap | 候选来源路线相对算法内部最好路线的差 |",
        "| ref_diff | 重启候选路线与当前路线的边差异 |",
        "| region_excess | 候选区域路线边相对局部距离尺度的超额长度 |",
        "| archive_disagreement | 候选区域的路线边与档案路线的不一致程度 |",
        "| pheromone_strength | 候选区域路线边的信息素强度 |",
        "| region_dispersion | 候选区域路线边连向区域外部的比例 |",
        "",
    ]
    if eta is not None and done < 300:
        lines += [
            f"按近期工作量与当前用卡数估算，训练剩余约{eta:.1f}小时；不含选模与测试，"
            "策略工作量和可用GPU变化后会更新。",
            "",
        ]
    pset = make_primitive_set(feature_spec_id=2)
    for r in records:
        if not r["history"]:
            continue
        c = Controller.from_dict(
            r["selected"]["controller"] if r["selected"] else r["history"][-1]["winner"]
        )
        lines += [
            f"### {r['run_id']}（{'验证选中' if r['selected'] else '最近训练冠军'}）",
            "",
            "```text",
        ]
        roles = ["joint"] if c.representation == "joint_single" else ["restart", "region", "mne"]
        for role, tree in zip(roles, c.trees, strict=True):
            expression = individual_from_program(tree, pset)
            used = sorted(
                {
                    feature_names(2)[arg]
                    for op, arg in zip(tree.opcode, tree.operand, strict=True)
                    if op == 0
                }
            )
            lines += [
                f"{role}: {expression}",
                f"  节点={len(tree.opcode)}，高度={expression.height}，输入={', '.join(used)}",
            ]
        lines += ["```", ""]
        samples = [s for s in examples.get(f"{r['run_id']}-n500", []) if s["colony"] == 0]
        if samples:
            sample = samples[-1]
            action = sample["action"]
            lines += [
                f"实际回放：TSP500第1个解释实例，第{sample['iteration']}次动作前。"
                f"选择restart={action // 16}、region={action % 16 // 4}、MNE={2 << (action % 4)}。",
                "",
                "所选动作的12个输入值（完整候选输入保存在results）：",
                "",
                "```text",
            ]
            lines += [
                f"{name}: {sample['features'][i][action]:.7g}"
                for i, name in enumerate(feature_names(2))
            ]
            lines += ["```", ""]
            if "stages" in sample:
                for role, stage in zip(("restart", "region", "mne"), sample["stages"], strict=True):
                    lines.append(
                        f"- {role}：分数`{stage['scores']}`，合法`{stage['legal']}`，"
                        f"选择{stage['selected']}。"
                    )
            else:
                lines.append(f"32个动作分数：`{sample['scores']}`。")
            lines.append("")
    lines += [
        "单树对32个完整动作打分。三树在同一动作前状态依次决定restart、region、MNE，分别比较2、4、4个条件分数，最后只执行一个动作。",
        "",
        "ref_gap是当前来源路线相对算法内部最好路线的差，输入不包含参考最优值。",
        "",
        "## 与FACO比较",
        "",
    ]
    if stats:
        lines += [
            "正的改善百分点表示GP更好；相对路线改善与gap降幅是不同指标。",
            "",
            "| 规模 | 方法 | 对照 | GP gap% | 对照gap% | 改善百分点 | 95%区间 |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
        for row in stats["comparisons"]:
            lo, hi = row["improvement_ci95"]
            lines.append(
                f"| {row['dimension']} | {row['method']} | "
                f"{NAMES.get(row['comparator'], row['comparator'])} | "
                f"{row['method_gap_percent']:.5f} | {row['comparator_gap_percent']:.5f} | "
                f"{row['improvement_percentage_points']:.5f} | "
                f"[{lo:.5f}, {hi:.5f}] |"
            )
        lines += [
            "",
            "TSP500主比较的路线长度变化与配对检验（九项共同做Holm校正）：",
            "",
            "| 方法 | 对照 | 相对路线改善% | 实例胜/平/负 | Holm p |",
            "|---|---|---:|---|---:|",
        ]
        for row in stats["comparisons"]:
            if row["dimension"] == 500:
                counts = "/".join(map(str, row["wins_ties_losses"]))
                lines.append(
                    f"| {row['method']} | {NAMES.get(row['comparator'], row['comparator'])} | "
                    f"{row['relative_route_improvement_percent']:.5f} | {counts} | "
                    f"{row['holm_p']:.4g} |"
                )
        timings = {}
        for row in stats.get("timings", []):
            if row["computed_pairs"]:
                key = row["dimension"], row["method"]
                total = timings.setdefault(key, [0, 0.0])
                total[0] += row["computed_pairs"]
                total[1] += row["solve_seconds"]
        if timings:
            lines += [
                "",
                "各方法独立批次的求解时间：GPU每批最多32实例-seed组合，CPU原生逐组合求解。"
                "下表摊销秒数反映各自配置的吞吐，不能当作单实例延迟；主机和设备明细保存在统计文件。",
                "",
                "| 规模 | 方法 | 本轮计算组合数 | 累计求解秒 | 每组合摊销秒 |",
                "|---:|---|---:|---:|---:|",
            ]
            for (n, method), (count, seconds) in sorted(timings.items()):
                lines.append(
                    f"| {n} | {NAMES.get(method, method)} | {count} | "
                    f"{seconds:.2f} | {seconds / count:.4f} |"
                )
    else:
        lines += ["正式测试尚未完成。10代开发集成绩见预实验结论，不将它标为本轮测试结果。"]
    lines += [
        "",
        "## 资源、预算与恢复",
        "",
        "训练、监控和选模分别累计GPU求解时间，多卡并行时这些时间之和大于实际墙钟时间。",
        "",
        "| 运行 | 训练GPU分钟 | 监控GPU分钟 | 选模GPU分钟 |",
        "|---|---:|---:|---:|",
    ]
    for r in records:
        training = sum(m["costs"]["gpu_seconds"] for m in r["history"]) / 60
        monitor = (
            sum(m.get("monitor", {}).get("costs", {}).get("gpu_seconds", 0) for m in r["history"])
            / 60
        )
        validation = (
            f"{r['selected']['costs']['gpu_seconds'] / 60:.2f}" if r["selected"] else "待选模"
        )
        lines.append(f"| {r['run_id']} | {training:.2f} | {monitor:.2f} | {validation} |")
    generation_fe = (
        config["population"] * config["instances_per_panel"] * 128 * config["iterations"]
    )
    lines += [
        "",
        f"每代{config['population']}个体×{config['instances_per_panel']}实例×1 ACO seed×128蚂蚁×"
        f"{config['iterations']}迭代，共{generation_fe:,} FE。"
        f"六次完整训练{6 * config['generations'] * generation_fe:,} FE；"
        "baseline、监控、选模、测试分别记账。",
        "每60秒发现全服务器空闲A5000。一张卡一个worker；同代分片可分卡，收齐后才繁殖。没有算法时间上限；没有文件、数据或检查点哈希。",
        f"本轮构建：`{config['build']}`；个体分片：`{config['population_shard_size']}`。",
        "原始路线、日志、检查点位于artifacts；精简统计和图位于results。已完成分片可以直接恢复。",
        "",
    ]
    report_path = (
        target / "工程报告.md" if preview else PROJECT / "docs/reports/单树与三树50代实验.md"
    )
    report_path.write_text("\n".join(lines))
    if preview:
        return
    (PROJECT / "docs/reports/progress.md").write_text(
        "# 当前实验进度\n\n"
        + f"六次50代实验：{status.get('stage', 'waiting')}，已完成{done}/300代。\n\n"
        + "[主报告：参数、曲线、FACO比较和控制器解释](单树与三树50代实验.md)\n"
    )
