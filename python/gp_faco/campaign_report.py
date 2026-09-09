"""中文实验报告、可导出曲线和真实 GP 输入输出展示；只读取现有评分。"""

from __future__ import annotations

import ast
import csv
import html
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from gp_faco.campaign import GPU_BASELINE, NATIVE_BASELINE, read, training_directory
from gp_faco.checkpoint import atomic_json, load_checkpoint
from gp_faco.primitives import feature_names
from gp_faco.remote import PROJECT

LABELS = {
    NATIVE_BASELINE: "FACO 2022 连续距离适配",
    GPU_BASELINE: "GPU FACO（无 GP）",
}
JOB_LABELS = {
    "calibration": "开发预算标定分片",
    "baseline_gpu": "训练前 GPU FACO 对照",
    "baseline_native": "训练前作者 FACO 对照",
    "training": "GP 进化",
    "population": "跨GPU种群分片",
    "validation_quick": "轻量验证初筛",
    "validation_final": "入围候选最终验证",
    "validation": "完整验证分片",
    "test_gpu": "三棵 GP 与 GPU FACO 测试",
    "test_native": "作者 FACO 测试",
    "explain": "真实输入输出回放",
}
FEATURE_HELP = [
    "已完成评价次数占总预算的比例",
    "连续未改善批次数，32 批封顶后归一化",
    "局部搜索返回当前参考路线的比例 EMA",
    "局部搜索工作量 EMA",
    "候选动作是否重启（0/1）",
    "候选 MNE 的四级编码：0.25/0.5/0.75/1 对应 2/4/8/16",
    "候选参考路线相对在线最优路线的归一化差距；不读取最优标签",
    "候选参考与当前参考的采样结构差异",
    "候选区域的边长超额",
    "候选区域相对档案的结构分歧",
    "候选区域的信息素强度",
    "候选区域的空间分散程度",
]


def expression_depth(expression):
    def depth(node):
        return 1 + max(map(depth, node.args)) if isinstance(node, ast.Call) else 0

    return depth(ast.parse(expression, mode="eval").body)


def tree_svg(expression):
    tree = ast.parse(expression, mode="eval").body

    def build(node):
        if isinstance(node, ast.Call):
            return {"label": node.func.id, "children": [build(x) for x in node.args]}
        return {"label": ast.unparse(node), "children": []}

    root = build(tree)
    leaves = 0
    depth = 0

    def place(node, level):
        nonlocal leaves, depth
        depth = max(depth, level)
        for child in node["children"]:
            place(child, level + 1)
        if node["children"]:
            node["x"] = sum(c["x"] for c in node["children"]) / len(node["children"])
        else:
            node["x"] = 90 + leaves * 170
            leaves += 1
        node["y"] = 30 + level * 70

    place(root, 0)
    items = []

    def draw(node):
        x, y = node["x"], node["y"]
        for child in node["children"]:
            items.append(
                f'<line x1="{x}" y1="{y + 16}" x2="{child["x"]}" y2="{child["y"] - 16}"'
                ' stroke="#94a3b8"/>'
            )
            draw(child)
        label = html.escape(node["label"])
        items.append(
            f'<rect x="{x - 77}" y="{y - 16}" width="154" height="32" rx="5"'
            ' fill="#eff6ff" stroke="#93c5fd"/>'
            f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="12">{label}</text>'
        )

    draw(root)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max(190, leaves * 170)}"'
        f' height="{(depth + 1) * 70}" role="img" aria-label="GP expression tree">'
        + "".join(items)
        + "</svg>"
    )


def individual_page(programs, traces):
    data = json.dumps(
        {
            "programs": programs,
            "traces": traces,
            "features": feature_names(2),
            "help": FEATURE_HELP,
        },
        ensure_ascii=False,
        allow_nan=False,
    ).replace("<", "\\u003c")
    trees = "".join(
        f'<section id="tree-{html.escape(p["method"])}" class="tree">'
        f"{tree_svg(p['expression'])}</section>"
        for p in programs
    )
    return (
        """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>进化个体：真实输入与动作评分</title>
<style>
body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#182235}
h1{font-size:26px}select{font:inherit;margin:6px;padding:6px;max-width:95%}
table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}
.chosen{background:#dbeafe}.illegal{color:#999}.tree{overflow:auto;padding:15px;background:#fafafa}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:15px}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:800px){.columns{display:block}}
</style><h1>进化个体：它实际读到了什么，又选择了什么</h1>
<p>每棵树对 32 个候选动作分别评分。输入包含当前搜索状态，也包含候选动作的信息；
最终从合法动作中选择评分最高者。这里的输入来自独立开发实例回放，不是人为编造的数据。</p>
<label>个体 <select id="program"></select></label>
<label>实际决策 <select id="trace"></select></label>
<pre id="expression"></pre>"""
        + trees
        + """
<p id="summary"></p><div class="columns"><section><h2>选中动作的 12 个输入</h2>
<table id="inputs"></table></section><section><h2>32 个动作及输出评分</h2>
<table id="scores"></table></section></div>
<p>灰色行为非法动作；蓝色行为实际选中动作。GP 内部数值是 FP32，算子逐步裁剪到 [-8,8]。
完整 IR 和常量保存在对应程序 JSON；树图与这里的数值仅用于展示。</p>
<script id="data" type="application/json">"""
        + data
        + """</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const program=document.getElementById('program'),trace=document.getElementById('trace');
const number=v=>v===null?'不可选':Number(v).toPrecision(7);
data.programs.forEach(p=>{
 const o=new Option(p.method+' · '+p.nodes+' 节点',p.method);program.add(o)});
function show(){
 const p=data.programs.find(p=>p.method===program.value);
 document.getElementById('expression').textContent=p.expression;
 document.querySelectorAll('.tree').forEach(t=>t.hidden=t.id!=='tree-'+p.method);
 trace.innerHTML='';
 data.traces.forEach((t,i)=>{if(t.method===p.method)trace.add(new Option(
  'TSP'+t.dimension+' / '+t.instance_id+' / seed '+t.seed+' / 第 '+t.iteration+' 次决策',i))});
 decision();
}
function decision(){
 const t=data.traces[Number(trace.value)];
 if(!t){document.getElementById('summary').textContent='实际回放尚未完成。';return}
 const a=t.action;
 document.getElementById('summary').textContent='实际选择：动作 '+a+'，重启 '+(a>=16?'是':'否')+
  '，MNE '+(2**(1+a%4))+'，区域 '+(Math.floor(a/4)%4)+'，评分 '+number(t.scores[a]);
 const inputs=document.getElementById('inputs');
 inputs.innerHTML='<tr><th>输入</th><th>值</th><th>含义</th></tr>';
 data.features.forEach((f,i)=>{const r=inputs.insertRow();[f,number(t.features[i][a]),data.help[i]]
  .forEach(v=>r.insertCell().textContent=v)});
 const scores=document.getElementById('scores');
 scores.innerHTML='<tr><th>动作</th><th>重启</th><th>MNE</th><th>区域</th><th>评分</th></tr>';
 for(let i=0;i<32;i++){const r=scores.insertRow();
  r.className=i===a?'chosen':((t.legal_mask>>>i)&1)?'':'illegal';
  [i,i>=16?'是':'否',2**(1+i%4),Math.floor(i/4)%4,number(t.scores[i])]
  .forEach(v=>r.insertCell().textContent=v)}
}
program.onchange=show;trace.onchange=decision;show();
</script></html>"""
    )


def plot_curves(path, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not any(r.get("generations") for r in runs.values()):
        return
    figure, axes = plt.subplots(len(runs), 2, figsize=(13, 3.8 * len(runs)), squeeze=False)
    for row, (seed, run) in enumerate(runs.items()):
        training = run.get("generations", [])
        left, right = axes[row]
        for key, label, style in (
            ("fitness", "Champion train", "-"),
            ("population_mean_gap", "Population mean", "--"),
            ("population_median_gap", "Population median", ":"),
        ):
            available = [r for r in training if r.get(key) is not None]
            if available:
                left.plot(
                    [r["generation"] for r in available],
                    [r[key] for r in available],
                    style,
                    label=label,
                )
        for method, label in ((NATIVE_BASELINE, "FACO 2022"), (GPU_BASELINE, "GPU FACO")):
            paired = []
            for record in training:
                metrics = record.get("paired_baselines") or {}
                values = metrics.get("comparisons", {})
                value = values.get(method) if isinstance(values, dict) else None
                if value:
                    gap = value.get("faco_gap_percent")
                    if gap is not None:
                        paired.append((record["generation"], gap))
            if paired:
                left.plot(*zip(*paired, strict=True), label=label, alpha=0.7)
        monitor = run.get("monitoring", [])
        if monitor:
            right.plot(
                [r["generation"] for r in monitor],
                [r["fitness"] for r in monitor],
                "o-",
                label="Fixed development monitor",
            )
        validation = run.get("validation", [])
        if validation:
            right.plot(
                [r["generation"] for r in validation],
                [r["fitness"] for r in validation],
                label="Generation champions: validation",
            )
        for axis in (left, right):
            axis.set(
                xlabel="Evaluated generation", ylabel="Reference gap (%)", title=f"Seed {seed}"
            )
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path / "training_validation.svg")
    figure.savefig(path / "training_validation.png", dpi=160)
    plt.close(figure)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(directory, *, final=False):
    config, status = read(directory / "campaign.json"), read(directory / "status.json", {})
    formal = config["scope"] != "engineering_campaign_only"
    light = config.get("two_stage_validation", False)
    figures = PROJECT / "results/v2" / (directory.name if light else "round-3seed") if formal else directory / "report"
    figures.mkdir(parents=True, exist_ok=True)
    report = PROJECT / "docs/reports/三次进化实验.md" if formal else figures / "report.md"

    def relative(path):
        return os.path.relpath(path, report.parent)

    now = datetime.now(ZoneInfo("Pacific/Auckland")).isoformat(timespec="seconds")
    stages = {
        "waiting_horizon": "等待现有开发曲线完成并冻结训练 H",
        "calibration_and_fixed_baselines": "多卡开发标定与固定预算 FACO 对照并行执行",
        "waiting_shared_files": "等待已完成任务的文件在共享目录可见，计算不会重复",
        "baselines": "准备完整配对 FACO 基线",
        "training_and_validation": "执行训练与候选验证",
        "test_and_explanation": "最终程序已冻结，正在测试与记录真实决策",
        "complete": "本轮训练、验证、测试与解释全部完成",
        "failed": "有失败任务，保留已有结果，依赖阶段尚未完成",
    }
    frozen = read(directory / "frozen.json", {})
    declaration = read(directory / "panels.json")
    test_seed_count = len(declaration["test"][0]["seeds"])
    text = [
        "# 三次进化实验：训练、验证、测试与个体解释",
        "",
        f"更新于 {now}。当前阶段：**{stages.get(status.get('stage'), '尚未启动')}**。",
        "",
        "本轮比较 Full GP 与两种 FACO。它是三次进化的实际实验，"
        "不代表 NoFeedback、Static/Rule 或 E2–E4 已完成。",
        (
            "完整参数依据、数据划分、评价预算和统计规则见"
            "[本轮实验协议](../experiments/round_three_seed_v2.md)。"
            if formal
            else ""
        ),
        "" if formal else "**此目录仅为开发数据上的工程检查，不是正式实验成绩。**",
        "",
        "## 实验怎样设置",
        "",
        "| 项目 | 本轮设置 |",
        "|---|---|",
        f"| 进化 seeds | {', '.join(map(str, config['seeds']))} |",
        f"| 种群与代数 | {config['population']} 个体 × {config['generations']} 个完整评价代 |",
        "| GP 算子 | 精英4、锦标赛4、交叉0.8、变异0.2、初始深度1–3、"
        "变异子树深度0–2、最大深度5、最多63节点 |",
        (f"| 每代面板 | TSP500：{config['instances_per_panel']} 实例 × 1 个 ACO seed；三个GP seed独立进化 |" if light else
         f"| 每代面板 | TSP500、TSP1000 各 {config['instances_per_panel']} 实例 × 2 求解 seeds |"),
        "| 蚂蚁与基础参数 | 两规模均128蚂蚁；beta=1、retention=0.5、"
        "候选16/64、LS候选20、p_best=0.1 |",
        f"| ACO 迭代 | 训练 H={frozen.get('training_iterations', '待开发曲线冻结')}；"
        f"验证 {config['validation_iterations']}；测试 {config['test_iterations']} |",
        (f"| 两级验证 | 初筛：{len(declaration['validation'][0]['ids'])}实例 × 1 seed × {config['validation_screen_iterations']}迭代；"
         f"前{config['validation_finalists']}候选：{len(declaration['validation_final'][0]['ids'])}实例 × 3 seeds × {config['validation_iterations']}迭代 |" if light else ""),
        ("| 数据用途 | TSP500训练与选模；冻结后TSP500正式测试、TSP1000迁移测试 |" if light else ""),
        "| 数值与资源 | exact；只用空闲 A5000；按评价次数结束，无算法时间上限 |",
        "| 两种对照 | 作者 FACO 2022 的连续距离适配；同 GPU 底座 MNE8、全路线均匀起点、不重启 |",
        "",
        ("训练 H 按已登记的开发曲线规则选择：保留到5000迭代改进的至少95%，"
         + ("且TSP500控制器排名相关至少0.90。本轮选中1000迭代；TSP1000不参与训练预算或选模。" if light else
            "并且两个规模的控制器排名相关都至少0.90；没有短预算通过则用5000。")
         if formal else "工程流程使用小预算检查依赖、分片与归集，不用于预算选择或算法质量结论。"),
        "GP 的种群、选择和树限制是预先登记的设计选择，不宣称已经调参最优。"
        "每代的全部个体都重新评价，精英和重复表达式也不复用训练 fitness。",
        "",
        "## 当前执行情况",
        "",
        "| 任务类别 | 已完成 | 正在运行 | 待运行 | 失败 |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, counts in status.get("by_kind", {}).items():
        text.append(
            f"| {JOB_LABELS.get(kind, kind)} | "
            + " | ".join(
                str(counts.get(k, 0)) for k in ("completed", "running", "pending", "failed")
            )
            + " |"
        )
    text += ["", "活动设备与任务：", ""]
    for job, value in status.get("active", {}).items():
        resource = value['gpu_uuid'] or ("CPU进化协调器" if value.get('kind') == 'training' and light else "CPU / 8线程")
        text.append(f"- {job}：{value['host']}，{resource}")
    if not status.get("active"):
        text.append("- 当前没有已派发的活动任务。")
    if formal and status.get("stage") == "waiting_horizon":
        calibration = read(PROJECT / config["source_protocol"] / "curves/exact/progress.json")
        if calibration:
            text += [
                "",
                f"现有开发预算标定已完成 {calibration['completed_jobs']}/240 个 GPU 调用；"
                "协调器等待其最终结果，随后自动冻结 H 和准备 baseline。",
            ]
    runs, programs, traces = {}, [], []
    for seed in config["seeds"]:
        run_path = training_directory(directory, seed)
        curve = (
            load_checkpoint(run_path / "learning_curve.json")
            if (run_path / "learning_curve.json").exists()
            else {"generations": [], "monitoring": []}
        )
        summary = (
            load_checkpoint(run_path / "summary.json")
            if (run_path / "summary.json").exists()
            else {}
        )
        curve["validation"] = summary.get("champion_validation_curve", [])
        runs[seed] = curve
        if summary.get("selected"):
            export = load_checkpoint(run_path / "selected_program.json")
            programs.append(
                {
                    "method": f"gp-{seed}",
                    "seed": seed,
                    "program": export["program"],
                    "expression": summary["selected"]["expression"],
                    "nodes": summary["selected"]["nodes"],
                    "depth": expression_depth(summary["selected"]["expression"]),
                    "features_used": sorted(
                        {
                            node.id
                            for node in ast.walk(ast.parse(summary["selected"]["expression"]))
                            if isinstance(node, ast.Name) and node.id in feature_names(2)
                        }
                    ),
                    "validation_gap_percent": summary["selected"]["fitness"],
                }
            )
    # 曲线只在新增代际点/验证完成时重绘；状态刷新不反复生成相同图。
    signature = [
        [seed, len(r["generations"]), len(r["monitoring"]), len(r["validation"])]
        for seed, r in runs.items()
    ]
    if final or read(figures / "curve_counts.json") != signature:
        plot_curves(figures, runs)
        atomic_json(figures / "curves.json", {str(k): v for k, v in runs.items()})
        atomic_json(figures / "curve_counts.json", signature)
        write_csv(
            figures / "learning_curve.csv",
            [
                {
                    "seed": seed,
                    "generation": r["generation"],
                    **{
                        k: r.get(k)
                        for k in (
                            "fitness",
                            "population_mean_gap",
                            "population_median_gap",
                            "wall_seconds",
                        )
                    },
                }
                for seed, curve in runs.items()
                for r in curve["generations"]
            ],
        )
    text += [
        "",
        "## 训练与验证曲线",
        "",
        "左图是每代训练面板上的冠军、种群平均及中位 gap；面板逐代变化。"
        "右图是固定开发监控与验证面板上的代际冠军曲线，"
        f"最终验证完成后才填齐{config['generations']}个点。"
        + (f"本轮监控{config.get('monitor_iterations')}迭代，快速验证{config.get('validation_screen_iterations')}迭代，数据面板不同。" if light else
         "不同预算的两条曲线不能直接当成泛化差距。"),
        ("轻量验证曲线使用固定32实例、1 seed和1000迭代；前4候选的最终选模成绩单列。"
         "正式配置的两级验证最多为训练FE的9.28%，包含代际监控为9.44%。" if light and formal else ""),
        "",
    ]
    if (figures / "training_validation.svg").exists():
        text.append(f"![训练与验证曲线]({relative(figures / 'training_validation.svg')})")
    else:
        text.append("尚无完整代际记录，不绘制虚构曲线。")
    if any(r["generations"] for r in runs.values()):
        text += [
            "",
            "| 进化 seed | 已完成代数 | 每代平均用时（秒） | 训练/监控已用 FE |",
            "|---|---:|---:|---:|",
        ]
        for seed, curve in runs.items():
            generations = curve["generations"]
            costs = generations[-1].get("costs_cumulative", {}) if generations else {}
            finished = read(directory / "jobs" / f"training-{seed}" / "result.json", {})
            elapsed = statistics.mean(r["wall_seconds"] for r in generations) if generations else 0
            fe = finished.get("total_fe", costs.get("search_tour_evaluations", 0))
            text.append(f"| {seed} | {len(generations)} | {elapsed:.3f} | {fe} |")
    if light:
        timing_rows = []
        for seed in config["seeds"]:
            run = training_directory(directory, seed)
            summary = load_checkpoint(run / "summary.json") if (run / "summary.json").exists() else None
            if summary and summary.get("validation_costs"):
                training_seconds = sum(r.get("generation_costs", {}).get("native_actual_seconds", 0)
                                       for r in runs[seed]["generations"])
                cost = summary["validation_costs"]
                timing_rows.append((seed, training_seconds, cost["solve_seconds"], cost["training_fe_fraction"]))
        if timing_rows:
            text += ["", "训练与验证成本分别记账；GPU秒数是各分片求解时间之和，不是多卡经过时间。", "",
                     "| seed | 训练GPU秒 | 两级验证GPU秒 | 验证/训练时间 | 验证/训练FE |",
                     "|---|---:|---:|---:|---:|"]
            for seed, train, validation, fraction in timing_rows:
                text.append(f"| {seed} | {train:.2f} | {validation:.2f} | {validation/train:.2%} | {fraction:.2%} |")
        performance = read(PROJECT / "results/v2/tsp500_performance.json")
        if formal and performance:
            text += ["", "开发面板实测：128个体×16实例×1 seed×1000迭代，单张A5000从252.66秒降到137.52秒，"
                     "约1.84倍加速；完整路线、状态与工作量相同。正式每代时间以上表为准，包含任务等待与训练管理。"
                     "[测量与分片选择](../../results/v2/tsp500_performance.json)。"]
    text += [
        "",
        "## 测试比 FACO 提升多少",
        "",
        f"先在每个实例内平均{test_seed_count}个求解seed，再汇总实例和三次进化。"
        "改进百分点 = FACO gap − GP gap，正数表示GP更好；"
        "相对路线改善 = 100×(FACO长度−GP长度)/FACO长度。"
        "reference gap 使用用户提供的参考解，不另行宣称参考解已被证明最优。",
        "",
    ]
    stats = read(directory / "statistics.json")
    if stats:
        atomic_json(figures / "test_statistics.json", stats)
        write_csv(figures / "test_comparison.csv", stats["rows"])
        text += [
            "| 对照 | 规模 | FACO gap | GP gap | 改进百分点 | 95%区间 | 相对路线改善 | 胜/平/负 |",
            "|---|---|---:|---:|---:|---|---:|---|",
        ]
        for row in stats["rows"]:
            low, high = row["ci95_pp"]
            text.append(
                f"| {LABELS[row['baseline']]} | {row['dimension']} | "
                f"{row['baseline_gap_percent']:.5f}% | {row['gp_gap_percent']:.5f}% | "
                f"{row['improvement_pp']:+.5f} | [{low:+.5f}, {high:+.5f}] | "
                f"{row['relative_cost_improvement_percent']:+.5f}% | "
                f"{row['wins']}/{row['ties']}/{row['losses']} |"
            )
        text += ["", "TSP500两项主比较的 Holm 校正 p 值：" if light else "整体两项比较的 Holm 校正 p 值：", ""]
        for row in stats["rows"]:
            if "holm_p_value" in row:
                text.append(f"- {LABELS[row['baseline']]}：{row['holm_p_value']:.6g}。")
        text += [
            "",
            "各次进化单独列出，不能只展示测试最好的一个：",
            "",
            "| 个体 | 规模 | 平均 gap |",
            "|---|---|---:|",
        ]
        for row in stats["per_evolution_seed"]:
            text.append(
                f"| {row['method']} | {row['dimension']} | {row['mean_gap_percent']:.5f}% |"
            )
    else:
        text.append("正式测试尚未完整完成，暂不报告提升幅度或显著性结论。")
    text += ["", "## 进化出的 individual 与真实输入输出", ""]
    for program in programs:
        text += [
            f"### {program['method']}",
            "",
            f"验证 gap={program['validation_gap_percent']:.5f}%，{program['nodes']} 个节点，"
            f"深度 {program['depth']}。实际使用特征："
            + (", ".join(program["features_used"]) or "无；这是常量程序。"),
            "",
            "    " + program["expression"],
            "",
        ]
        atomic_json(figures / f"{program['method']}.json", program)
    if not programs:
        text.append("最终个体尚未完成验证选择。")
    if final:
        for path in sorted((directory / "jobs").glob("explain-*/result.json")):
            traces.extend(read(path).get("decisions", []))
        atomic_json(figures / "decision_samples.json", traces)
        (figures / "individuals.html").write_text(individual_page(programs, traces))
        text += [
            "",
            f"[打开树图、12个真实输入和32个动作评分]({relative(figures / 'individuals.html')})。",
        ]
        timing = []
        for path in sorted((directory / "jobs").glob("test_*/result.json")):
            result = read(path)
            for row in result["rows"]:
                timing.append(
                    {
                        "method": row["method"],
                        "dimension": row["dimension"],
                        "instance_id": row["instance_id"],
                        "seed": row["seed"],
                        "amortized_seconds": row["seconds"],
                    }
                )
        write_csv(figures / "test_runtime.csv", timing)
        grouped = defaultdict(list)
        for row in timing:
            grouped[row["method"], row["dimension"]].append(row["amortized_seconds"])
        text += [
            "",
            "## 效率与结果位置",
            "",
            f"[测试时间明细]({relative(figures / 'test_runtime.csv')})按方法、实例和seed列出。"
            "GPU 时间为完整面板摊销值；准备和外部评分另记在分片结果中。"
            "作者 FACO 使用 CPU，其时间单列，不能把跨硬件时间差解释为 GP 的净收益。",
            "",
            "| 方法 | 规模 | 实例/seed 平均摊销秒 |",
            "|---|---|---:|",
        ]
        for (method, n), values in sorted(grouped.items()):
            text.append(f"| {LABELS.get(method, method)} | {n} | {statistics.mean(values):.6f} |")
    text += [
        "",
        "原始路线只在外部评分一次。恢复读取已保存结果；报告不重算路线长度，"
        "不计算代码、文件、程序或检查点的完整性哈希。",
        "",
        f"[机器可读队列状态]({relative(directory / 'status.json')})；"
        f"[固定实验配置]({relative(directory / 'campaign.json')})。",
    ]
    if formal:
        text += [
            "",
            ("[启动前检查记录](../../results/v2/tsp500_verification.json)包含回归测试、" if light else
             "[启动前检查记录](../../results/v2/campaign_verification.json)包含回归测试、")
            + "跨 A5000 一致性和小规模完整流程证据。小规模成绩不计入正式结果。",
        ]
    report.write_text("\n".join(text) + "\n")
    if formal:
        (PROJECT / "docs/reports/progress.md").write_text(
            "# 当前研究进度\n\n"
            f"更新于 {now}。\n\n"
            f"当前三次进化实验：**{stages.get(status.get('stage'), '尚未启动')}**。\n\n"
            "[直接阅读三次进化实验报告](三次进化实验.md)。"
            "该报告集中展示配置、运行状态、曲线、FACO对比和个体解释。\n\n"
            "[已有 FACO 复现与 GPU 优化证据](实验说明与结果.md)继续保留。"
            "完整 E1 的其他对照与 E2–E4 尚未完成。\n"
        )
