#!/usr/bin/env python3
# ruff: noqa: E402
"""分析已冻结的六次50代结果；只读原始记录，不运行ACO、不改选模。"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))

import numpy as np
from gp_faco.campaign import read
from gp_faco.checkpoint import atomic_json
from gp_faco.representation_pilot import BASELINES, REPRESENTATIONS

ROOT = PROJECT / "artifacts/v3/representation-50gen"
OUTPUT = PROJECT / "results/v3/representation-50gen"
FIXED = "gpu_fixed_mne16_region0_no_restart"
NAMES = {
    "joint_single": "Single tree",
    "conditional_three": "Three trees",
    BASELINES[0]: "Native FACO 2022",
    BASELINES[1]: "GPU uniform MNE8",
    FIXED: "GPU region0 MNE16",
    BASELINES[3]: "GPU uniform MNE16",
}


def describe(values):
    a = np.asarray(values)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.1)),
        "p90": float(np.quantile(a, 0.9)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def action_counts(actions):
    a = np.asarray(actions, dtype=int)
    return {
        "count": len(a),
        "mne": np.bincount(a % 4, minlength=4).tolist(),
        "region": np.bincount((a // 4) % 4, minlength=4).tolist(),
        "restart": np.bincount(a // 16, minlength=2).tolist(),
    }


def analyze(summary, examples):
    stats = summary["statistics"]
    result = {
        "scope": "posthoc_analysis_of_frozen_methods",
        "new_aco_evaluations": 0,
        "formal_statistics_source": "statistics.json",
        "runs": [],
        "test": [],
        "comparisons": stats["comparisons"],
    }
    for run in summary["runs"]:
        rid, h = run["run_id"], run["history"]
        selected = run["selected"]
        screen = read(ROOT / "runs" / rid / "validation_screen.json")
        training_job = read(ROOT / "jobs" / f"training-{rid}" / "result.json")
        finalists = [c["controller_id"] for c in screen["finalists"]]
        sid = selected["controller"]["controller_id"]
        val = np.asarray([x["gap_percent"] for x in run["validation_curve"]])
        diag = h[-1]["diagnostics"]
        # 所有趋势使用固定验证面板；变化训练面板仅用于同代配对差值。
        row = {
            "run_id": rid,
            "selected_id": sid,
            "selected_nodes": selected["selected"]["nodes"],
            "selected_quick_rank": finalists.index(sid) + 1,
            "selected_is_g50_champion": sid == h[-1]["winner"]["controller_id"],
            "finalists": [
                {
                    "id": c,
                    "quick_gap": screen["scores"][c]["gap_percent"],
                    "final_gap": selected["scores"][c]["gap_percent"],
                }
                for c in finalists
            ],
            "validation_first10": float(val[:10].mean()),
            "validation_last10": float(val[-10:].mean()),
            "validation_best_first10": float(val[:10].min()),
            "validation_best_all50": float(val.min()),
            "validation_best_generation": int(val.argmin() + 1),
            "validation_10gen_means": [float(v.mean()) for v in val.reshape(5, 10)],
            "monitor_curve": [
                {"generation": v["generation"], "gap_percent": v["monitor"]["gap_percent"]}
                for v in h
                if "monitor" in v
            ],
            "train_vs_fixed16_10gen_means": [
                float(
                    np.mean(
                        [
                            run["baseline_curve"][i][FIXED] - h[i]["training_gap_percent"]
                            for i in range(start, start + 10)
                        ]
                    )
                )
                for start in range(0, 50, 10)
            ],
            "g50_unique_ir": diag["unique_ir"],
            "g50_unique_behavior": diag["unique_behavior"],
            "g50_max_behavior_group": diag["max_behavior_group"],
            "g50_max_height_per_individual": describe([max(v) for v in diag["heights"]]),
            "g50_nodes_per_individual": describe([sum(v) for v in diag["nodes"]]),
            "g50_depth_at_most1_individuals": sum(max(v) <= 1 for v in diag["heights"]),
            "g50_context_action_counts": diag["context_action_counts"],
            "fitness_unique_g1_g10_g50": [
                len(np.unique(h[i]["population_fitness"])) for i in (0, 9, 49)
            ],
            "g50_best_fitness_ties": sum(
                v == min(h[-1]["population_fitness"]) for v in h[-1]["population_fitness"]
            ),
            "training_gpu_minutes": sum(v["costs"]["gpu_seconds"] for v in h) / 60,
            "training_wall_hours": training_job["wall_seconds"] / 3600,
            "monitor_gpu_minutes": sum(
                v["monitor"]["costs"]["gpu_seconds"] for v in h if "monitor" in v
            )
            / 60,
            "validation_gpu_minutes": selected["costs"]["gpu_seconds"] / 60,
            "generation_evaluation_wait_minutes": describe(
                [v["evaluation_wait_wall_seconds"] / 60 for v in h]
            ),
            "explanation_actions": {
                str(n): action_counts([v["action"] for v in examples[f"{rid}-n{n}"]])
                for n in (500, 1000)
            },
        }
        result["runs"].append(row)

    tests = read(ROOT / "test_scores.json")
    methods = [r["run_id"] for r in summary["runs"]] + list(BASELINES)
    for n in (500, 1000):
        for method in methods:
            rows = [r for r in tests["rows"] if r["method"] == method and r["dimension"] == n]
            timings = [r for r in tests["timings"] if r["method"] == method and r["dimension"] == n]
            row = {
                "method": method,
                "dimension": n,
                "gap_percent": describe([r["gap_percent"] for r in rows]),
                "solve_seconds": sum(r["solve_seconds"] for r in timings),
            }
            assert len(rows) == 1280
            official = next(
                r for r in stats["per_run"] if r["method"] == method and r["dimension"] == n
            )
            assert abs(row["gap_percent"]["mean"] - official["mean_gap_percent"]) < 1e-12
            if method != BASELINES[0]:
                states, work = (
                    [],
                    {
                        k: 0
                        for k in (
                            "completed_construction_steps",
                            "completed_ls_evaluations",
                            "total_tour_evaluations",
                        )
                    },
                )
                # 每个测试分片只读取一次，保存统计量，不复制tour或逐任务日志。
                for timing in timings:
                    raw = read(ROOT / "jobs" / timing["job"] / "result.json")
                    assert len(raw["members"]) == 1
                    native = raw["members"][0]["native_result"]
                    states.extend(native.get("control_states", []))
                    for key in work:
                        work[key] += native[key]
                row["work"] = work
                row["construction_steps_per_tour"] = (
                    work["completed_construction_steps"] / work["total_tour_evaluations"]
                )
                row["ls_evaluations_per_tour"] = (
                    work["completed_ls_evaluations"] / work["total_tour_evaluations"]
                )
                if states:
                    assert len(states) == 1280
                    row["restarts"] = describe([s["restarts"] for s in states])
                    row["no_restart_solves"] = sum(s["restarts"] == 0 for s in states)
                    row["stagnant_batches"] = describe([s["stagnant_batches"] for s in states])
                else:
                    # 均匀FACO路径没有控制状态输出；按冻结配置标记不重启。
                    assert method in (BASELINES[1], BASELINES[3])
                    row["restarts_by_configuration"] = 0
            result["test"].append(row)
    panels = read(ROOT / "panels.json")
    panel = panels["validation"][0]
    baseline_rows = read(ROOT / "baseline_scores.json")["rows"]
    result["quick_validation_baselines"] = {}
    for method in BASELINES:
        values = [
            r["gap_percent"]
            for r in baseline_rows
            if r["method"] == method
            and r["iterations"] == 1000
            and r["instance_id"] in panel["ids"]
            and r["seed"] in panel["seeds"]
        ]
        assert len(values) == 32
        result["quick_validation_baselines"][method] = float(np.mean(values))
    return result


def diagnose_gpu(gpu_uuid, summary, examples):
    from gp_faco.gpu_session import GpuSession

    session = GpuSession(gpu_uuid, "exact", extension_directory="build/v3-representation-optimized")
    contexts = np.load(ROOT / "contexts.npz")
    output = {
        "scope": "one_step_counterfactual_not_aco_ablation",
        "new_aco_evaluations": 0,
        "device": session.device,
        "runs": [],
    }
    try:
        for run in summary["runs"]:
            rid, controller = run["run_id"], run["selected"]["controller"]
            for label in ("training_contexts", "explain_n500", "explain_n1000"):
                if label == "training_contexts":
                    features = np.ascontiguousarray(contexts["features"], dtype=np.float32)
                    masks = np.ascontiguousarray(contexts["masks"], dtype=np.uint32)
                    saved = None
                else:
                    decisions = examples[f"{rid}-n{label.removeprefix('explain_n')}"]
                    features = np.ascontiguousarray(
                        np.stack([x["features"] for x in decisions], axis=1), dtype=np.float32
                    )
                    masks = np.asarray([x["legal_mask"] for x in decisions], dtype=np.uint32)
                    saved = np.asarray([x["action"] for x in decisions])

                def score(f, controller=controller, masks=masks):
                    return np.asarray(
                        session.native.score_controller_cuda(controller, f, masks)["actions"]
                    )

                original = score(features)
                if saved is not None:
                    assert np.array_equal(original, saved), "回放必须与保存的真实CUDA动作一致"
                changes = {}
                for index, name in ((1, "stagnation"), (2, "return_rate"), (3, "ls_work")):
                    changes[name] = {}
                    for value in (0, 1):
                        altered = features.copy()
                        altered[index] = value
                        actions = score(altered)
                        changes[name][str(value)] = {
                            "any": int(np.count_nonzero(actions != original)),
                            "restart": int(np.count_nonzero(actions // 16 != original // 16)),
                            "region": int(np.count_nonzero(actions // 4 % 4 != original // 4 % 4)),
                            "mne": int(np.count_nonzero(actions % 4 != original % 4)),
                        }
                output["runs"].append(
                    {
                        "run_id": rid,
                        "contexts": label,
                        "actions": action_counts(original),
                        "feedback_interventions": changes,
                        "saved_actions_match": saved is not None,
                    }
                )
    finally:
        session.close()
    atomic_json(OUTPUT / "selected_feedback_diagnostic.json", output)


def plot(summary, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    colors = ["#2478ad", "#d47a25"]
    for ax, n in zip(axes, (500, 1000), strict=True):
        for i, rep in enumerate(REPRESENTATIONS):
            comparisons = [
                next(
                    r
                    for r in analysis["comparisons"]
                    if r["dimension"] == n and r["method"] == rep and r["comparator"] == base
                )
                for base in BASELINES
            ]
            y = np.arange(4) + (i - 0.5) * 0.18
            x = np.array([r["improvement_percentage_points"] for r in comparisons])
            ci = np.array([r["improvement_ci95"] for r in comparisons])
            ax.errorbar(
                x,
                y,
                xerr=np.stack((x - ci[:, 0], ci[:, 1] - x)),
                fmt="o",
                capsize=3,
                color=colors[i],
                label=NAMES[rep],
            )
        ax.axvline(0, color="0.5", lw=1)
        ax.set(
            yticks=range(4),
            yticklabels=[NAMES[b] for b in BASELINES],
            title=f"TSP{n}: baseline gap minus GP gap",
            xlabel="Gap percentage points; positive favors GP",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(loc="lower right")
    fig.suptitle("Frozen test methods: paired mean differences and 95% bootstrap intervals")
    fig.savefig(OUTPUT / "test_effects.svg", metadata={"Date": None})
    fig.savefig(PROJECT / ".tmp/representation_test_effects.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", sharey=True)
    for ax, rep in zip(axes, REPRESENTATIONS, strict=True):
        curves = []
        for run in summary["runs"]:
            if run["run_id"].startswith(rep):
                curve = np.asarray([v["gap_percent"] for v in run["validation_curve"]])
                curves.append(curve)
                ax.plot(
                    np.arange(1, 51), curve, lw=0.8, alpha=0.45, label=run["run_id"].split("-s")[1]
                )
        ax.plot(
            np.arange(5.5, 51, 10),
            np.stack(curves).reshape(3, 5, 10).mean(axis=(0, 2)),
            "ko-",
            lw=2,
            label="3-seed / 10-gen mean",
        )
        for base, color, ls in ((BASELINES[1], "#858585", "--"), (FIXED, "#6b963b", ":")):
            ax.axhline(
                analysis["quick_validation_baselines"][base], color=color, ls=ls, label=NAMES[base]
            )
        ax.set(title=NAMES[rep], xlabel="Generation", ylabel="Fixed validation mean gap (%)")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Same 32 validation instances, one ACO seed, H1000; evaluated after training")
    fig.savefig(OUTPUT / "validation_analysis.svg", metadata={"Date": None})
    fig.savefig(PROJECT / ".tmp/representation_validation_analysis.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-uuid", help="仅做同一CUDA评分器的单步反馈诊断；需先用gpu-free选择空闲A5000"
    )
    parser.add_argument(
        "--plots-only", action="store_true", help="复用精简analysis.json，不再读取原始测试分片"
    )
    args = parser.parse_args()
    summary = read(OUTPUT / "summary.json")
    examples = read(OUTPUT / "decision_examples.json")
    if args.gpu_uuid:
        diagnose_gpu(args.gpu_uuid, summary, examples)
        return
    analysis = read(OUTPUT / "analysis.json") if args.plots_only else analyze(summary, examples)
    atomic_json(OUTPUT / "analysis.json", analysis)
    plot(summary, analysis)
    print(
        json.dumps(
            {
                "runs": len(analysis["runs"]),
                "test_method_scale_groups": len(analysis["test"]),
                "new_aco_evaluations": 0,
                "output": str(OUTPUT / "analysis.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
