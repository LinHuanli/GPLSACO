"""六个冻结控制器的配对统计；不以测试表现选择seed或修改方法。"""

import numpy as np
from scipy.stats import wilcoxon

from gp_faco.representation_campaign import RUN_IDS
from gp_faco.representation_pilot import BASELINES, REPRESENTATIONS, SEEDS


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        previous = min(1.0, max(previous, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def comparison_statistics(rows, config, panels):
    methods = [*RUN_IDS, *BASELINES]
    indexed = {}
    for row in rows:
        key = row["method"], row["dimension"], row["instance_id"], row["seed"]
        if key in indexed or row["iterations"] != config["test_iterations"]:
            raise ValueError("测试重复结果或预算不符")
        indexed[key] = row
    expected = {
        (m, p["dimension"], name, seed)
        for m in methods
        for p in panels
        for name in p["ids"]
        for seed in p["seeds"]
    }
    if indexed.keys() != expected:
        raise ValueError("全部冻结方法的测试实例与seed必须收齐")
    rng = np.random.default_rng(config["statistics_seed"])
    report = {
        "per_run": [],
        "comparisons": [],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "primary_dimension": 500,
        "primary_family_size": 9,
        "selection_uses_test": False,
    }
    for panel in panels:
        n, ids, aco_seeds = panel["dimension"], panel["ids"], panel["seeds"]
        cubes = {
            m: {
                field: np.asarray(
                    [[indexed[m, n, name, seed][field] for seed in aco_seeds] for name in ids]
                )
                for field in ("gap_percent", "cost")
            }
            for m in methods
        }
        for m in methods:
            report["per_run"].append(
                {
                    "method": m,
                    "dimension": n,
                    "mean_gap_percent": float(cubes[m]["gap_percent"].mean()),
                }
            )
        comparisons = [(rep, base) for rep in REPRESENTATIONS for base in BASELINES]
        comparisons.append((REPRESENTATIONS[0], REPRESENTATIONS[1]))
        for left, right in comparisons:
            a = np.stack([cubes[f"{left}-s{s}"]["gap_percent"] for s in SEEDS])
            ac = np.stack([cubes[f"{left}-s{s}"]["cost"] for s in SEEDS])
            if right in REPRESENTATIONS:
                b = np.stack([cubes[f"{right}-s{s}"]["gap_percent"] for s in SEEDS])
                bc = np.stack([cubes[f"{right}-s{s}"]["cost"] for s in SEEDS])
            else:
                b = np.broadcast_to(cubes[right]["gap_percent"], a.shape)
                bc = np.broadcast_to(cubes[right]["cost"], ac.shape)
            delta = b - a
            instance = delta.mean(axis=(0, 2))
            samples = []
            # 同步重采样GP seed、实例及实例内ACO seed，始终保持方法间配对。
            for _ in range(config["bootstrap_replicates"]):
                gs = rng.integers(0, len(SEEDS), len(SEEDS))
                ii = rng.integers(0, len(ids), len(ids))
                ss = rng.integers(0, len(aco_seeds), (len(ids), len(aco_seeds)))
                samples.append(
                    float(delta[gs[:, None, None], ii[None, :, None], ss[None, :, :]].mean())
                )
            value = {
                "dimension": n,
                "method": left,
                "comparator": right,
                "method_gap_percent": float(a.mean()),
                "comparator_gap_percent": float(b.mean()),
                "improvement_percentage_points": float(delta.mean()),
                "relative_route_improvement_percent": float((100 * (bc - ac) / bc).mean()),
                "wins_ties_losses": [
                    int((instance > 0).sum()),
                    int((instance == 0).sum()),
                    int((instance < 0).sum()),
                ],
                "improvement_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            }
            if n == 500:
                value["wilcoxon_p"] = (
                    float(wilcoxon(instance).pvalue) if np.any(instance != 0) else 1.0
                )
            report["comparisons"].append(value)
    primary = [r for r in report["comparisons"] if r["dimension"] == 500]
    for row, adjusted in zip(primary, holm([r["wilcoxon_p"] for r in primary]), strict=True):
        row["holm_p"] = adjusted
    return report
