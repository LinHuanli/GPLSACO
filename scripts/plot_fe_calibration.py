#!/usr/bin/env python3
"""从已审计的完整开发FE曲线导出科研图；保留全部24配置，不按最好结果筛图。"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["status"] != "passed" or report["purpose"] != "calibration":
        raise ValueError("只绘制已完整审计的校准")
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "gpfaco-calibration-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    colors = {"static": "#5776a5", "rule": "#c78a45"}
    identities = sorted({v["policy_sha256"] for v in report["curves"]})
    limits = report["limits"]
    for n, ax in zip((500, 1000), axes, strict=True):
        for identity in identities:
            rows = sorted(
                (v for v in report["curves"] if v["policy_sha256"] == identity),
                key=lambda v: v["limit"],
            )
            ax.plot(
                limits,
                [v[f"gap_{n}"] for v in rows],
                color=colors[rows[0]["kind"]],
                alpha=0.45,
                linewidth=0.8,
            )
        means = [
            statistics.mean(v[f"gap_{n}"] for v in report["curves"] if v["limit"] == limit)
            for limit in limits
        ]
        ax.plot(limits, means, color="#20252b", linewidth=2.0, marker="o", markersize=3.2)
        ax.axvline(4096, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xscale("symlog", base=4, linthresh=256, linscale=0.8)
        ax.set_xticks(limits, ["0", "256", "1,024", "4,096", "16,384"])
        ax.set_xlim(-25, 21000)
        ax.set_ylim(0, 6.6)
        ax.set_xlabel("Search-tour evaluations per colony")
        ax.set_title(f"TSP{n}: 32 development instances × 2 solve seeds")
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Mean reference gap (%)")
    fig.legend(
        handles=[
            Line2D([0], [0], color=colors["static"], label="20 Static configurations"),
            Line2D([0], [0], color=colors["rule"], label="4 Rule configurations"),
            Line2D([0], [0], color="#20252b", linewidth=2, label="Mean of all 24"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(args.output / "fe-calibration.svg", metadata={"Date": None})
    fig.savefig(args.output / "fe-calibration.pdf", metadata={"CreationDate": None})
    fig.savefig(args.output / "fe-calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
