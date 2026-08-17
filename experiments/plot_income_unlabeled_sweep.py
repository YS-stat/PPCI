#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Income PPCI unlabeled-N sweep.")
    parser.add_argument("summary", help="Merged summary_by_N.csv")
    parser.add_argument(
        "--output",
        default="figures/income_mean_sex1_age70_ppci_sweepN_coverage_width.pdf",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.summary).sort_values("n_unlab")
    required = {"n_unlab", "coverage", "width"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plt.rcParams.update({
        "font.family": "Helvetica",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.titlesize": 18,
    })
    blue = "#005A9E"
    red = "#C0392B"
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    axes[0].plot(data["n_unlab"], data["coverage"], color=blue, marker="o", linewidth=2.2, markersize=7)
    axes[0].axhline(0.95, color=red, linestyle="--", linewidth=2.0, label="Nominal 95%")
    axes[0].set(xlabel=r"Unlabeled sample size $N$", ylabel="Coverage", ylim=(0.0, 1.0))
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(data["n_unlab"], data["width"], color=blue, marker="o", linewidth=2.2, markersize=7)
    axes[1].set(xlabel=r"Unlabeled sample size $N$", ylabel="Average CI width")

    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", rotation=30)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
