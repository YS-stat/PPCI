#!/usr/bin/env python3
"""Plot the three PPCI variance components for the Income experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/income/sigma_by_age.csv")
    parser.add_argument("--output", default="figures/income_mean_age_sigma2_two_sex.pdf")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"sex", "age", "sigma2_Y", "sigma2_Y_minus_f", "sigma2_f"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labels = {
        "sigma2_Y": r"$\widehat\sigma_Y^2$",
        "sigma2_Y_minus_f": r"$\widehat\sigma_{Y-f}^2$",
        "sigma2_f": r"$\widehat\sigma_f^2$",
    }
    colors = {"sigma2_Y": "#1f4e79", "sigma2_Y_minus_f": "#c43d3d", "sigma2_f": "#2f7d4a"}
    markers = {"sigma2_Y": "o", "sigma2_Y_minus_f": "s", "sigma2_f": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True, constrained_layout=True)
    for ax, sex in zip(axes, [1, 2]):
        subset = data[data["sex"] == sex].sort_values("age")
        if subset.empty:
            raise ValueError(f"No rows found for sex={sex}")
        for column in labels:
            ax.plot(
                subset["age"].to_numpy(),
                subset[column].to_numpy(),
                label=labels[column],
                color=colors[column],
                marker=markers[column],
                markersize=3.2,
                linewidth=1.5,
            )
        ax.set_title(f"Sex = {sex}")
        ax.set_xlabel("Age")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Estimated variance component")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
