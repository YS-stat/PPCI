from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def set_paper_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": False,
    })


def plot_metric_summary(summary_csv: str | Path, output_pdf: str | Path, x_col: str = "x0_index", title: str | None = None):
    """Plot only the three paper-facing metrics: bias, coverage, and width."""
    set_paper_style()
    df = pd.read_csv(summary_csv)
    methods = [m for m in ["PPCI", "LO", "PPI"] if f"{m}_coverage" in df.columns]
    colors = {"PPCI": "#005a9e", "LO": "#e69f00", "PPI": "#c0392b"}
    labels = {"PPCI": "PPCI", "LO": "LO", "PPI": "PPI"}
    metrics = [("bias", "Bias"), ("coverage", "Coverage"), ("width", "Width")]
    if x_col not in df.columns:
        x_col = df.columns[0]
    x = df[x_col].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=False)
    for ax, (metric, label) in zip(axes, metrics):
        for method in ["LO", "PPI", "PPCI"]:
            if method not in methods:
                continue
            col = f"{method}_{metric}"
            if col not in df.columns:
                continue
            ax.plot(x, df[col].to_numpy(), marker="o", markersize=4, linewidth=2.2,
                    color=colors[method], label=labels[method], zorder={"LO": 3, "PPI": 4, "PPCI": 5}[method])
        if metric == "coverage":
            ax.axhline(0.95, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(label, loc="left", pad=8)
        ax.set_xlabel(x_col.replace("_", " "))
        ax.grid(axis="y", color="0.85", linewidth=0.8)
    axes[0].set_ylabel("Value")
    axes[-1].legend(frameon=True, facecolor="white", edgecolor="none", loc="best")
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
