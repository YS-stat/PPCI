#!/usr/bin/env python3
"""Analytic mechanism comparison for NW localization and signed balance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from ppci_condmean.utils import source_sha256


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def compute_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    h_grid = np.geomspace(args.h_min, args.h_max, args.h_points)
    records: list[dict] = []
    slope_records: list[dict] = []
    signed_refs: dict[str, dict] = {}

    for b in args.b_values:
        bias = args.curvature * h_grid**2 / 3.0
        signal = b**2 * h_grid / 3.0 + 4.0 * args.curvature**2 * h_grid**3 / 45.0
        var_lo = args.sigma**2 / (args.n * h_grid) + signal / args.n
        var_ppci = args.sigma**2 / (args.n * h_grid) + signal / args.N
        gain = 1.0 - var_ppci / var_lo
        se_ppci = np.sqrt(var_ppci)
        ratio = np.abs(bias) / se_ppci
        coverage = norm.cdf(args.z_alpha - ratio) - norm.cdf(-args.z_alpha - ratio)

        assert np.all(var_lo > 0.0) and np.all(var_ppci > 0.0)
        assert np.all((gain >= -1e-12) & (gain <= 1.0 + 1e-12))

        var_lo_star = (
            9.0 * args.sigma**2 / 4.0
            + 9.0 * b**2 / 28.0
            + 23.0 * args.curvature**2 / 140.0
        ) / args.n
        var_ppci_star = 9.0 * args.sigma**2 / (4.0 * args.n) + (
            9.0 * b**2 / 28.0 + 23.0 * args.curvature**2 / 140.0
        ) / args.N
        gain_star = 1.0 - var_ppci_star / var_lo_star
        assert var_lo_star > 0.0 and var_ppci_star > 0.0 and 0.0 <= gain_star <= 1.0
        signed_refs[f"b={b:g}"] = {
            "var_lo": var_lo_star,
            "var_ppci": var_ppci_star,
            "gain": gain_star,
        }

        for index, h in enumerate(h_grid):
            records.append(
                {
                    "b": b,
                    "h": h,
                    "bias_nw": bias[index],
                    "var_lo_nw": var_lo[index],
                    "var_ppci_nw": var_ppci[index],
                    "se_ppci_nw": se_ppci[index],
                    "bias_to_se": ratio[index],
                    "normal_approx_coverage": coverage[index],
                    "gain_nw": gain[index],
                    "var_lo_signed_reference": var_lo_star,
                    "var_ppci_signed_reference": var_ppci_star,
                    "gain_signed_reference": gain_star,
                }
            )

        n_small = max(5, int(np.ceil(args.slope_fraction * len(h_grid))))
        selected = slice(0, n_small)
        slope, intercept = np.polyfit(np.log(h_grid[selected]), np.log(gain[selected]), 1)
        expected = 2.0 if abs(b) > 1e-12 else 4.0
        slope_records.append(
            {
                "b": b,
                "grid_fraction": args.slope_fraction,
                "n_points": n_small,
                "h_max_used": h_grid[n_small - 1],
                "slope": slope,
                "expected_slope": expected,
                "absolute_error": abs(slope - expected),
            }
        )
        assert abs(slope - expected) <= args.slope_tolerance, (b, slope, expected)

    # Exact moments under X ~ Unif[-1, 1].
    moments = {
        "E_wstar": 9.0 / 4.0 - (15.0 / 4.0) / 3.0,
        "E_wstar_X": 0.0,
        "E_wstar_X2": (9.0 / 4.0) / 3.0 - (15.0 / 4.0) / 5.0,
    }
    assert max(abs(value) for key, value in moments.items() if key != "E_wstar") < 1e-14
    assert abs(moments["E_wstar"] - 1.0) < 1e-14
    checks = {"wstar_moments": moments, "signed_references": signed_refs}
    return pd.DataFrame(records), pd.DataFrame(slope_records), checks


def make_figures(metrics: pd.DataFrame, slopes: pd.DataFrame, output: Path) -> None:
    colors = {0.0: "#2f7d4a", 2.0: "#1f4e79"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for b, block in metrics.groupby("b"):
        color = colors.get(float(b), "#444444")
        label = f"b = {b:g}"
        h_values = block["h"].to_numpy()
        axes[0, 0].plot(h_values, block["bias_nw"].abs().to_numpy(), color=color, label=f"Bias, {label}")
        axes[0, 0].plot(h_values, block["se_ppci_nw"].to_numpy(), color=color, linestyle="--", label=f"SE, {label}")
        axes[0, 1].plot(h_values, block["bias_to_se"].to_numpy(), color=color, label=label)
        axes[1, 0].plot(h_values, block["gain_nw"].to_numpy(), color=color, label=f"NW, {label}")
        axes[1, 0].axhline(
            block["gain_signed_reference"].iloc[0],
            color=color,
            linestyle=":",
            label=f"Signed reference, {label}",
        )

        n_small = int(slopes.loc[slopes["b"] == b, "n_points"].iloc[0])
        small = block.iloc[:n_small]
        small_h = small["h"].to_numpy()
        small_gain = small["gain_nw"].to_numpy()
        axes[1, 1].loglog(small_h, small_gain, "o", markersize=3, color=color, label=label)
        slope = float(slopes.loc[slopes["b"] == b, "slope"].iloc[0])
        intercept = np.polyfit(np.log(small["h"]), np.log(small["gain_nw"]), 1)[1]
        axes[1, 1].loglog(small_h, np.exp(intercept) * small_h**slope, color=color, linestyle="--")

    axes[0, 0].set(title="NW localization bias and PPCI SE", xlabel="Bandwidth h", ylabel="Magnitude")
    axes[0, 1].set(title="NW bias-to-SE ratio", xlabel="Bandwidth h", ylabel="|Bias| / SE")
    axes[0, 1].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[1, 0].set(title="Relative variance reduction", xlabel="Bandwidth h", ylabel="1 - Var(PPCI) / Var(LO)")
    axes[1, 1].set(title="Small-h log-log slope diagnostic", xlabel="Bandwidth h", ylabel="Relative variance reduction")
    for ax in axes.flat:
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / "nw_mechanism_panels.pdf", bbox_inches="tight")
    fig.savefig(output / "nw_mechanism_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(args: argparse.Namespace, slopes: pd.DataFrame, output: Path) -> None:
    slope_text = ", ".join(f"b={row.b:g}: {row.slope:.3f}" for row in slopes.itertuples())
    text = f"""# Analytic NW Mechanism Experiment

The model is `X ~ Unif[-1,1]`, `Y = beta + b X + A X^2 + epsilon`, with a perfect predictor equal to the conditional mean. The target is `theta(0)=beta`. The configuration is `beta={args.beta:g}`, `A={args.curvature:g}`, `sigma={args.sigma:g}`, `n={args.n}`, and `N={args.N}`.

The NW calculations use the analytic uniform localization weight. As `h` decreases, NW localization bias falls, but the local variation explained by the fixed-covariate predictor also falls. Consequently, the relative prediction-powered variance reduction approaches zero.

The log-log slope uses the smallest {100 * args.slope_fraction:.0f}% of the predeclared bandwidth grid. Estimated slopes are {slope_text}, matching the predicted orders 2 and 4.

The horizontal signed-weight references use `w*(x)=9/4-15x^2/4`, the exact signed-balance representer only in `span{{1,x,x^2}}`. They are finite-dimensional references, not claims that a general RKHS localization estimator recovers `w*`.

`normal_approx_coverage` is a normal-approximation calculation and is not exact finite-sample coverage.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/nw_mechanism_analytic_v1")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--curvature", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--b-values", default="0,2")
    parser.add_argument("--h-min", type=float, default=0.05)
    parser.add_argument("--h-max", type=float, default=0.8)
    parser.add_argument("--h-points", type=int, default=120)
    parser.add_argument("--slope-fraction", type=float, default=0.40)
    parser.add_argument("--slope-tolerance", type=float, default=0.25)
    parser.add_argument("--z-alpha", type=float, default=1.959963984540054)
    args = parser.parse_args()
    args.b_values = parse_floats(args.b_values)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics, slopes, checks = compute_metrics(args)
    metrics.to_csv(output / "analytic_metrics.csv", index=False)
    slopes.to_csv(output / "slope_checks.csv", index=False)
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["sanity_checks"] = checks
    config["source_sha256"], config["source_file_count"] = source_sha256()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_figures(metrics, slopes, output)
    write_readme(args, slopes, output)
    print(slopes.to_string(index=False))
    print(f"Saved analytic experiment to {output}")


if __name__ == "__main__":
    main()
