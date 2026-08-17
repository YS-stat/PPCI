#!/usr/bin/env python3
"""Same-bandwidth mechanism follow-up for NW and RKHS localization PPCI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppci_condmean.estimator import lo_mean_from_weights, ppci_mean_from_weight_values
from ppci_condmean.gpu import configure_backend
from ppci_condmean.joint_tuning import (
    JointTuningConfig,
    collect_joint_candidate_cache,
    select_joint_from_cache,
    weight_from_joint_cache,
)
from ppci_condmean.kernels import get_kernel
from ppci_condmean.tuning import distance_median_anchor
from ppci_condmean.utils import source_sha256
from run_nw_localization_quadratic_mc import (
    effective_sample_size,
    m_function,
    nw_cv_bandwidth,
    nw_twofold_weights,
    parse_floats,
    w_star,
)


def fixed_h_subcache(cache: dict, h: float) -> dict:
    tolerance = max(1e-11, 1e-9 * max(abs(h), 1.0))

    def at_h(item) -> bool:
        return abs(float(item[0]["h"]) - h) <= tolerance

    stable = [item for item in cache["stable_pool"] if at_h(item)]
    candidates = [item for item in cache["candidate_pool"] if at_h(item)]
    if not candidates:
        raise RuntimeError(f"No cached candidates found at fixed h={h}")
    return {
        **cache,
        "stable_pool": stable,
        "candidate_pool": candidates,
        "spec_by_h": {key: value for key, value in cache["spec_by_h"].items() if abs(key - h) <= tolerance},
        "first_stable": None,
        "fallback": min(
            candidates,
            key=lambda item: max(float(item[0]["op_score"]), float(item[0]["loc_score"])),
        ),
    }


def normalized_moments(weights: np.ndarray, x: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    mean_w = float(np.mean(weights))
    mean_wx = float(np.mean(weights * x))
    mean_wx2 = float(np.mean(weights * x**2))
    denominator = max(abs(mean_w), 1e-14)
    return {
        "mean_weight": mean_w,
        "mean_weight_x": mean_wx,
        "mean_weight_x2": mean_wx2,
        "normalized_linear_imbalance": abs(mean_wx) / denominator,
        "normalized_quadratic_imbalance": abs(mean_wx2) / denominator,
        "negative_weight_fraction": float(np.mean(weights < 0.0)),
        "mean_absolute_weight": float(np.mean(np.abs(weights))),
        "effective_sample_size": effective_sample_size(weights),
    }


def performance_row(
    result,
    *,
    replicate: int,
    seed: int,
    method: str,
    family: str,
    h: float,
    A: float,
    b: float,
    args: argparse.Namespace,
    moments: dict,
    tuning=None,
) -> dict:
    error = float(result.theta_hat - args.beta)
    row = {
        "replicate": replicate,
        "seed": seed,
        "method": method,
        "family": family,
        "h": h,
        "A": A,
        "b": b,
        "n": args.n,
        "N": args.N,
        "estimate": float(result.theta_hat),
        "error": error,
        "squared_error": error**2,
        "estimated_se": float(result.se),
        "ci_lower": float(result.ci_low),
        "ci_upper": float(result.ci_high),
        "coverage": float(result.ci_low <= args.beta <= result.ci_high),
        "ci_width": float(result.ci_high - result.ci_low),
        "selected_lambda": np.nan,
        "fallback_indicator": np.nan,
        "operator_stability_pass": np.nan,
        "local_leverage_pass": np.nan,
        "bias_screen_pass": np.nan,
        "bias_score": np.nan,
        "bias_budget": np.nan,
        "D_hat": np.nan,
        "Q_hat": np.nan,
        **moments,
    }
    if tuning is not None:
        tr1, tr2 = tuning
        row.update(
            selected_lambda=0.5 * (tr1.lam + tr2.lam),
            fallback_indicator=float("fallback" in tr1.status or "fallback" in tr2.status),
            operator_stability_pass=float(tr1.op_score <= args.tau_op and tr2.op_score <= args.tau_op),
            local_leverage_pass=float(tr1.loc_score <= args.tau_loc and tr2.loc_score <= args.tau_loc),
            bias_screen_pass=float(tr1.bias_score <= tr1.bias_budget and tr2.bias_score <= tr2.bias_budget),
            bias_score=0.5 * (tr1.bias_score + tr2.bias_score),
            bias_budget=0.5 * (tr1.bias_budget + tr2.bias_budget),
            D_hat=0.5 * (tr1.D_h_point + tr2.D_h_point),
            Q_hat=0.5 * (tr1.Q_h + tr2.Q_h),
        )
    return row


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, block in raw.groupby(["A", "b", "h", "method", "family"], sort=True):
        A, b, h, method, family = keys
        empirical_sd = float(block["estimate"].std(ddof=1))
        mean_se = float(block["estimated_se"].mean())
        rows.append(
            {
                "A": A,
                "b": b,
                "h": h,
                "method": method,
                "family": family,
                "repetitions": len(block),
                "coverage": block["coverage"].mean(),
                "bias": block["error"].mean(),
                "absolute_bias": block["error"].abs().mean(),
                "rmse": math.sqrt(block["squared_error"].mean()),
                "empirical_sd": empirical_sd,
                "mean_estimated_se": mean_se,
                "sd_over_mean_se": empirical_sd / mean_se,
                "mean_width": block["ci_width"].mean(),
                "fallback_rate": block["fallback_indicator"].mean(),
                "selected_lambda_mean": block["selected_lambda"].mean(),
                "negative_weight_fraction": block["negative_weight_fraction"].mean(),
                "normalized_linear_imbalance": block["normalized_linear_imbalance"].mean(),
                "normalized_quadratic_imbalance": block["normalized_quadratic_imbalance"].mean(),
                "effective_sample_size": block["effective_sample_size"].mean(),
            }
        )
    return pd.DataFrame(rows)


def pair_gain(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, block in raw.groupby(["A", "b", "h", "family"], sort=True):
        A, b, h, family = keys
        lo = block[block["method"].str.endswith("_LO")].sort_values("replicate")
        ppci = block[block["method"].str.endswith("_PPCI")].sort_values("replicate")
        if len(lo) != len(ppci) or len(lo) < 2:
            continue
        var_lo = float(lo["estimate"].var(ddof=1))
        var_ppci = float(ppci["estimate"].var(ddof=1))
        rows.append(
            {
                "A": A,
                "b": b,
                "h": h,
                "family": family,
                "variance_lo": var_lo,
                "variance_ppci": var_ppci,
                "relative_variance_reduction": 1.0 - var_ppci / var_lo,
            }
        )
    return pd.DataFrame(rows)


def run_replicate(replicate: int, args: argparse.Namespace):
    seed = args.seed + replicate
    rng = np.random.default_rng(seed)
    X_l = rng.uniform(-1.0, 1.0, size=(args.n, 1))
    X_u = rng.uniform(-1.0, 1.0, size=(args.N, 1))
    epsilon_l = rng.normal(0.0, args.sigma, size=args.n)
    X_pilot = rng.uniform(-1.0, 1.0, size=(args.n_pilot, 1))
    epsilon_pilot = rng.normal(0.0, args.sigma, size=args.n_pilot)
    permutation = rng.permutation(args.N)
    I1 = np.sort(permutation[: args.N // 2])
    I2 = np.sort(permutation[args.N // 2 :])
    split = (I1, I2)
    x0 = np.zeros(1)

    base_cfg = JointTuningConfig(
        h_grid_mode="median_grid",
        h_factors=(1.0,),
        lambda_factor_min=args.lambda_factor_min,
        lambda_factor_max=args.lambda_factor_max,
        lambda_grid_size=args.lambda_grid_size,
        lambda_grid_mode="shrinking",
        tau_op=args.tau_op,
        tau_loc=args.tau_loc,
        bias_screen="p1_label",
        c_bias=args.c_bias,
        constraint_fallback="least_violation",
        kernel=args.kernel,
        backend=args.backend_resolved,
    )
    med1 = distance_median_anchor(X_u[I1], x0)
    med2 = distance_median_anchor(X_u[I2], x0)
    cfg1 = replace(base_cfg, h_factors=tuple(h / med1 for h in args.h_values))
    cfg2 = replace(base_cfg, h_factors=tuple(h / med2 for h in args.h_values))
    cache1 = collect_joint_candidate_cache(X_u[I1], x0, n=args.n, cfg=cfg1)
    cache2 = collect_joint_candidate_cache(X_u[I2], x0, n=args.n, cfg=cfg2)

    weights_by_h = {}
    moment_rows = []
    curves = []
    kernel = get_kernel(args.kernel)
    x_curve = np.linspace(-1.0, 1.0, args.curve_points).reshape(-1, 1)

    for h in args.h_values:
        sub1 = fixed_h_subcache(cache1, h)
        sub2 = fixed_h_subcache(cache2, h)
        tr1 = select_joint_from_cache(sub1, "GH", cfg=base_cfg)
        tr2 = select_joint_from_cache(sub2, "GH", cfg=base_cfg)
        w1 = weight_from_joint_cache(sub1, tr1, base_cfg)
        w2 = weight_from_joint_cache(sub2, tr2, base_cfg)
        w_l_localization = 0.5 * (w1(X_l) + w2(X_l))
        w_u_localization = np.empty(args.N, dtype=float)
        w_u_localization[I1] = w2(X_u[I1])
        w_u_localization[I2] = w1(X_u[I2])
        w_l_nw, w_u_nw, nw_info = nw_twofold_weights(X_l, X_u, h, split, args.kernel)
        nw_moments = normalized_moments(w_u_nw, X_u)
        localization_moments = normalized_moments(w_u_localization, X_u)
        weights_by_h[h] = (w_l_nw, w_u_nw, nw_moments, w_l_localization, w_u_localization, localization_moments, tr1, tr2)
        moment_rows.extend(
            [
                {"replicate": replicate, "h": h, "family": "NW", "selected_lambda": np.nan, **nw_moments},
                {"replicate": replicate, "h": h, "family": "RKHS_LOCALIZATION", "selected_lambda": 0.5 * (tr1.lam + tr2.lam), **localization_moments},
            ]
        )

        if replicate == 0:
            base_curve = kernel(x_curve, np.zeros((1, 1)), h).reshape(-1)
            nw_curve = 0.5 * (base_curve / nw_info["nw_denominator_fold1"] + base_curve / nw_info["nw_denominator_fold2"])
            localization_curve = 0.5 * (w1(x_curve) + w2(x_curve))
            for family, values in [("NW", nw_curve), ("RKHS_LOCALIZATION", localization_curve), ("WSTAR_REFERENCE", w_star(x_curve))]:
                curves.extend(
                    {"h": h, "x": float(x), "family": family, "weight": float(value)}
                    for x, value in zip(x_curve.reshape(-1), values)
                )

    performance = []
    for A in args.A_values:
        for b in args.b_values:
            f_l = m_function(X_l, args.beta, b, A)
            f_u = m_function(X_u, args.beta, b, A)
            Y_l = f_l + epsilon_l
            for h, values in weights_by_h.items():
                w_l_nw, w_u_nw, nw_moments, w_l_localization, w_u_localization, localization_moments, tr1, tr2 = values
                nw_lo = lo_mean_from_weights(X_l, Y_l, w_l_nw, z_alpha=args.z_alpha)
                nw_ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l_nw, w_u_nw, z_alpha=args.z_alpha)
                localization_lo = lo_mean_from_weights(X_l, Y_l, w_l_localization, z_alpha=args.z_alpha)
                localization_ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l_localization, w_u_localization, z_alpha=args.z_alpha)
                performance.extend(
                    [
                        performance_row(nw_lo, replicate=replicate, seed=seed, method="NW_FIXED_H_LO", family="NW", h=h, A=A, b=b, args=args, moments=nw_moments),
                        performance_row(nw_ppci, replicate=replicate, seed=seed, method="NW_FIXED_H_PPCI", family="NW", h=h, A=A, b=b, args=args, moments=nw_moments),
                        performance_row(localization_lo, replicate=replicate, seed=seed, method="RKHS_LOCALIZATION_FIXED_H_LO", family="RKHS_LOCALIZATION", h=h, A=A, b=b, args=args, moments=localization_moments, tuning=(tr1, tr2)),
                        performance_row(localization_ppci, replicate=replicate, seed=seed, method="RKHS_LOCALIZATION_FIXED_H_PPCI", family="RKHS_LOCALIZATION", h=h, A=A, b=b, args=args, moments=localization_moments, tuning=(tr1, tr2)),
                    ]
                )

    grid_rows = []
    old_grid = np.geomspace(args.nw_h_min, args.nw_h_max, args.nw_h_grid_size)
    extended_grid = np.geomspace(args.nw_extended_h_min, args.nw_h_max, args.nw_extended_grid_size)
    for b in args.b_values:
        A = args.main_A
        f_l = m_function(X_l, args.beta, b, A)
        f_u = m_function(X_u, args.beta, b, A)
        Y_l = f_l + epsilon_l
        Y_pilot = m_function(X_pilot, args.beta, b, A) + epsilon_pilot
        for grid_name, grid in [("current", old_grid), ("extended", extended_grid)]:
            h_cv, objective, invalid = nw_cv_bandwidth(X_pilot, Y_pilot, grid, args.kernel)
            h_us_raw = h_cv * args.n ** (-args.undersmooth_delta)
            h_us = float(np.clip(h_us_raw, float(grid.min()), float(grid.max())))
            for rule, h_selected in [("CV", h_cv), ("US", h_us)]:
                w_l, w_u, _ = nw_twofold_weights(X_l, X_u, h_selected, split, args.kernel)
                result = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, z_alpha=args.z_alpha)
                error = result.theta_hat - args.beta
                grid_rows.append(
                    {
                        "replicate": replicate,
                        "b": b,
                        "grid": grid_name,
                        "rule": rule,
                        "selected_h": h_selected,
                        "lower_bound_hit": float(abs(h_cv - float(grid.min())) <= 1e-12),
                        "us_clipped": float(rule == "US" and abs(h_selected - h_us_raw) > 1e-12),
                        "cv_objective": objective,
                        "invalid_candidates": invalid,
                        "coverage": float(result.ci_low <= args.beta <= result.ci_high),
                        "error": error,
                        "squared_error": error**2,
                        "width": result.ci_high - result.ci_low,
                    }
                )

    return performance, moment_rows, grid_rows, curves


def grid_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, block in raw.groupby(["b", "grid", "rule"], sort=True):
        b, grid, rule = keys
        rows.append(
            {
                "b": b,
                "grid": grid,
                "rule": rule,
                "coverage": block["coverage"].mean(),
                "bias": block["error"].mean(),
                "rmse": math.sqrt(block["squared_error"].mean()),
                "width": block["width"].mean(),
                "selected_h_mean": block["selected_h"].mean(),
                "selected_h_median": block["selected_h"].median(),
                "lower_bound_hit_rate": block["lower_bound_hit"].mean(),
                "us_clipping_rate": block["us_clipped"].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def make_figures(summary: pd.DataFrame, moments: pd.DataFrame, curves: pd.DataFrame, grid: pd.DataFrame, args, output: Path) -> None:
    colors = {"NW": "#c43d3d", "RKHS_LOCALIZATION": "#1f4e79"}
    markers = {"NW": "o", "RKHS_LOCALIZATION": "s"}
    main = summary[(summary["A"] == args.main_A) & summary["method"].str.endswith("PPCI")]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ["absolute_bias", "rmse", "coverage", "mean_width"], ["Absolute bias", "RMSE", "Coverage", "Average CI width"]):
        for (family, b), block in main.groupby(["family", "b"]):
            block = block.sort_values("h")
            ax.plot(block["h"].to_numpy(), block[metric].to_numpy(), marker=markers[family], color=colors[family], linestyle="-" if b == 0 else "--", label=f"{family}, b={b:g}")
        ax.set(xlabel="Common fixed bandwidth h", ylabel=title)
        ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].axhline(0.95, color="#555555", linestyle=":", linewidth=1)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "same_h_performance_A4")

    moment_summary = moments.groupby(["h", "family"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for family, block in moment_summary.groupby("family"):
        block = block.sort_values("h")
        axes[0].plot(block["h"].to_numpy(), block["normalized_linear_imbalance"].to_numpy(), marker=markers[family], color=colors[family], label=family)
        axes[1].plot(block["h"].to_numpy(), block["normalized_quadratic_imbalance"].to_numpy(), marker=markers[family], color=colors[family], label=family)
    axes[0].set(xlabel="Common fixed bandwidth h", ylabel="Normalized |P(wX)|", title="Linear balance")
    axes[1].set(xlabel="Common fixed bandwidth h", ylabel="Normalized |P(wX^2)|", title="Quadratic balance")
    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    save_figure(fig, output, "moment_balance_path")

    selected_curves = curves[curves["h"].isin([min(args.h_values), max(args.h_values)])]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for ax, h in zip(axes, [min(args.h_values), max(args.h_values)]):
        for family, block in selected_curves[selected_curves["h"] == h].groupby("family"):
            style = {"NW": ("#c43d3d", "-", "NW"), "RKHS_LOCALIZATION": ("#1f4e79", "-", "RKHS localization"), "WSTAR_REFERENCE": ("#2f7d4a", ":", "Finite-dimensional w* reference")}[family]
            ax.plot(block["x"].to_numpy(), block["weight"].to_numpy(), color=style[0], linestyle=style[1], label=style[2])
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set(title=f"Common h = {h:g}", xlabel="x", ylabel="Weight")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "representative_weight_functions")

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), constrained_layout=True)
    for row_index, b in enumerate(sorted(main["b"].unique())):
        for family, block in main[main["b"] == b].groupby("family"):
            block = block.sort_values("h")
            axes[row_index, 0].plot(block["absolute_bias"].to_numpy(), block["rmse"].to_numpy(), marker=markers[family], color=colors[family], label=family)
            axes[row_index, 1].plot(block["coverage"].to_numpy(), block["mean_width"].to_numpy(), marker=markers[family], color=colors[family], label=family)
            for point_index, point in enumerate(block.itertuples()):
                offset = (4, 4) if family == "NW" else (4, -10)
                axes[row_index, 0].annotate(f"h={point.h:g}", (point.absolute_bias, point.rmse), xytext=offset, textcoords="offset points", fontsize=7)
                axes[row_index, 1].annotate(f"h={point.h:g}", (point.coverage, point.mean_width), xytext=offset, textcoords="offset points", fontsize=7)
        axes[row_index, 0].set(xlabel="Absolute bias", ylabel="RMSE", title=f"P1-constrained fixed-h frontier, b={b:g}")
        axes[row_index, 1].set(xlabel="Coverage", ylabel="Average CI width", title=f"Coverage-width frontier, b={b:g}")
        axes[row_index, 1].axvline(0.95, color="#555555", linestyle=":", linewidth=1)
        for ax in axes[row_index]:
            ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "fixed_h_frontiers")

    stress = summary[(summary["h"] == max(args.h_values)) & summary["method"].str.endswith("PPCI")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for (family, b), block in stress.groupby(["family", "b"]):
        block = block.sort_values("A")
        axes[0].plot(block["A"].to_numpy(), block["absolute_bias"].to_numpy(), marker=markers[family], color=colors[family], linestyle="-" if b == 0 else "--", label=f"{family}, b={b:g}")
        axes[1].plot(block["A"].to_numpy(), block["rmse"].to_numpy(), marker=markers[family], color=colors[family], linestyle="-" if b == 0 else "--", label=f"{family}, b={b:g}")
    axes[0].set(xlabel="Curvature A", ylabel="Absolute bias", title=f"Curvature stress at h={max(args.h_values):g}")
    axes[1].set(xlabel="Curvature A", ylabel="RMSE", title=f"Curvature stress at h={max(args.h_values):g}")
    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "curvature_stress_h06")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for b, block in grid.groupby("b"):
        for grid_name, style in [("current", "--"), ("extended", "-")]:
            part = block[block["grid"] == grid_name].sort_values("rule")
            axes[0].plot(part["selected_h_mean"].to_numpy(), part["rmse"].to_numpy(), "o", linestyle=style, label=f"b={b:g}, {grid_name}")
            axes[1].plot(part["coverage"].to_numpy(), part["width"].to_numpy(), "o", linestyle=style, label=f"b={b:g}, {grid_name}")
    axes[0].set(xlabel="Mean selected h", ylabel="RMSE", title="NW grid sensitivity")
    axes[1].set(xlabel="Coverage", ylabel="Width", title="NW grid sensitivity")
    axes[1].axvline(0.95, color="#555555", linestyle=":", linewidth=1)
    for ax in axes:
        ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    save_figure(fig, output, "nw_lower_grid_sensitivity")


def write_readme(summary, moments, grid, args, output):
    main = summary[(summary["A"] == args.main_A) & summary["method"].str.endswith("PPCI")][
        ["b", "h", "family", "coverage", "bias", "rmse", "mean_width", "normalized_quadratic_imbalance"]
    ]
    at_wide = summary[(summary["h"] == max(args.h_values)) & summary["method"].str.endswith("PPCI")][
        ["A", "b", "family", "coverage", "bias", "rmse", "mean_width"]
    ]
    text = f"""# Same-Bandwidth NW versus RKHS localization Follow-up

This follow-up fixes the same Matérn-5/2 base bandwidth `h in {args.h_values}` for NW and RKHS localization weights. At every fixed h, RKHS localization selects only lambda using the existing stability screens, P1 labelled-scale budget `c_bias={args.c_bias:g}`, and least-normalized-violation fallback. Thus bandwidth selection cannot explain the comparison.

All methods use paired samples with `n={args.n}`, `N={args.N}`, and `{args.reps}` replicates. The main curvature remains `A={args.main_A:g}`; `A in {args.A_values}` is a predeclared mechanism stress test. The finite-dimensional `w*` appears only in the representative weight plot.

## Main A=4 Same-h Results

{main.to_markdown(index=False, floatfmt='.4f')}

## Curvature Stress at the Widest Common h

{at_wide.to_markdown(index=False, floatfmt='.4f')}

## NW Lower-grid Robustness

{grid.to_markdown(index=False, floatfmt='.4f')}

The key diagnostic is `normalized_quadratic_imbalance = |P_N(w X^2) / P_N(w)|`. It directly measures the residual curvature bias channel. The frontier is explicitly a P1-constrained fixed-h path, not an exhaustive unconstrained lambda search.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/nw_localization_same_h_followup_v1")
    parser.add_argument("--seed", type=int, default=35790)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "auto", "torch", "gpu", "cuda"])
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--n-pilot", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--A-values", default="2,4,8")
    parser.add_argument("--main-A", type=float, default=4.0)
    parser.add_argument("--b-values", default="0,2")
    parser.add_argument("--h-values", default="0.08,0.20,0.40,0.60")
    parser.add_argument("--kernel", default="matern52")
    parser.add_argument("--lambda-factor-min", type=float, default=0.05)
    parser.add_argument("--lambda-factor-max", type=float, default=20.0)
    parser.add_argument("--lambda-grid-size", type=int, default=41)
    parser.add_argument("--tau-op", type=float, default=12.0)
    parser.add_argument("--tau-loc", type=float, default=4.0)
    parser.add_argument("--c-bias", type=float, default=0.18)
    parser.add_argument("--nw-h-min", type=float, default=0.05)
    parser.add_argument("--nw-extended-h-min", type=float, default=0.02)
    parser.add_argument("--nw-h-max", type=float, default=0.8)
    parser.add_argument("--nw-h-grid-size", type=int, default=35)
    parser.add_argument("--nw-extended-grid-size", type=int, default=45)
    parser.add_argument("--undersmooth-delta", type=float, default=0.05)
    parser.add_argument("--curve-points", type=int, default=401)
    parser.add_argument("--z-alpha", type=float, default=1.959963984540054)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.A_values = parse_floats(args.A_values)
    args.b_values = parse_floats(args.b_values)
    args.h_values = parse_floats(args.h_values)
    if args.main_A not in args.A_values:
        raise ValueError("main A must be included in A-values")
    if args.smoke:
        args.reps = 10
        args.workers = min(args.workers, 8)
    args.backend_resolved = configure_backend(args.backend, args.gpu_id)
    if args.backend_resolved != "cpu":
        args.workers = 1

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.workers > 1:
        from joblib import Parallel, delayed

        blocks = Parallel(n_jobs=args.workers, backend="loky", verbose=10)(
            delayed(run_replicate)(replicate, args) for replicate in range(args.reps)
        )
    else:
        blocks = [run_replicate(replicate, args) for replicate in range(args.reps)]
    elapsed = time.perf_counter() - start

    performance = pd.DataFrame([row for block in blocks for row in block[0]])
    moments = pd.DataFrame([row for block in blocks for row in block[1]])
    grid_raw = pd.DataFrame([row for block in blocks for row in block[2]])
    curves = pd.DataFrame([row for block in blocks for row in block[3]])
    summary = summarize(performance)
    gains = pair_gain(performance)
    moment_summary = moments.groupby(["h", "family"], as_index=False).mean(numeric_only=True)
    nw_grid_summary = grid_summary(grid_raw)

    expected_rows = args.reps * len(args.A_values) * len(args.b_values) * len(args.h_values) * 4
    assert len(performance) == expected_rows
    assert np.isfinite(performance[["estimate", "estimated_se", "ci_lower", "ci_upper"]]).all().all()
    assert (performance.loc[performance["family"] == "RKHS_LOCALIZATION", "negative_weight_fraction"] > 0).any()

    performance.to_csv(output / "raw_replicates.csv", index=False)
    summary.to_csv(output / "same_h_summary.csv", index=False)
    gains.to_csv(output / "paired_gain_summary.csv", index=False)
    moments.to_csv(output / "moment_balance_raw.csv", index=False)
    moment_summary.to_csv(output / "moment_balance_summary.csv", index=False)
    curves.to_csv(output / "representative_weights.csv", index=False)
    grid_raw.to_csv(output / "nw_grid_sensitivity_raw.csv", index=False)
    nw_grid_summary.to_csv(output / "nw_grid_sensitivity_summary.csv", index=False)
    config = vars(args).copy(); config["output_dir"] = str(config["output_dir"]); config["elapsed_seconds"] = elapsed
    config["source_sha256"], config["source_file_count"] = source_sha256()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_figures(summary, moments, curves, nw_grid_summary, args, output)
    write_readme(summary, moments, nw_grid_summary, args, output)
    print(summary[(summary["A"] == args.main_A) & summary["method"].str.endswith("PPCI")].to_string(index=False))
    print(f"Saved follow-up to {output} in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
