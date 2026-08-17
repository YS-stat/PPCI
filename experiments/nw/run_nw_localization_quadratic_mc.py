#!/usr/bin/env python3
"""Monte Carlo comparison of NW and kernel RKHS localization PPCI."""

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
from ppci_condmean.utils import source_sha256


MAIN_METHODS = [
    "NW_LO_CV",
    "NW_PPCI_CV",
    "NW_LO_US",
    "NW_PPCI_US",
    "RKHS_LOCALIZATION_LO_GH_P1",
    "RKHS_LOCALIZATION_PPCI_GH_P1",
    "ORACLE_WSTAR_LO",
    "ORACLE_WSTAR_PPCI",
]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def m_function(x: np.ndarray, beta: float, b: float, curvature: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return beta + b * x + curvature * x**2


def w_star(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return 9.0 / 4.0 - 15.0 * x**2 / 4.0


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    denominator = float(np.sum(weights**2))
    return float(np.sum(weights) ** 2 / denominator) if denominator > 1e-14 else np.nan


def weight_diagnostics(w_l: np.ndarray, w_u: np.ndarray) -> dict:
    combined = np.concatenate([np.asarray(w_l).reshape(-1), np.asarray(w_u).reshape(-1)])
    return {
        "negative_weight_fraction": float(np.mean(combined < 0.0)),
        "mean_absolute_weight": float(np.mean(np.abs(combined))),
        "effective_sample_size": effective_sample_size(combined),
    }


def nw_cv_bandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    h_grid: np.ndarray,
    kernel_name: str,
) -> tuple[float, float, int]:
    kernel = get_kernel(kernel_name)
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    objectives = np.full(len(h_grid), np.inf)
    invalid = 0
    for index, h in enumerate(h_grid):
        matrix = kernel(X, X, float(h))
        np.fill_diagonal(matrix, 0.0)
        denominator = matrix.sum(axis=1)
        if np.any(denominator <= 1e-12) or not np.isfinite(denominator).all():
            invalid += 1
            continue
        prediction = matrix @ Y / denominator
        objectives[index] = float(np.mean((Y - prediction) ** 2))
    if not np.isfinite(objectives).any():
        raise RuntimeError("All NW-CV bandwidth candidates are invalid")
    selected = int(np.nanargmin(objectives))
    return float(h_grid[selected]), float(objectives[selected]), invalid


def nw_twofold_weights(
    X_l: np.ndarray,
    X_u: np.ndarray,
    h: float,
    split: tuple[np.ndarray, np.ndarray],
    kernel_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    kernel = get_kernel(kernel_name)
    x0 = np.zeros((1, 1))
    X_l = np.asarray(X_l, dtype=float).reshape(-1, 1)
    X_u = np.asarray(X_u, dtype=float).reshape(-1, 1)
    I1, I2 = split
    base_l = kernel(X_l, x0, h).reshape(-1)
    base_u = kernel(X_u, x0, h).reshape(-1)
    mu1 = float(np.mean(base_u[I1]))
    mu2 = float(np.mean(base_u[I2]))
    if min(mu1, mu2) <= 1e-12 or not np.isfinite([mu1, mu2]).all():
        raise RuntimeError(f"Unstable NW denominator at h={h}: {mu1}, {mu2}")
    w_l = 0.5 * (base_l / mu1 + base_l / mu2)
    w_u = np.empty_like(base_u)
    w_u[I1] = base_u[I1] / mu2
    w_u[I2] = base_u[I2] / mu1
    diagnostics = {
        "nw_denominator": 0.5 * (mu1 + mu2),
        "nw_denominator_fold1": mu1,
        "nw_denominator_fold2": mu2,
        "nw_local_labeled_ess": effective_sample_size(base_l),
        "nw_local_unlabeled_ess": effective_sample_size(base_u),
    }
    return w_l, w_u, diagnostics


def uniform_nw_twofold_weights(
    X_l: np.ndarray,
    X_u: np.ndarray,
    h: float,
    split: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    base_l = (np.abs(np.asarray(X_l).reshape(-1)) <= h).astype(float)
    base_u = (np.abs(np.asarray(X_u).reshape(-1)) <= h).astype(float)
    I1, I2 = split
    mu1 = float(np.mean(base_u[I1]))
    mu2 = float(np.mean(base_u[I2]))
    if min(mu1, mu2) <= 1e-12:
        raise RuntimeError(f"Unstable uniform-NW denominator at h={h}")
    w_l = 0.5 * (base_l / mu1 + base_l / mu2)
    w_u = np.empty_like(base_u)
    w_u[I1] = base_u[I1] / mu2
    w_u[I2] = base_u[I2] / mu1
    return w_l, w_u


def result_row(
    result,
    *,
    method: str,
    replicate: int,
    setting: str,
    seed: int,
    b: float,
    args: argparse.Namespace,
    selected_h: float = np.nan,
    selected_lambda: float = np.nan,
    selected_c_bias: float = np.nan,
    diagnostics: dict | None = None,
) -> dict:
    diagnostics = diagnostics or {}
    estimate = float(result.theta_hat)
    error = estimate - args.beta
    return {
        "setting": setting,
        "replicate": replicate,
        "seed": seed,
        "method": method,
        "b": b,
        "A": args.curvature,
        "sigma": args.sigma,
        "n": args.n,
        "N": args.N,
        "theta0": args.beta,
        "estimate": estimate,
        "error": error,
        "squared_error": error**2,
        "estimated_se": float(result.se),
        "ci_lower": float(result.ci_low),
        "ci_upper": float(result.ci_high),
        "coverage": float(result.ci_low <= args.beta <= result.ci_high),
        "ci_width": float(result.ci_high - result.ci_low),
        "selected_h": selected_h,
        "selected_lambda": selected_lambda,
        "selected_c_bias": selected_c_bias,
        "fallback_indicator": diagnostics.get("fallback_indicator", np.nan),
        "operator_stability_pass": diagnostics.get("operator_stability_pass", np.nan),
        "local_leverage_pass": diagnostics.get("local_leverage_pass", np.nan),
        "bias_screen_pass": diagnostics.get("bias_screen_pass", np.nan),
        "D_hat": diagnostics.get("D_hat", np.nan),
        "Q_hat": diagnostics.get("Q_hat", np.nan),
        "negative_weight_fraction": diagnostics.get("negative_weight_fraction", np.nan),
        "mean_absolute_weight": diagnostics.get("mean_absolute_weight", np.nan),
        "effective_sample_size": diagnostics.get("effective_sample_size", np.nan),
        "nw_denominator": diagnostics.get("nw_denominator", np.nan),
        "nw_denominator_fold1": diagnostics.get("nw_denominator_fold1", np.nan),
        "nw_denominator_fold2": diagnostics.get("nw_denominator_fold2", np.nan),
        "nw_local_labeled_ess": diagnostics.get("nw_local_labeled_ess", np.nan),
        "nw_local_unlabeled_ess": diagnostics.get("nw_local_unlabeled_ess", np.nan),
        "nw_cv_objective": diagnostics.get("nw_cv_objective", np.nan),
        "nw_cv_invalid_candidates": diagnostics.get("nw_cv_invalid_candidates", np.nan),
        "h_us_clipped": diagnostics.get("h_us_clipped", np.nan),
        "op_score": diagnostics.get("op_score", np.nan),
        "loc_score": diagnostics.get("loc_score", np.nan),
        "bias_score": diagnostics.get("bias_score", np.nan),
        "bias_budget": diagnostics.get("bias_budget", np.nan),
        "tuning_status": diagnostics.get("tuning_status", ""),
    }


def method_label(prefix: str, c_bias: float, main_c_bias: float) -> str:
    if abs(c_bias - main_c_bias) < 1e-12:
        return prefix
    return f"{prefix}_c{c_bias:g}"


def run_replicate(replicate: int, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    seed = int(args.seed + replicate)
    rng = np.random.default_rng(seed)
    X_l = rng.uniform(-1.0, 1.0, size=(args.n, 1))
    X_u = rng.uniform(-1.0, 1.0, size=(args.N, 1))
    epsilon_l = rng.normal(0.0, args.sigma, size=args.n)
    X_pilot = rng.uniform(-1.0, 1.0, size=(args.n_pilot, 1))
    epsilon_pilot = rng.normal(0.0, args.sigma, size=args.n_pilot)
    perm = rng.permutation(args.N)
    I1 = np.sort(perm[: args.N // 2])
    I2 = np.sort(perm[args.N // 2 :])
    split = (I1, I2)
    x0 = np.zeros(1)

    base_cfg = JointTuningConfig(
        h_grid_mode="median_grid",
        h_factors=tuple(args.localization_h_factors),
        lambda_factor_min=args.lambda_factor_min,
        lambda_factor_max=args.lambda_factor_max,
        lambda_grid_size=args.lambda_grid_size,
        lambda_grid_mode="shrinking",
        tau_op=args.tau_op,
        tau_loc=args.tau_loc,
        bias_screen="p1_label",
        c_bias=args.main_c_bias,
        constraint_fallback="least_violation",
        kernel=args.kernel,
        backend=args.backend_resolved,
    )
    cache1 = collect_joint_candidate_cache(X_u[I1], x0, n=args.n, cfg=base_cfg)
    cache2 = collect_joint_candidate_cache(X_u[I2], x0, n=args.n, cfg=base_cfg)

    localization_weights: dict[float, tuple[np.ndarray, np.ndarray, object, object]] = {}
    weight_cache: dict[tuple, object] = {}
    for c_bias in args.c_bias_values:
        cfg = replace(base_cfg, c_bias=float(c_bias))
        tr1 = select_joint_from_cache(cache1, "GH", cfg=cfg)
        tr2 = select_joint_from_cache(cache2, "GH", cfg=cfg)

        def get_weight(fold: int, cache: dict, tuning):
            key = (fold, float(tuning.h), float(tuning.lam))
            if key not in weight_cache:
                weight_cache[key] = weight_from_joint_cache(cache, tuning, cfg)
            return weight_cache[key]

        w1 = get_weight(1, cache1, tr1)
        w2 = get_weight(2, cache2, tr2)
        w_l = 0.5 * (w1(X_l) + w2(X_l))
        w_u = np.empty(args.N, dtype=float)
        w_u[I1] = w2(X_u[I1])
        w_u[I2] = w1(X_u[I2])
        localization_weights[float(c_bias)] = (w_l, w_u, tr1, tr2)

    rows: list[dict] = []
    fixed_rows: list[dict] = []
    h_grid = np.geomspace(args.nw_h_min, args.nw_h_max, args.nw_h_grid_size)

    for b in args.b_values:
        setting = f"b={b:g}"
        f_l = m_function(X_l, args.beta, b, args.curvature)
        f_u = m_function(X_u, args.beta, b, args.curvature)
        Y_l = f_l + epsilon_l
        Y_pilot = m_function(X_pilot, args.beta, b, args.curvature) + epsilon_pilot

        h_cv, cv_objective, invalid_cv = nw_cv_bandwidth(X_pilot, Y_pilot, h_grid, args.kernel)
        h_us_raw = h_cv * args.n ** (-args.undersmooth_delta)
        h_us = float(np.clip(h_us_raw, args.nw_h_min, args.nw_h_max))
        clipped = bool(abs(h_us - h_us_raw) > 1e-12)

        for suffix, h, is_us in [("CV", h_cv, False), ("US", h_us, True)]:
            w_l, w_u, nw_diag = nw_twofold_weights(X_l, X_u, h, split, args.kernel)
            nw_diag.update(weight_diagnostics(w_l, w_u))
            nw_diag.update(
                nw_cv_objective=cv_objective,
                nw_cv_invalid_candidates=invalid_cv,
                h_us_clipped=clipped if is_us else False,
            )
            lo = lo_mean_from_weights(X_l, Y_l, w_l, z_alpha=args.z_alpha)
            ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, z_alpha=args.z_alpha)
            rows.append(result_row(lo, method=f"NW_LO_{suffix}", replicate=replicate, setting=setting, seed=seed, b=b, args=args, selected_h=h, diagnostics=nw_diag))
            rows.append(result_row(ppci, method=f"NW_PPCI_{suffix}", replicate=replicate, setting=setting, seed=seed, b=b, args=args, selected_h=h, diagnostics=nw_diag))

        for c_bias, (w_l, w_u, tr1, tr2) in localization_weights.items():
            diag = weight_diagnostics(w_l, w_u)
            diag.update(
                fallback_indicator=float("fallback" in tr1.status or "fallback" in tr2.status),
                operator_stability_pass=float(tr1.op_score <= args.tau_op and tr2.op_score <= args.tau_op),
                local_leverage_pass=float(tr1.loc_score <= args.tau_loc and tr2.loc_score <= args.tau_loc),
                bias_screen_pass=float(tr1.bias_score <= tr1.bias_budget and tr2.bias_score <= tr2.bias_budget),
                D_hat=0.5 * (tr1.D_h_point + tr2.D_h_point),
                Q_hat=0.5 * (tr1.Q_h + tr2.Q_h),
                op_score=0.5 * (tr1.op_score + tr2.op_score),
                loc_score=0.5 * (tr1.loc_score + tr2.loc_score),
                bias_score=0.5 * (tr1.bias_score + tr2.bias_score),
                bias_budget=0.5 * (tr1.bias_budget + tr2.bias_budget),
                tuning_status=f"fold1:{tr1.status};fold2:{tr2.status}",
            )
            selected_h = 0.5 * (tr1.h + tr2.h)
            selected_lambda = 0.5 * (tr1.lam + tr2.lam)
            lo = lo_mean_from_weights(X_l, Y_l, w_l, z_alpha=args.z_alpha)
            ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, z_alpha=args.z_alpha)
            lo_label = method_label("RKHS_LOCALIZATION_LO_GH_P1", c_bias, args.main_c_bias)
            ppci_label = method_label("RKHS_LOCALIZATION_PPCI_GH_P1", c_bias, args.main_c_bias)
            rows.append(result_row(lo, method=lo_label, replicate=replicate, setting=setting, seed=seed, b=b, args=args, selected_h=selected_h, selected_lambda=selected_lambda, selected_c_bias=c_bias, diagnostics=diag))
            rows.append(result_row(ppci, method=ppci_label, replicate=replicate, setting=setting, seed=seed, b=b, args=args, selected_h=selected_h, selected_lambda=selected_lambda, selected_c_bias=c_bias, diagnostics=diag))

        w_l_star = w_star(X_l)
        w_u_star = w_star(X_u)
        oracle_diag = weight_diagnostics(w_l_star, w_u_star)
        lo_star = lo_mean_from_weights(X_l, Y_l, w_l_star, z_alpha=args.z_alpha)
        ppci_star = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l_star, w_u_star, z_alpha=args.z_alpha)
        rows.append(result_row(lo_star, method="ORACLE_WSTAR_LO", replicate=replicate, setting=setting, seed=seed, b=b, args=args, diagnostics=oracle_diag))
        rows.append(result_row(ppci_star, method="ORACLE_WSTAR_PPCI", replicate=replicate, setting=setting, seed=seed, b=b, args=args, diagnostics=oracle_diag))

        fixed_w_l, fixed_w_u = uniform_nw_twofold_weights(X_l, X_u, args.fixed_h_check, split)
        fixed = ppci_mean_from_weight_values(Y_l, f_l, f_u, fixed_w_l, fixed_w_u, z_alpha=args.z_alpha)
        fixed_rows.append(
            {
                "setting": setting,
                "replicate": replicate,
                "b": b,
                "h": args.fixed_h_check,
                "estimate": fixed.theta_hat,
                "error": fixed.theta_hat - args.beta,
                "analytic_uniform_bias": args.curvature * args.fixed_h_check**2 / 3.0,
                "note": "Uniform NW fixed-bandwidth sanity check",
            }
        )

    assert len(rows) == 2 * (4 + 2 * len(args.c_bias_values) + 2)
    numeric = pd.DataFrame(rows).select_dtypes(include="number")
    assert np.isfinite(numeric[["estimate", "estimated_se", "ci_lower", "ci_upper"]]).all().all()
    return rows, fixed_rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (setting, method), block in raw.groupby(["setting", "method"], sort=True):
        empirical_sd = float(block["estimate"].std(ddof=1))
        mean_se = float(block["estimated_se"].mean())
        records.append(
            {
                "setting": setting,
                "method": method,
                "repetitions": len(block),
                "coverage": block["coverage"].mean(),
                "bias": block["error"].mean(),
                "absolute_bias": block["error"].abs().mean(),
                "rmse": math.sqrt(block["squared_error"].mean()),
                "empirical_sd": empirical_sd,
                "mean_estimated_se": mean_se,
                "sd_over_mean_se": empirical_sd / mean_se if mean_se > 0 else np.nan,
                "mean_width": block["ci_width"].mean(),
                "median_width": block["ci_width"].median(),
                "fallback_rate": block["fallback_indicator"].mean(),
                "selected_h_mean": block["selected_h"].mean(),
                "selected_h_median": block["selected_h"].median(),
                "selected_h_q10": block["selected_h"].quantile(0.10),
                "selected_h_q90": block["selected_h"].quantile(0.90),
                "selected_lambda_mean": block["selected_lambda"].mean(),
                "selected_lambda_median": block["selected_lambda"].median(),
                "selected_lambda_q10": block["selected_lambda"].quantile(0.10),
                "selected_lambda_q90": block["selected_lambda"].quantile(0.90),
                "negative_weight_fraction": block["negative_weight_fraction"].mean(),
                "D_hat": block["D_hat"].mean(),
                "Q_hat": block["Q_hat"].mean(),
                "effective_sample_size": block["effective_sample_size"].mean(),
            }
        )
    return pd.DataFrame(records)


def paired_gain_summary(raw: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "setting",
        "pair",
        "lo_method",
        "ppci_method",
        "empirical_variance_lo",
        "empirical_variance_ppci",
        "relative_variance_reduction",
        "paired_error_correlation",
    ]
    pairs = [
        ("NW_LO_CV", "NW_PPCI_CV", "NW_CV"),
        ("NW_LO_US", "NW_PPCI_US", "NW_US"),
        ("RKHS_LOCALIZATION_LO_GH_P1", "RKHS_LOCALIZATION_PPCI_GH_P1", "RKHS_LOCALIZATION_GH_P1"),
        ("ORACLE_WSTAR_LO", "ORACLE_WSTAR_PPCI", "ORACLE_WSTAR"),
    ]
    rows = []
    for setting, setting_data in raw.groupby("setting"):
        for lo_method, ppci_method, label in pairs:
            lo = setting_data[setting_data["method"] == lo_method].sort_values("replicate")
            ppci = setting_data[setting_data["method"] == ppci_method].sort_values("replicate")
            if len(lo) != len(ppci) or len(lo) < 2:
                continue
            var_lo = float(lo["estimate"].var(ddof=1))
            var_ppci = float(ppci["estimate"].var(ddof=1))
            rows.append(
                {
                    "setting": setting,
                    "pair": label,
                    "lo_method": lo_method,
                    "ppci_method": ppci_method,
                    "empirical_variance_lo": var_lo,
                    "empirical_variance_ppci": var_ppci,
                    "relative_variance_reduction": 1.0 - var_ppci / var_lo,
                    "paired_error_correlation": np.corrcoef(lo["error"], ppci["error"])[0, 1],
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def selection_summary(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "selected_h",
        "selected_lambda",
        "fallback_indicator",
        "operator_stability_pass",
        "local_leverage_pass",
        "bias_screen_pass",
        "negative_weight_fraction",
        "mean_absolute_weight",
        "effective_sample_size",
        "D_hat",
        "Q_hat",
        "nw_denominator",
        "nw_local_labeled_ess",
        "nw_local_unlabeled_ess",
        "nw_cv_objective",
        "h_us_clipped",
    ]
    return raw.groupby(["setting", "method"], as_index=False)[columns].mean(numeric_only=True)


def analytic_oracle_variance(args: argparse.Namespace, b: float, ppci: bool) -> float:
    signal = 9.0 * b**2 / 28.0 + 23.0 * args.curvature**2 / 140.0
    if ppci:
        return 9.0 * args.sigma**2 / (4.0 * args.n) + signal / args.N
    return (9.0 * args.sigma**2 / 4.0 + signal) / args.n


def sanity_checks(raw: pd.DataFrame, fixed: pd.DataFrame, args: argparse.Namespace) -> dict:
    checks: dict[str, object] = {}
    oracle = {}
    for b in args.b_values:
        setting = f"b={b:g}"
        oracle[setting] = {}
        for method, ppci in [("ORACLE_WSTAR_LO", False), ("ORACLE_WSTAR_PPCI", True)]:
            block = raw[(raw["setting"] == setting) & (raw["method"] == method)]
            empirical_var = float(block["estimate"].var(ddof=1))
            analytic_var = analytic_oracle_variance(args, b, ppci)
            relative_error = abs(empirical_var / analytic_var - 1.0)
            oracle[setting][method] = {
                "bias": float(block["error"].mean()),
                "empirical_variance": empirical_var,
                "analytic_variance": analytic_var,
                "relative_variance_error": relative_error,
            }
            if len(block) >= 100:
                assert relative_error < 0.35
                assert abs(block["error"].mean()) < 4.0 * math.sqrt(analytic_var / len(block))
    checks["oracle"] = oracle
    fixed_summary = fixed.groupby("setting").agg(empirical_bias=("error", "mean"), empirical_sd=("error", "std"), repetitions=("error", "size")).reset_index()
    fixed_summary["analytic_bias"] = args.curvature * args.fixed_h_check**2 / 3.0
    fixed_summary["absolute_bias_error"] = (fixed_summary["empirical_bias"] - fixed_summary["analytic_bias"]).abs()
    if len(fixed) >= 200:
        tolerance = 4.0 * fixed_summary["empirical_sd"] / np.sqrt(fixed_summary["repetitions"]) + 0.01
        assert (fixed_summary["absolute_bias_error"] <= tolerance).all()
    checks["fixed_h_uniform_nw"] = fixed_summary.to_dict("records")
    checks["all_finite_core_outputs"] = bool(np.isfinite(raw[["estimate", "estimated_se", "ci_lower", "ci_upper"]]).all().all())
    checks["target_is_beta"] = bool(np.allclose(raw["theta0"], args.beta))
    checks["localization_has_signed_weights"] = bool((raw.loc[raw["method"].str.startswith("RKHS_LOCALIZATION"), "negative_weight_fraction"] > 0).any())
    assert checks["all_finite_core_outputs"] and checks["target_is_beta"]
    return checks


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def make_figures(raw: pd.DataFrame, summary: pd.DataFrame, gains: pd.DataFrame, output: Path) -> None:
    main = summary[summary["method"].isin(MAIN_METHODS)].copy()
    method_order = MAIN_METHODS
    short = {
        "NW_LO_CV": "NW LO CV", "NW_PPCI_CV": "NW PPCI CV", "NW_LO_US": "NW LO US", "NW_PPCI_US": "NW PPCI US",
        "RKHS_LOCALIZATION_LO_GH_P1": "RKHS localization LO", "RKHS_LOCALIZATION_PPCI_GH_P1": "RKHS localization PPCI", "ORACLE_WSTAR_LO": "Oracle LO", "ORACLE_WSTAR_PPCI": "Oracle PPCI",
    }
    colors = {"b=0": "#2f7d4a", "b=2": "#1f4e79"}

    def paired_bars(metric: str, ylabel: str, stem: str) -> None:
        fig, ax = plt.subplots(figsize=(11.0, 4.4), constrained_layout=True)
        x = np.arange(len(method_order))
        width = 0.36
        for offset, setting in [(-width / 2, "b=0"), (width / 2, "b=2")]:
            block = main[main["setting"] == setting].set_index("method").reindex(method_order)
            ax.bar(x + offset, block[metric].to_numpy(), width, label=setting, color=colors[setting])
        ax.set_xticks(x, [short[item] for item in method_order], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.legend(frameon=False)
        save_figure(fig, output, stem)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["coverage", "mean_width"], ["Coverage", "Average CI width"]):
        x = np.arange(len(method_order)); width = 0.36
        for offset, setting in [(-width / 2, "b=0"), (width / 2, "b=2")]:
            block = main[main["setting"] == setting].set_index("method").reindex(method_order)
            ax.bar(x + offset, block[metric].to_numpy(), width, label=setting, color=colors[setting])
        ax.set_xticks(x, [short[item] for item in method_order], rotation=35, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(axis="y", color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0.95, color="#555555", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    save_figure(fig, output, "coverage_width")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["bias", "rmse"], ["Bias", "RMSE"]):
        x = np.arange(len(method_order)); width = 0.36
        for offset, setting in [(-width / 2, "b=0"), (width / 2, "b=2")]:
            block = main[main["setting"] == setting].set_index("method").reindex(method_order)
            ax.bar(x + offset, block[metric].to_numpy(), width, label=setting, color=colors[setting])
        ax.set_xticks(x, [short[item] for item in method_order], rotation=35, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(axis="y", color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0.0, color="#555555", linewidth=0.8); axes[0].legend(frameon=False)
    save_figure(fig, output, "bias_rmse")

    if not gains.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
        pair_order = ["NW_CV", "NW_US", "RKHS_LOCALIZATION_GH_P1", "ORACLE_WSTAR"]
        x = np.arange(len(pair_order)); width = 0.36
        for offset, setting in [(-width / 2, "b=0"), (width / 2, "b=2")]:
            block = gains[gains["setting"] == setting].set_index("pair").reindex(pair_order)
            ax.bar(x + offset, block["relative_variance_reduction"].to_numpy(), width, label=setting, color=colors[setting])
        ax.set_xticks(x, ["NW CV", "NW US", "RKHS localization GH/P1", "Oracle w*"])
        ax.set_ylabel("Empirical relative variance reduction")
        ax.axhline(0.0, color="#555555", linewidth=0.8); ax.grid(axis="y", color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
        save_figure(fig, output, "paired_variance_reduction")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True, sharey=True)
    for ax, setting in zip(axes, ["b=0", "b=2"]):
        for method, label, color in [("NW_LO_CV", "NW-CV", "#1f4e79"), ("NW_LO_US", "NW-US", "#c43d3d"), ("RKHS_LOCALIZATION_LO_GH_P1", "RKHS localization GH/P1", "#2f7d4a")]:
            values = raw[(raw["setting"] == setting) & (raw["method"] == method)]["selected_h"].dropna()
            ax.hist(values.to_numpy(), bins=18, alpha=0.55, label=label, color=color)
        ax.set_title(setting); ax.set_xlabel("Selected h"); ax.grid(axis="y", color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Count"); axes[0].legend(frameon=False)
    save_figure(fig, output, "selected_h_distribution")

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    block = raw[raw["method"] == "RKHS_LOCALIZATION_PPCI_GH_P1"]
    for setting, group in block.groupby("setting"):
        ax.scatter(group["selected_h"].to_numpy(), group["negative_weight_fraction"].to_numpy(), s=16, alpha=0.55, label=setting, color=colors[setting])
    ax.set(xlabel="Selected RKHS localization h", ylabel="Negative-weight fraction", title="Signed RKHS localization weights")
    ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    save_figure(fig, output, "localization_negative_weights")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    nw = main[main["method"].isin(["NW_PPCI_CV", "NW_PPCI_US"])]
    for ax, metric, title in zip(axes[:2], ["bias", "coverage"], ["Bias", "Coverage"]):
        for setting, group in nw.groupby("setting"):
            ax.plot(group["selected_h_mean"].to_numpy(), group[metric].to_numpy(), "o-", label=setting, color=colors[setting])
        ax.set(xlabel="Mean selected h", ylabel=title); ax.grid(color="#dddddd", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
    if not gains.empty:
        for setting, group in gains[gains["pair"].isin(["NW_CV", "NW_US"])].groupby("setting"):
            h_lookup = nw[nw["setting"] == setting].set_index("method")["selected_h_mean"]
            h_values = [h_lookup["NW_PPCI_CV" if pair == "NW_CV" else "NW_PPCI_US"] for pair in group["pair"]]
            axes[2].plot(h_values, group["relative_variance_reduction"].to_numpy(), "o-", label=setting, color=colors[setting])
    axes[2].set(xlabel="Mean selected h", ylabel="Relative variance reduction")
    axes[2].grid(color="#dddddd", linewidth=0.7); axes[2].spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0.95, color="#555555", linestyle="--", linewidth=1); axes[0].legend(frameon=False)
    save_figure(fig, output, "nw_bias_coverage_gain_tradeoff")


def write_readme(summary: pd.DataFrame, gains: pd.DataFrame, checks: dict, args: argparse.Namespace, output: Path) -> None:
    main = summary[summary["method"].isin(MAIN_METHODS)][["setting", "method", "coverage", "bias", "rmse", "mean_width"]]
    table = main.to_markdown(index=False, floatfmt=".4f")
    gain_table = gains.to_markdown(index=False, floatfmt=".4f")
    text = f"""# NW versus Kernel RKHS localization PPCI

## Design

`X ~ Unif[-1,1]` and `Y = beta + b X + A X^2 + epsilon`, with `beta={args.beta:g}`, `A={args.curvature:g}`, `sigma={args.sigma:g}`, `n={args.n}`, `N={args.N}`, and `b in {args.b_values}`. The auxiliary predictor is the exact conditional mean, and every confidence interval targets `theta(0)=beta`.

Each setting has {args.reps} paired Monte Carlo replicates. Within a replicate all methods share the same labelled and unlabelled samples. NW-CV uses an independent pilot sample of size {args.n_pilot} and leave-one-out regression CV over a predeclared {args.nw_h_grid_size}-point bandwidth grid. NW-US uses `h_CV * n^(-{args.undersmooth_delta:g})`, clipped only to the predeclared grid range.

The actual RKHS localization methods call the project implementation with median-based `G_h={args.localization_h_factors}`, shrinking `G_lambda`, stability thresholds `{args.tau_op:g}/{args.tau_loc:g}`, the P1 labelled-scale screen, and least-normalized-violation fallback. The main table fixes `c_bias={args.main_c_bias:g}`; `{args.c_bias_values}` are reported in `cbias_sensitivity.csv`.

`ORACLE_WSTAR_*` uses `w*(x)=9/4-15x^2/4`. It is a finite-dimensional oracle signed-balance reference, not a weight that general kernel RKHS localization PPCI is required to recover.

## Main Results

{table}

## Empirical Relative Variance Reduction

{gain_table}

## Checks and Interpretation

- Core estimates and standard errors are finite: `{checks['all_finite_core_outputs']}`.
- All intervals target beta: `{checks['target_is_beta']}`.
- Actual RKHS localization weights exhibit signed weights: `{checks['localization_has_signed_weights']}`.
- Oracle empirical variance is compared with its analytic formula in `sanity_checks.json`.
- Fixed-bandwidth uniform NW is recorded separately in `fixed_h_sanity.csv` and checked against the analytic bias `A h^2 / 3`.

The results are descriptive Monte Carlo evidence. No tuning constant was selected after inspecting coverage, and the two non-main `c_bias` values are sensitivity analyses only.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/nw_vs_localization_quadratic_v1")
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backend", default="cpu", choices=["auto", "cpu", "torch", "gpu", "cuda"])
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--curvature", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--b-values", default="0,2")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--n-pilot", type=int, default=200)
    parser.add_argument("--kernel", default="matern52")
    parser.add_argument("--nw-h-min", type=float, default=0.05)
    parser.add_argument("--nw-h-max", type=float, default=0.8)
    parser.add_argument("--nw-h-grid-size", type=int, default=35)
    parser.add_argument("--undersmooth-delta", type=float, default=0.05)
    parser.add_argument("--localization-h-factors", default="0.8,1.0,1.15,1.2")
    parser.add_argument("--lambda-factor-min", type=float, default=0.05)
    parser.add_argument("--lambda-factor-max", type=float, default=20.0)
    parser.add_argument("--lambda-grid-size", type=int, default=41)
    parser.add_argument("--tau-op", type=float, default=12.0)
    parser.add_argument("--tau-loc", type=float, default=4.0)
    parser.add_argument("--main-c-bias", type=float, default=0.18)
    parser.add_argument("--c-bias-values", default="0.12,0.18,0.25")
    parser.add_argument("--fixed-h-check", type=float, default=0.3)
    parser.add_argument("--z-alpha", type=float, default=1.959963984540054)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    args.b_values = parse_floats(args.b_values)
    args.localization_h_factors = parse_floats(args.localization_h_factors)
    args.c_bias_values = parse_floats(args.c_bias_values)
    if args.main_c_bias not in args.c_bias_values:
        raise ValueError("--main-c-bias must be included in --c-bias-values")
    if args.smoke:
        args.reps = 10
    if args.benchmark:
        args.reps = 1
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
        blocks = []
        for replicate in range(args.reps):
            blocks.append(run_replicate(replicate, args))
            print(f"[done] replicate={replicate + 1}/{args.reps}", flush=True)
    elapsed = time.perf_counter() - start

    rows = [row for block, _ in blocks for row in block]
    fixed_rows = [row for _, fixed_block in blocks for row in fixed_block]
    raw = pd.DataFrame(rows)
    fixed = pd.DataFrame(fixed_rows)
    summary = summarize(raw)
    gains = paired_gain_summary(raw)
    selection = selection_summary(raw)
    sensitivity = summary[summary["method"].str.startswith("RKHS_LOCALIZATION")].copy()
    checks = sanity_checks(raw, fixed, args)
    checks["elapsed_seconds"] = elapsed
    checks["backend_resolved"] = args.backend_resolved
    checks["workers"] = args.workers

    raw.to_csv(output / "raw_replicates.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    gains.to_csv(output / "paired_gain_summary.csv", index=False)
    selection.to_csv(output / "selection_summary.csv", index=False)
    sensitivity.to_csv(output / "cbias_sensitivity.csv", index=False)
    fixed.to_csv(output / "fixed_h_sanity.csv", index=False)
    (output / "sanity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    config = vars(args).copy()
    config["output_dir"] = str(config["output_dir"])
    config["elapsed_seconds"] = elapsed
    config["source_sha256"], config["source_file_count"] = source_sha256()
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_figures(raw, summary, gains, output)
    write_readme(summary, gains, checks, args, output)
    print(summary[summary["method"].isin(MAIN_METHODS)].to_string(index=False))
    print(f"Saved Monte Carlo experiment to {output} in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
