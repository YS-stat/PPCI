#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ppci_condmean.data import prepare_blogfeedback_ppci
from ppci_condmean.diagnostics import average_nw_closeness
from ppci_condmean.estimator import ppci_mean_from_weight_values, lo_mean_from_weights, ppi_global_mean
from ppci_condmean.gpu import configure_backend
from ppci_condmean.joint_tuning import JointTuningConfig, collect_joint_candidate_cache, select_joint_from_cache, weight_from_joint_cache
from ppci_condmean.utils import ensure_dir, write_run_manifest
from ppci_condmean.weights import RKHSLocalizationWeight


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def parse_methods(s: str) -> list[str]:
    return [x.strip().upper() for x in str(s).split(",") if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(float(x)) for x in str(s).split(",") if str(x).strip()]


def make_cfg(args, *, pc_r: float | None = None, mb_gamma: float | None = None, gh_gamma: float | None = None) -> JointTuningConfig:
    return JointTuningConfig(
        h_grid_mode=args.h_grid_mode,
        h_factors=tuple(args.h_factors),
        k_min_floor=args.k_min_floor,
        k_max_frac=args.k_max_frac,
        k_growth=args.k_growth,
        lambda_factor_min=args.lambda_factor_min,
        lambda_factor_max=args.lambda_factor_max,
        lambda_grid_size=args.lambda_grid_size,
        lambda_grid_mode=args.lambda_grid_mode,
        tau_op=args.tau_op,
        tau_loc=args.tau_loc,
        pc_r=float(args.pc_rs[0] if pc_r is None else pc_r),
        mb_gamma=float(args.mb_gammas[0] if mb_gamma is None else mb_gamma),
        gh_gamma=float(args.gh_gammas[0] if gh_gamma is None else gh_gamma),
        bias_screen=getattr(args, "bias_screen_current", args.bias_screens[0]),
        c_bias=float(getattr(args, "c_bias_current", args.c_biases[0])),
        gh_adaptive=args.gh_adaptive,
        gh_adaptive_rule=args.gh_adaptive_rule,
        gh_c0=args.gh_c0,
        gh_gamma0=args.gh_gamma0,
        gh_rho=args.gh_rho,
        gh_ref_ratio=args.gh_ref_ratio,
        gh_edge_rho=args.gh_edge_rho,
        gh_edge_ridge=args.gh_edge_ridge,
        gh_a_tau=args.gh_a_tau,
        gh_pc_r=args.gh_pc_r,
        constraint_fallback=args.constraint_fallback,
        min_abs_j=args.min_abs_j,
        kernel=args.kernel,
        backend=args.backend_resolved,
    )


def method_configs(args) -> tuple[list[tuple[str, str, JointTuningConfig]], JointTuningConfig]:
    methods = set(args.methods)
    base_cfg = make_cfg(args)
    out: list[tuple[str, str, JointTuningConfig]] = []
    if "INC" in methods:
        out.append(("INC", "INC", base_cfg))
    if "PC" in methods:
        for r in args.pc_rs:
            out.append((f"PC_r{r:g}", "PC", make_cfg(args, pc_r=r)))
    if "MB" in methods:
        for g in args.mb_gammas:
            out.append((f"MB_g{g:g}", "MB", make_cfg(args, mb_gamma=g)))
    if "GH" in methods:
        for screen in args.bias_screens:
            if str(screen).lower() == "legacy":
                gh_budget_values = args.gh_gammas
                c_bias_values = [args.c_biases[0]]
            else:
                gh_budget_values = [args.gh_gammas[0]]
                c_bias_values = args.c_biases
            for g in gh_budget_values:
                for c_bias in c_bias_values:
                    args.bias_screen_current = screen
                    args.c_bias_current = c_bias
                    if str(screen).lower() == "legacy":
                        label = f"GH_log{args.gh_c0:g}" if args.gh_adaptive and args.gh_adaptive_rule == "log_ratio" else f"GH_g{g:g}"
                    else:
                        screen_label = {"p1_label": "P1", "p2_log": "P2", "p3_full": "P3"}.get(str(screen).lower(), str(screen))
                        label = f"GH_{screen_label}_c{c_bias:g}"
                    out.append((label, "GH", make_cfg(args, gh_gamma=g)))
    return out, base_cfg


def build_twofold_joint_cached(
    X_u: np.ndarray,
    x0: np.ndarray,
    n_label: int,
    seed: int,
    methods_to_run: list[tuple[str, str, JointTuningConfig]],
    *,
    base_cfg: JointTuningConfig,
) -> dict:
    rng = np.random.default_rng(seed)
    N = int(X_u.shape[0])
    perm = rng.permutation(N)
    I1 = np.sort(perm[: N // 2])
    I2 = np.sort(perm[N // 2 :])

    cache1 = collect_joint_candidate_cache(X_u[I1], x0, n=n_label, cfg=base_cfg)
    cache2 = collect_joint_candidate_cache(X_u[I2], x0, n=n_label, cfg=base_cfg)
    tuned = {}
    weight_cache = {}

    def get_weight(fold: str, cache: dict, tr, cfg: JointTuningConfig):
        key = (fold, float(tr.h), float(tr.lam), cfg.kernel, cfg.backend)
        if key not in weight_cache:
            weight_cache[key] = weight_from_joint_cache(cache, tr, cfg)
        return weight_cache[key]

    for label, family, cfg in methods_to_run:
        tr1 = select_joint_from_cache(cache1, family, cfg=cfg)
        tr2 = select_joint_from_cache(cache2, family, cfg=cfg)
        w1 = get_weight("fold1", cache1, tr1, cfg)
        w2 = get_weight("fold2", cache2, tr2, cfg)
        w_u = np.zeros(N, dtype=float)
        w_u[I1] = w2(X_u[I1])
        w_u[I2] = w1(X_u[I2])
        tuned[label] = (w1, w2, w_u, tr1, tr2)
    return tuned


def attach_tuning(res, tr1, tr2, label: str):
    res.method = label
    res.h_1, res.h_2 = tr1.h, tr2.h
    res.lambda_1, res.lambda_2 = tr1.lam, tr2.lam
    res.h_mean = 0.5 * (tr1.h + tr2.h)
    res.lambda_mean = 0.5 * (tr1.lam + tr2.lam)
    res.h = res.h_mean
    res.lambda_value = res.lambda_mean
    res.tuning_status = f"fold1:{tr1.status};fold2:{tr2.status}"
    res.h_factor_1, res.h_factor_2 = tr1.h_factor_vs_median, tr2.h_factor_vs_median
    res.lambda_factor_1, res.lambda_factor_2 = tr1.lambda_factor, tr2.lambda_factor
    res.ess0_1, res.ess0_2 = tr1.ess0, tr2.ess0
    res.op_score_1, res.op_score_2 = tr1.op_score, tr2.op_score
    res.loc_score_1, res.loc_score_2 = tr1.loc_score, tr2.loc_score
    res.h_factor = 0.5 * (tr1.h_factor_vs_median + tr2.h_factor_vs_median)
    res.lambda_factor = 0.5 * (tr1.lambda_factor + tr2.lambda_factor)
    res.op_score = 0.5 * (tr1.op_score + tr2.op_score)
    res.loc_score = 0.5 * (tr1.loc_score + tr2.loc_score)
    res.h_mode = tr1.h_grid_mode
    res.lambda_selection = label
    return res


def row_from_result(res, theta0: float, context: dict, tr1=None, tr2=None) -> dict:
    d = res.as_dict()
    d["lambda"] = d.get("lambda_value", d.get("lambda_mean", np.nan))
    d["width"] = float(d["ci_high"] - d["ci_low"])
    d["covered"] = bool((d["ci_low"] <= theta0) and (theta0 <= d["ci_high"]))
    d["error"] = float(d["theta_hat"] - theta0)
    d["theta0"] = float(theta0)
    d.update(context)
    if tr1 is not None and tr2 is not None:
        for prefix, tr in [("tune1", tr1), ("tune2", tr2)]:
            td = tr.as_dict()
            keep = [
                "status", "k_target", "lambda_factor", "eff_dim", "point_leverage", "stable",
                "n_stable_this_h", "n_feasible_total", "h_grid_mode", "J_w", "V_w", "Q_h",
                "D_h_point", "D_h_op", "sw_proxy", "power", "power_ref", "pc_ratio",
                "m1_norm", "M2_nuc", "B_geo", "R_MB", "R_GH", "bias_screen", "c_bias",
                "bias_score", "bias_score_label", "bias_score_full", "bias_budget",
                "gh_gamma_used", "gh_gamma_eff", "edge_score", "A_score", "M2_lambda_min",
                "h_factor_vs_median",
            ]
            for key in keep:
                d[f"{prefix}_{key}"] = td.get(key, np.nan)
        d["k_target_mean"] = 0.5 * (tr1.k_target + tr2.k_target)
        d["sw_proxy_mean"] = 0.5 * (tr1.sw_proxy + tr2.sw_proxy)
        d["R_MB_mean"] = 0.5 * (tr1.R_MB + tr2.R_MB)
        d["R_GH_mean"] = 0.5 * (tr1.R_GH + tr2.R_GH)
        d["bias_score_mean"] = 0.5 * (tr1.bias_score + tr2.bias_score)
        d["bias_budget_mean"] = 0.5 * (tr1.bias_budget + tr2.bias_budget)
        d["gh_gamma_eff_mean"] = 0.5 * (tr1.gh_gamma_eff + tr2.gh_gamma_eff)
        d["edge_score_mean"] = 0.5 * (tr1.edge_score + tr2.edge_score)
        d["pc_ratio_mean"] = 0.5 * (tr1.pc_ratio + tr2.pc_ratio)
        d["J_w_mean"] = 0.5 * (tr1.J_w + tr2.J_w)
    return d


def cluster_coverage_stats(g: pd.DataFrame) -> tuple[int, float, float]:
    if "unlab_rep" not in g:
        return 0, np.nan, np.nan
    cluster_cov = g.groupby("unlab_rep", dropna=False)["covered"].mean().astype(float)
    n_clusters = int(len(cluster_cov))
    if n_clusters <= 1:
        return n_clusters, np.nan, np.nan
    sd = float(cluster_cov.std(ddof=1))
    se = float(sd / np.sqrt(n_clusters))
    return n_clusters, se, sd


def summarize_replicates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols + ["method"], dropna=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        row = {col: val for col, val in zip(group_cols + ["method"], key_vals)}
        err = g["error"].to_numpy(float)
        tuned = g["tuning_status"].astype(str) if "tuning_status" in g else pd.Series([], dtype=str)
        has_tuning = bool(len(tuned) and tuned.str.len().gt(0).any())
        n_clusters, cov_cluster_se, cov_cluster_sd = cluster_coverage_stats(g)
        reps_per_cluster = len(g) / n_clusters if n_clusters > 0 else 1.0
        tuning_seconds = float(g["tuning_seconds"].mean()) if "tuning_seconds" in g else np.nan
        estimation_seconds = float(g["estimation_seconds"].mean()) if "estimation_seconds" in g else np.nan
        row.update({
            "n_reps": int(len(g)),
            "n_unlab_clusters": n_clusters,
            "coverage": float(g["covered"].mean()),
            "coverage_cluster_se": cov_cluster_se,
            "coverage_cluster_sd": cov_cluster_sd,
            "bias": float(np.mean(err)),
            "abs_bias": float(abs(np.mean(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "se_mean": float(g["se"].mean()),
            "width": float(g["width"].mean()),
            "emp_sd": float(np.std(g["theta_hat"].to_numpy(float), ddof=1)) if len(g) > 1 else 0.0,
            "theta0_mean": float(g["theta0"].mean()),
            "theta_hat_mean": float(g["theta_hat"].mean()),
            "lambda_factor_mean": float(g["lambda_factor"].mean()) if "lambda_factor" in g else np.nan,
            "h_factor_mean": float(g["h_factor"].mean()) if "h_factor" in g else np.nan,
            "R_MB_mean": float(g["R_MB_mean"].mean()) if "R_MB_mean" in g else np.nan,
            "R_GH_mean": float(g["R_GH_mean"].mean()) if "R_GH_mean" in g else np.nan,
            "bias_score_mean": float(g["bias_score_mean"].mean()) if "bias_score_mean" in g else np.nan,
            "bias_budget_mean": float(g["bias_budget_mean"].mean()) if "bias_budget_mean" in g else np.nan,
            "fallback_rate": float(tuned.str.contains("fallback", regex=False).mean()) if has_tuning else np.nan,
            "sigma2_Y_mean": float(g["sigma2_Y"].mean()) if "sigma2_Y" in g and g["sigma2_Y"].notna().any() else np.nan,
            "sigma2_Y_minus_f_mean": float(g["sigma2_Y_minus_f"].mean()) if "sigma2_Y_minus_f" in g and g["sigma2_Y_minus_f"].notna().any() else np.nan,
            "sigma2_f_mean": float(g["sigma2_f"].mean()) if "sigma2_f" in g and g["sigma2_f"].notna().any() else np.nan,
            "tuning_seconds_mean_per_unlab_draw": tuning_seconds,
            "estimation_seconds_mean": estimation_seconds,
            "amortized_procedure_seconds_mean": tuning_seconds / reps_per_cluster + estimation_seconds,
            "nw_corr_mean": float(g["nw_corr"].mean()) if "nw_corr" in g and g["nw_corr"].notna().any() else np.nan,
            "nw_relative_difference_mean": float(g["nw_relative_difference"].mean()) if "nw_relative_difference" in g and g["nw_relative_difference"].notna().any() else np.nan,
            "negative_weight_fraction_mean": float(g["negative_weight_fraction"].mean()) if "negative_weight_fraction" in g and g["negative_weight_fraction"].notna().any() else np.nan,
            "M_lambda_over_eigmax_mean": float(g["M_lambda_over_eigmax"].mean()) if "M_lambda_over_eigmax" in g and g["M_lambda_over_eigmax"].notna().any() else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def add_ratios(summary: pd.DataFrame, group_cols: list[str], baseline: str) -> pd.DataFrame:
    out = []
    for _, gs in summary.groupby(group_cols, dropna=False):
        gs = gs.copy()
        base = gs[gs["method"] == baseline]
        if not base.empty:
            w0 = float(base["width"].iloc[0])
            r0 = float(base["rmse"].iloc[0])
            gs[f"width_ratio_vs_{baseline}"] = gs["width"] / w0 if w0 > 0 else np.nan
            gs[f"rmse_ratio_vs_{baseline}"] = gs["rmse"] / r0 if r0 > 0 else np.nan
        out.append(gs)
    return pd.concat(out, ignore_index=True) if out else summary


def aggregate_from_x0(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ratio_cols = [c for c in summary.columns if c.startswith("width_ratio_vs_") or c.startswith("rmse_ratio_vs_")]
    for method, g in summary.groupby("method", dropna=False):
        row = {
            "method": method,
            "n_x0": int(len(g)),
            "coverage_mean": float(g["coverage"].mean()),
            "coverage_min_x0": float(g["coverage"].min()),
            "coverage_max_x0": float(g["coverage"].max()),
            "coverage_cluster_se_mean": float(g["coverage_cluster_se"].mean()) if "coverage_cluster_se" in g and g["coverage_cluster_se"].notna().any() else np.nan,
            "coverage_cluster_se_max": float(g["coverage_cluster_se"].max()) if "coverage_cluster_se" in g and g["coverage_cluster_se"].notna().any() else np.nan,
            "bias_mean": float(g["bias"].mean()),
            "abs_bias_mean": float(g["abs_bias"].mean()),
            "rmse_mean": float(g["rmse"].mean()),
            "width_mean": float(g["width"].mean()),
            "se_mean": float(g["se_mean"].mean()),
            "emp_sd_mean": float(g["emp_sd"].mean()),
            "emp_sd_over_se_mean": float(g["emp_sd"].mean() / g["se_mean"].mean()) if float(g["se_mean"].mean()) > 0 else np.nan,
            "fallback_rate_mean": float(g["fallback_rate"].mean()) if g["fallback_rate"].notna().any() else np.nan,
            "lambda_factor_mean": float(g["lambda_factor_mean"].mean()) if g["lambda_factor_mean"].notna().any() else np.nan,
            "h_factor_mean": float(g["h_factor_mean"].mean()) if g["h_factor_mean"].notna().any() else np.nan,
            "bias_score_mean": float(g["bias_score_mean"].mean()) if "bias_score_mean" in g and g["bias_score_mean"].notna().any() else np.nan,
            "bias_budget_mean": float(g["bias_budget_mean"].mean()) if "bias_budget_mean" in g and g["bias_budget_mean"].notna().any() else np.nan,
            "sigma2_Y_mean": float(g["sigma2_Y_mean"].mean()) if "sigma2_Y_mean" in g and g["sigma2_Y_mean"].notna().any() else np.nan,
            "sigma2_Y_minus_f_mean": float(g["sigma2_Y_minus_f_mean"].mean()) if "sigma2_Y_minus_f_mean" in g and g["sigma2_Y_minus_f_mean"].notna().any() else np.nan,
            "sigma2_f_mean": float(g["sigma2_f_mean"].mean()) if "sigma2_f_mean" in g and g["sigma2_f_mean"].notna().any() else np.nan,
            "tuning_seconds_mean_per_unlab_draw": float(g["tuning_seconds_mean_per_unlab_draw"].mean()) if "tuning_seconds_mean_per_unlab_draw" in g else np.nan,
            "estimation_seconds_mean": float(g["estimation_seconds_mean"].mean()) if "estimation_seconds_mean" in g else np.nan,
            "amortized_procedure_seconds_mean": float(g["amortized_procedure_seconds_mean"].mean()) if "amortized_procedure_seconds_mean" in g else np.nan,
            "nw_corr_mean": float(g["nw_corr_mean"].mean()) if "nw_corr_mean" in g and g["nw_corr_mean"].notna().any() else np.nan,
            "nw_relative_difference_mean": float(g["nw_relative_difference_mean"].mean()) if "nw_relative_difference_mean" in g and g["nw_relative_difference_mean"].notna().any() else np.nan,
            "negative_weight_fraction_mean": float(g["negative_weight_fraction_mean"].mean()) if "negative_weight_fraction_mean" in g and g["negative_weight_fraction_mean"].notna().any() else np.nan,
            "M_lambda_over_eigmax_mean": float(g["M_lambda_over_eigmax_mean"].mean()) if "M_lambda_over_eigmax_mean" in g and g["M_lambda_over_eigmax_mean"].notna().any() else np.nan,
        }
        for col in ratio_cols:
            row[f"{col}_mean"] = float(g[col].mean()) if g[col].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method")


def compact_tuning_decisions(df: pd.DataFrame) -> pd.DataFrame:
    if "tuning_status" not in df:
        return pd.DataFrame()
    tuned = df[df["tuning_status"].astype(str).str.len() > 0].copy()
    if tuned.empty:
        return pd.DataFrame()
    keys = ["x0_index", "method", "unlab_rep"]
    cols = keys + [
        "tuning_status", "lambda_factor", "h_factor", "R_MB_mean", "R_GH_mean",
        "bias_score_mean", "bias_budget_mean", "pc_ratio_mean", "J_w_mean",
        "sw_proxy_mean", "op_score", "loc_score", "tune1_status", "tune2_status",
    ]
    keep = [c for c in cols if c in tuned.columns]
    out = tuned.drop_duplicates(keys)[keep].copy()
    out["fallback"] = out["tuning_status"].astype(str).str.contains("fallback", regex=False)
    return out


def summarize_tuning_decisions(tune: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if tune.empty:
        return pd.DataFrame()
    rows = []
    value_cols = [
        "lambda_factor", "h_factor", "R_MB_mean", "R_GH_mean", "bias_score_mean",
        "bias_budget_mean", "pc_ratio_mean", "J_w_mean", "sw_proxy_mean", "op_score", "loc_score",
    ]
    for key, g in tune.groupby(group_cols + ["method"], dropna=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        row = {col: val for col, val in zip(group_cols + ["method"], key_vals)}
        row.update({"tuning_decisions": int(len(g)), "fallback_rate": float(g["fallback"].mean())})
        for col in value_cols:
            if col in g:
                row[f"{col}_mean"] = float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_one_x0(x0_index: int, x0: np.ndarray, theta0: float, data_tuple: tuple, args) -> list[dict]:
    Xp, Yp, fp = data_tuple
    methods_to_run, base_cfg = method_configs(args)
    all_idx = np.arange(len(Yp))
    exact = np.all(Xp == np.asarray(x0).reshape(1, -1), axis=1)
    pool = all_idx[~exact]
    need = int(args.n_label + args.n_unlab)
    if len(pool) < need:
        raise ValueError(f"Not enough BlogFeedback PPCI rows for x0={x0_index}: pool={len(pool)}, need={need}")

    rows = []
    for urep in range(args.unlab_reps):
        urep_global = int(args.unlab_rep_offset + urep)
        seed_u = int(args.seed + 100000 * (x0_index + 1) + 10000 * urep_global)
        rng_u = np.random.default_rng(seed_u)
        un_idx = rng_u.choice(pool, size=args.n_unlab, replace=False)
        label_pool = np.setdiff1d(pool, un_idx, assume_unique=False)
        if len(label_pool) < args.n_label:
            raise ValueError(f"Not enough BlogFeedback PPCI rows after unlabeled draw for x0={x0_index}: label_pool={len(label_pool)}, n_label={args.n_label}")
        X_u, f_u = Xp[un_idx], fp[un_idx]

        tuning_start = time.perf_counter()
        tuned = build_twofold_joint_cached(
            X_u,
            x0,
            args.n_label,
            seed_u + 17,
            methods_to_run,
            base_cfg=base_cfg,
        )
        tuning_seconds = float(time.perf_counter() - tuning_start)
        nw_diagnostics = {
            label: average_nw_closeness(packed[0], packed[1])
            for label, packed in tuned.items()
        }
        for brep in range(args.label_reps):
            rep = int(urep * args.label_reps + brep)
            seed_l = int(args.seed + 100000 * (x0_index + 1) + 5000000 + 10000 * urep_global + brep)
            rng_l = np.random.default_rng(seed_l)
            lab_idx = rng_l.choice(label_pool, size=args.n_label, replace=False)
            X_l, Y_l, f_l = Xp[lab_idx], Yp[lab_idx], fp[lab_idx]
            ctx = {
                "experiment": "blogfeedback_joint",
                "x0_index": int(x0_index),
                "rep": int(rep),
                "unlab_rep": int(urep_global),
                "label_rep": int(brep),
                "n_label": int(args.n_label),
                "n_unlab": int(args.n_unlab),
                "tuning_seconds": tuning_seconds,
            }
            for label, packed in tuned.items():
                estimation_start = time.perf_counter()
                w1, w2, w_u, tr1, tr2 = packed
                w_l = 0.5 * (w1(X_l) + w2(X_l))
                res = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, method=label)
                attach_tuning(res, tr1, tr2, label)
                diagnostic_ctx = dict(ctx, **nw_diagnostics[label], estimation_seconds=float(time.perf_counter() - estimation_start))
                rows.append(row_from_result(res, theta0, diagnostic_ctx, tr1, tr2))

            if args.include_lo_ppi:
                for source_label in lo_source_labels(tuned, args):
                    estimation_start = time.perf_counter()
                    w1, w2, _, tr1, tr2 = tuned[source_label]
                    w_l = 0.5 * (w1(X_l) + w2(X_l))
                    lo = lo_mean_from_weights(X_l, Y_l, w_l)
                    lo.method = lo_method_label(source_label)
                    diagnostic_ctx = dict(ctx, **nw_diagnostics[source_label], estimation_seconds=float(time.perf_counter() - estimation_start))
                    rows.append(row_from_result(lo, theta0, diagnostic_ctx, tr1, tr2))
                estimation_start = time.perf_counter()
                ppi = ppi_global_mean(Y_l, f_l, f_u)
                ppi.method = "PPI_global"
                rows.append(row_from_result(ppi, theta0, dict(ctx, tuning_seconds=0.0, estimation_seconds=float(time.perf_counter() - estimation_start))))
    print(f"[done] x0_index={x0_index}, theta0={theta0:.4f}, unlab_reps={args.unlab_reps}, label_reps={args.label_reps}", flush=True)
    return rows


def fmt_value(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4g}"
    return str(x)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "(empty)\n"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt_value(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def first_gh_label(args) -> str:
    if not args.bias_screens:
        return "GH"
    screen = str(args.bias_screens[0]).lower()
    if screen == "legacy":
        return f"GH_log{args.gh_c0:g}" if args.gh_adaptive and args.gh_adaptive_rule == "log_ratio" else f"GH_g{args.gh_gammas[0]:g}"
    screen_label = {"p1_label": "P1", "p2_log": "P2", "p3_full": "P3"}.get(screen, screen)
    return f"GH_{screen_label}_c{args.c_biases[0]:g}"


def lo_source_labels(tuned: dict, args) -> list[str]:
    if not tuned:
        return []
    mode = getattr(args, "lo_weight_method", "first")
    if mode == "inc":
        return ["INC"] if "INC" in tuned else []
    if mode == "all":
        return list(tuned.keys())
    for label in tuned:
        if label != "INC":
            return [label]
    return [next(iter(tuned))]


def lo_method_label(source_label: str) -> str:
    return "LO_INCw" if source_label == "INC" else f"LO_{source_label}w"


def write_report(out_dir: Path, args, aggregate: pd.DataFrame, by_x0: pd.DataFrame, tuning_method: pd.DataFrame):
    gh_label = first_gh_label(args)
    lines = [
        "# BlogFeedback Experiment Summary",
        "",
        "## Setup",
        "",
        f"- Data: `{args.data}`",
        f"- Predictor model: `{args.model}`, `max_train={args.max_train}`, `model_n_jobs={args.model_n_jobs}`",
        f"- Targets: `n_x0={args.n_x0}`",
        f"- Sampling: `n_label={args.n_label}`, `n_unlab={args.n_unlab}`, `unlab_reps={args.unlab_reps}`, `label_reps={args.label_reps}` (`reps={args.reps}` total) per x0",
        f"- Tuning constants: `h_grid_mode={args.h_grid_mode}`, `h_factors={args.h_factors}`, `tau_op={args.tau_op}`, `tau_loc={args.tau_loc}`",
        f"- A_bias: screens `{args.bias_screens}`, `c_biases={args.c_biases}`, fallback `{args.constraint_fallback}`",
        f"- Lambda grid: mode `{args.lambda_grid_mode}`, factors `[ {args.lambda_factor_min}, {args.lambda_factor_max} ]`, size `{args.lambda_grid_size}`",
        "- Bias/coverage use the complete held-out PPCI inference population to define the NW reference target on log(1+comments); predictor-training outcomes are disjoint.",
        "",
        "## Equal-x0 Aggregate",
        "",
        markdown_table(
            aggregate,
            [
                "method", "n_x0", "coverage_mean", "coverage_min_x0", "bias_mean",
                "coverage_cluster_se_mean", "coverage_cluster_se_max", "abs_bias_mean",
                "rmse_mean", "width_mean", "fallback_rate_mean",
                f"width_ratio_vs_{gh_label}_mean", f"rmse_ratio_vs_{gh_label}_mean",
            ],
        ),
        "## Tuning Diagnostics",
        "",
        markdown_table(
            tuning_method,
            [
                "method", "tuning_decisions", "fallback_rate", "lambda_factor_mean",
                "h_factor_mean", "R_GH_mean", "R_MB_mean", "bias_score_mean",
                "bias_budget_mean", "pc_ratio_mean",
            ],
        ),
    ]
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report}")


def main():
    ap = argparse.ArgumentParser(description="BlogFeedback experiment with P1 covariate-only tuning.")
    ap.add_argument("--data", default="data/blogfeedback/blogfeedback.zip")
    ap.add_argument("--output-dir", default="runs/blogfeedback")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--n-label", type=int, default=300)
    ap.add_argument("--n-unlab", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--unlab-reps", type=int, default=50, help="Number of independent unlabeled draws. If positive, overrides --reps together with --label-reps.")
    ap.add_argument("--unlab-rep-offset", type=int, default=0, help="Nonnegative offset for the unlabeled-replicate random stream; use it to validate fixed targets on fresh Monte Carlo draws.")
    ap.add_argument("--label-reps", type=int, default=20, help="Number of labeled draws per unlabeled draw. Defaults to --reps when --unlab-reps is not set.")
    ap.add_argument("--n-x0", type=int, default=50)
    ap.add_argument("--x0-indices", default="", help="Optional comma-separated subset of x0 indices to run.")
    ap.add_argument("--ppci-fraction", type=float, default=0.3)
    ap.add_argument("--max-train", type=int, default=0, help="Predictor-training cap; 0 uses the complete disjoint training split.")
    ap.add_argument("--max-raw-rows", type=int, default=0)
    ap.add_argument("--model", default="lightgbm", choices=["lightgbm", "lgbm", "extratrees", "ridge"])
    ap.add_argument("--model-n-jobs", type=int, default=1)
    ap.add_argument("--methods", default="GH")
    ap.add_argument("--pc-rs", default="1.02")
    ap.add_argument("--mb-gammas", default="0.25")
    ap.add_argument("--gh-gammas", default="0.30")
    ap.add_argument("--gh-adaptive", action="store_true", default=True)
    ap.add_argument("--gh-adaptive-rule", default="log_ratio", choices=["legacy", "log_ratio"])
    ap.add_argument("--gh-c0", type=float, default=0.18)
    ap.add_argument("--gh-gamma0", type=float, default=0.30)
    ap.add_argument("--gh-rho", type=float, default=0.15)
    ap.add_argument("--gh-ref-ratio", type=float, default=5.0)
    ap.add_argument("--gh-edge-rho", type=float, default=0.0)
    ap.add_argument("--gh-edge-ridge", type=float, default=1e-6)
    ap.add_argument("--gh-a-tau", type=float, default=float("inf"))
    ap.add_argument("--gh-pc-r", type=float, default=float("inf"))
    ap.add_argument("--constraint-fallback", default="least_violation", choices=["min_sw", "least_violation"])
    ap.add_argument("--h-grid-mode", default="median_grid", choices=["ess", "median_grid"])
    ap.add_argument("--h-factors", default="0.8,1.0,1.15,1.2")
    ap.add_argument("--bias-screens", default="p1_label", help="Comma-separated: p1_label,p2_log,p3_full,legacy")
    ap.add_argument("--c-biases", default="60", help="Comma-separated c_bias values for A_bias.")
    ap.add_argument("--k-min-floor", type=int, default=50)
    ap.add_argument("--k-max-frac", type=float, default=0.80)
    ap.add_argument("--k-growth", type=float, default=1.50)
    ap.add_argument("--lambda-factor-min", type=float, default=0.1)
    ap.add_argument("--lambda-factor-max", type=float, default=1000.0)
    ap.add_argument("--lambda-grid-size", type=int, default=41)
    ap.add_argument("--lambda-grid-mode", default="shrinking", choices=["n", "shrinking"])
    ap.add_argument("--tau-op", type=float, default=12.0)
    ap.add_argument("--tau-loc", type=float, default=4.0)
    ap.add_argument("--min-abs-j", type=float, default=1e-6)
    ap.add_argument("--kernel", default="matern52")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "torch", "gpu", "cuda"])
    ap.add_argument("--gpu-id", default="auto")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--include-lo-ppi", action="store_true", default=True)
    ap.add_argument("--no-lo-ppi", dest="include_lo_ppi", action="store_false")
    ap.add_argument("--lo-weight-method", default="all", choices=["first", "all", "inc"], help="Which tuned weights to use for LO when LO/PPI is included.")
    ap.add_argument("--save-replicates", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.unlab_rep_offset < 0:
        ap.error("--unlab-rep-offset must be nonnegative")

    if args.smoke:
        args.n_label = min(args.n_label, 60)
        args.n_unlab = min(args.n_unlab, 180)
        args.reps = min(args.reps, 2)
        args.unlab_reps = 1
        args.label_reps = min(args.label_reps if args.label_reps > 0 else args.reps, 2)
        args.n_x0 = min(args.n_x0, 2)
        args.lambda_grid_size = min(args.lambda_grid_size, 9)
        args.max_train = min(args.max_train, 500)
        args.max_raw_rows = 3000 if args.max_raw_rows <= 0 else min(args.max_raw_rows, 3000)
        args.workers = 1

    args.methods = parse_methods(args.methods)
    args.pc_rs = parse_floats(args.pc_rs)
    args.mb_gammas = parse_floats(args.mb_gammas)
    args.gh_gammas = parse_floats(args.gh_gammas)
    args.h_factors = parse_floats(args.h_factors)
    args.bias_screens = [s.strip() for s in str(args.bias_screens).split(",") if s.strip()]
    args.c_biases = parse_floats(args.c_biases)
    if args.unlab_reps <= 0:
        args.unlab_reps = 1
    if args.label_reps <= 0:
        args.label_reps = args.reps
    args.reps = int(args.unlab_reps * args.label_reps)
    args.backend_resolved = configure_backend(args.backend, args.gpu_id)
    if args.backend_resolved != "cpu" and args.workers != 1:
        print("[warning] GPU backend uses one worker to avoid process/GPU contention.", flush=True)
        args.workers = 1
    print(f"[backend] {args.backend_resolved}", flush=True)

    out_dir = ensure_dir(args.output_dir)
    predictor_start = time.perf_counter()
    data = prepare_blogfeedback_ppci(
        args.data,
        seed=args.seed,
        n_x0=args.n_x0,
        ppci_fraction=args.ppci_fraction,
        max_train=args.max_train,
        model=args.model,
        model_n_jobs=args.model_n_jobs,
        include_x0_in_model_train=False,
        max_raw_rows=None if args.max_raw_rows <= 0 else args.max_raw_rows,
    )
    predictor_training_seconds = float(time.perf_counter() - predictor_start)
    Xp, Yp, fp = data["X_ppci"], data["Y_ppci"], data["f_ppci"]
    print(
        f"[predictor] train_n={data['n_model_train']}, ppci_pool_n={data['n_ppci_pool']}, "
        f"targets_excluded={data['targets_excluded_from_model']}",
        flush=True,
    )
    write_run_manifest(
        out_dir / "run_manifest.json",
        args,
        extra={
            "n_model_train": data["n_model_train"],
            "n_ppci_pool": data["n_ppci_pool"],
            "targets_excluded_from_model": data["targets_excluded_from_model"],
            "reference_population": data["reference_population"],
            "predictor_training_seconds": predictor_training_seconds,
        },
    )
    x0_all = list(enumerate(zip(data["x0"][: args.n_x0], data["theta0"][: args.n_x0])))
    if str(args.x0_indices).strip():
        wanted = set(parse_ints(args.x0_indices))
        x0_list = [(idx, pair) for idx, pair in x0_all if idx in wanted]
    else:
        x0_list = x0_all
    if not x0_list:
        raise ValueError("--x0-indices selected no targets.")
    data_tuple = (Xp, Yp, fp)

    if args.workers and args.workers > 1 and len(x0_list) > 1:
        try:
            from joblib import Parallel, delayed

            nested = Parallel(n_jobs=args.workers, backend="loky")(
                delayed(run_one_x0)(idx, x0, float(theta0), data_tuple, args)
                for idx, (x0, theta0) in x0_list
            )
            rows = [r for block in nested for r in block]
        except Exception as exc:
            print(f"[warning] parallel execution failed ({exc}); falling back to sequential.", flush=True)
            rows = []
            for idx, (x0, theta0) in x0_list:
                rows.extend(run_one_x0(idx, x0, float(theta0), data_tuple, args))
    else:
        rows = []
        for idx, (x0, theta0) in x0_list:
            rows.extend(run_one_x0(idx, x0, float(theta0), data_tuple, args))

    rep = pd.DataFrame(rows)
    gh_label = first_gh_label(args)
    by_x0 = summarize_replicates(rep, ["x0_index"])
    by_x0 = add_ratios(by_x0, ["x0_index"], gh_label)
    by_x0 = add_ratios(by_x0, ["x0_index"], "INC")
    by_x0_path = out_dir / "summary_by_x0.csv"
    by_x0.to_csv(by_x0_path, index=False)

    aggregate = aggregate_from_x0(by_x0)
    aggregate_path = out_dir / "aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    tune = compact_tuning_decisions(rep)
    tune_path = out_dir / "tuning_decisions.csv"
    tune.to_csv(tune_path, index=False)
    tune_by_x0 = summarize_tuning_decisions(tune, ["x0_index"])
    tune_by_x0_path = out_dir / "tuning_by_x0.csv"
    tune_by_x0.to_csv(tune_by_x0_path, index=False)
    tune_by_method = summarize_tuning_decisions(tune, [])
    tune_by_method_path = out_dir / "tuning_by_method.csv"
    tune_by_method.to_csv(tune_by_method_path, index=False)

    if args.save_replicates:
        rep_path = out_dir / "replicates.csv"
        rep.to_csv(rep_path, index=False)
        print(f"Saved {rep_path}")
    else:
        print("Skipped replicate-level CSV; use --save-replicates to write it.")

    write_report(out_dir, args, aggregate, by_x0, tune_by_method)
    print(f"Saved {by_x0_path}")
    print(f"Saved {aggregate_path}")
    print(f"Saved {tune_path}")
    print(f"Saved {tune_by_x0_path}")
    print(f"Saved {tune_by_method_path}")


if __name__ == "__main__":
    main()
