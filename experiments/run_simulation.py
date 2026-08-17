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
from dataclasses import asdict
import numpy as np
import pandas as pd

from ppci_condmean.data import (
    generate_simulation_unlabeled,
    m_true_simulation,
    simulation_predictor,
    standardize_unif01,
)
from ppci_condmean.estimator import (
    lo_mean_from_weights,
    ppci_mean_from_weight_values,
    ppci_plus_mean_from_weight_values,
    ppi_global_mean,
)
from ppci_condmean.weights import RKHSLocalizationWeight
from ppci_condmean.joint_tuning import (
    JointTuningConfig,
    collect_joint_candidate_cache,
    select_joint_from_cache,
    tune_joint_from_covariates,
    weight_from_joint_cache,
)
from ppci_condmean.utils import ensure_dir, write_run_manifest
from ppci_condmean.gpu import configure_backend


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def parse_methods(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(float(x)) for x in str(s).split(",") if str(x).strip()]


def stable_setting_offset(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(str(name))) % 99991


def make_x0_grid(num: int, region: str) -> list[np.ndarray]:
    region_raw = str(region)
    region = region_raw.lower()
    if region in {"boundary", "near_boundary", "default"}:
        vals = np.linspace(0.72, 0.84, int(num))
    elif region in {"interior", "inside"}:
        vals = np.linspace(0.40, 0.55, int(num))
    elif region in {"edge", "far_boundary"}:
        vals = np.linspace(0.88, 0.95, int(num))
    elif region.startswith("cube:"):
        parts = region_raw.split(":")
        if len(parts) not in {3, 4}:
            raise ValueError("cube x0-region must be cube:lo:hi[:m], e.g. cube:0.7:0.85:10")
        lo = float(parts[1])
        hi = float(parts[2])
        m = int(parts[3]) if len(parts) == 4 else int(num)
        vals3 = np.linspace(lo, hi, m)
        return [np.array([a, b, c], dtype=float) for a in vals3 for b in vals3 for c in vals3]
    else:
        vals = np.array(parse_floats(region), dtype=float)
        if vals.size == 0:
            raise ValueError("Unknown x0-region. Use boundary, interior, edge, or comma-separated values.")
    return [np.array([a, a, a], dtype=float) for a in vals]


def subset_x0_grid(x0_list: list[np.ndarray], x0_indices: list[int] | None) -> list[tuple[int, np.ndarray]]:
    if not x0_indices:
        return list(enumerate(x0_list))
    n = len(x0_list)
    out: list[tuple[int, np.ndarray]] = []
    for idx in x0_indices:
        if idx < 0 or idx >= n:
            raise ValueError(f"x0 index {idx} is out of range for grid size {n}")
        out.append((idx, x0_list[idx]))
    return out


def make_cfg(args, *, method: str, pc_r: float | None = None, mb_gamma: float | None = None, gh_gamma: float | None = None) -> JointTuningConfig:
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
        pc_r=(args.pc_rs[0] if pc_r is None and getattr(args, "pc_rs", None) else 1.04) if pc_r is None else float(pc_r),
        mb_gamma=(args.mb_gammas[0] if mb_gamma is None and getattr(args, "mb_gammas", None) else 0.25) if mb_gamma is None else float(mb_gamma),
        gh_gamma=(args.gh_gammas[0] if gh_gamma is None and getattr(args, "gh_gammas", None) else 0.30) if gh_gamma is None else float(gh_gamma),
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


def build_twofold_joint(X_u: np.ndarray, x0: np.ndarray, n_label: int, seed: int, method: str, cfg: JointTuningConfig):
    rng = np.random.default_rng(seed)
    N = X_u.shape[0]
    perm = rng.permutation(N)
    I1 = np.sort(perm[: N // 2])
    I2 = np.sort(perm[N // 2:])
    tr1 = tune_joint_from_covariates(X_u[I1], x0, n=n_label, method=method, cfg=cfg)
    tr2 = tune_joint_from_covariates(X_u[I2], x0, n=n_label, method=method, cfg=cfg)
    w1 = RKHSLocalizationWeight(X_u[I1], x0, tr1.h, tr1.lam, cfg.kernel, backend=cfg.backend)
    w2 = RKHSLocalizationWeight(X_u[I2], x0, tr2.h, tr2.lam, cfg.kernel, backend=cfg.backend)
    w_u = np.zeros(N, dtype=float)
    # Cross-fitted unlabeled weights: evaluate held-out fold by the weight trained on the other fold.
    w_u[I1] = w2(X_u[I1])
    w_u[I2] = w1(X_u[I2])
    return w1, w2, w_u, tr1, tr2


def build_twofold_joint_cached(
    X_u: np.ndarray,
    x0: np.ndarray,
    n_label: int,
    seed: int,
    methods_to_run: list[tuple[str, str, JointTuningConfig]],
    *,
    include_fix3: bool,
    base_cfg: JointTuningConfig,
) -> dict:
    rng = np.random.default_rng(seed)
    N = X_u.shape[0]
    perm = rng.permutation(N)
    I1 = np.sort(perm[: N // 2])
    I2 = np.sort(perm[N // 2:])
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

    if include_fix3 and "INC" in tuned:
        from copy import deepcopy
        _, _, _, tr1_inc, tr2_inc = tuned["INC"]
        tr1_fix = deepcopy(tr1_inc)
        tr2_fix = deepcopy(tr2_inc)
        lam_fix = 3.0 / float(max(n_label, 1))
        tr1_fix.lam = lam_fix
        tr2_fix.lam = lam_fix
        tr1_fix.lambda_factor = 3.0
        tr2_fix.lambda_factor = 3.0
        w1_fix = get_weight("fold1", cache1, tr1_fix, base_cfg)
        w2_fix = get_weight("fold2", cache2, tr2_fix, base_cfg)
        w_u_fix = np.zeros(N, dtype=float)
        w_u_fix[I1] = w2_fix(X_u[I1])
        w_u_fix[I2] = w1_fix(X_u[I2])
        tuned["FIX3N"] = (w1_fix, w2_fix, w_u_fix, tr1_fix, tr2_fix)

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
    return "LO_INCw" if source_label == "INC" else "LO"


def row_from_result(res, theta0: float, context: dict, tr1=None, tr2=None) -> dict:
    d = res.as_dict()
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
                "D_h_point", "D_h_op", "sw_proxy", "power",
                "power_ref", "pc_ratio", "m1_norm", "M2_nuc", "B_geo", "R_MB", "R_GH",
                "bias_screen", "c_bias", "bias_score", "bias_score_label", "bias_score_full",
                "bias_budget", "gh_gamma_used", "gh_gamma_eff", "edge_score", "A_score", "M2_lambda_min",
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
        d["A_score_mean"] = 0.5 * (tr1.A_score + tr2.A_score)
        d["M2_lambda_min_mean"] = 0.5 * (tr1.M2_lambda_min + tr2.M2_lambda_min)
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


def summarize_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, method), g in df.groupby(["setting", "method"], dropna=False):
        err = g["error"].to_numpy(float)
        n_clusters, cov_cluster_se, cov_cluster_sd = cluster_coverage_stats(g)
        reps_per_cluster = len(g) / n_clusters if n_clusters > 0 else 1.0
        tuning_seconds = float(g["tuning_seconds"].mean()) if "tuning_seconds" in g else np.nan
        estimation_seconds = float(g["estimation_seconds"].mean()) if "estimation_seconds" in g else np.nan
        rows.append({
            "setting": setting,
            "method": method,
            "n_reps": int(len(g)),
            "n_unlab_clusters": n_clusters,
            "coverage": float(g["covered"].mean()),
            "coverage_cluster_se": cov_cluster_se,
            "coverage_cluster_sd": cov_cluster_sd,
            "bias": float(np.mean(err)),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "se_mean": float(g["se"].mean()),
            "width": float(g["width"].mean()),
            "emp_sd": float(np.std(g["theta_hat"].to_numpy(float), ddof=1)) if len(g) > 1 else 0.0,
            "lambda_factor_mean": float(g["lambda_factor"].mean()) if "lambda_factor" in g else np.nan,
            "h_factor_mean": float(g["h_factor"].mean()) if "h_factor" in g else np.nan,
            "k_target_mean": float(g["k_target_mean"].mean()) if "k_target_mean" in g else np.nan,
            "R_MB_mean": float(g["R_MB_mean"].mean()) if "R_MB_mean" in g else np.nan,
            "R_GH_mean": float(g["R_GH_mean"].mean()) if "R_GH_mean" in g else np.nan,
            "bias_score_mean": float(g["bias_score_mean"].mean()) if "bias_score_mean" in g else np.nan,
            "bias_budget_mean": float(g["bias_budget_mean"].mean()) if "bias_budget_mean" in g else np.nan,
            "gh_gamma_eff_mean": float(g["gh_gamma_eff_mean"].mean()) if "gh_gamma_eff_mean" in g else np.nan,
            "edge_score_mean": float(g["edge_score_mean"].mean()) if "edge_score_mean" in g else np.nan,
            "A_score_mean": float(g["A_score_mean"].mean()) if "A_score_mean" in g else np.nan,
            "pc_ratio_mean": float(g["pc_ratio_mean"].mean()) if "pc_ratio_mean" in g else np.nan,
            "fallback_rate": float(g["tuning_status"].astype(str).str.contains("fallback").mean()) if "tuning_status" in g else np.nan,
            "omega_mean": float(g["omega"].mean()) if "omega" in g and g["omega"].notna().any() else np.nan,
            "omega_clipped_rate": float(g["omega_clipped_rate"].mean()) if "omega_clipped_rate" in g and g["omega_clipped_rate"].notna().any() else np.nan,
            "omega_sd_mean": float(g["omega_sd"].mean()) if "omega_sd" in g and g["omega_sd"].notna().any() else np.nan,
            "omega_min_mean": float(g["omega_min"].mean()) if "omega_min" in g and g["omega_min"].notna().any() else np.nan,
            "omega_max_mean": float(g["omega_max"].mean()) if "omega_max" in g and g["omega_max"].notna().any() else np.nan,
            "sigma2_Y_mean": float(g["sigma2_Y"].mean()) if "sigma2_Y" in g and g["sigma2_Y"].notna().any() else np.nan,
            "sigma2_Y_minus_f_mean": float(g["sigma2_Y_minus_f"].mean()) if "sigma2_Y_minus_f" in g and g["sigma2_Y_minus_f"].notna().any() else np.nan,
            "sigma2_f_mean": float(g["sigma2_f"].mean()) if "sigma2_f" in g and g["sigma2_f"].notna().any() else np.nan,
            "tuning_seconds_mean_per_unlab_draw": tuning_seconds,
            "estimation_seconds_mean": estimation_seconds,
            "amortized_procedure_seconds_mean": tuning_seconds / reps_per_cluster + estimation_seconds,
        })
    out = pd.DataFrame(rows)
    # Ratios relative to INC within each setting.
    ratio_rows = []
    for setting, gs in out.groupby("setting"):
        inc = gs[gs["method"] == "INC"]
        if inc.empty:
            ratio_rows.append(gs)
            continue
        w0 = float(inc["width"].iloc[0])
        r0 = float(inc["rmse"].iloc[0])
        gs = gs.copy()
        gs["width_ratio_vs_INC"] = gs["width"] / w0 if w0 > 0 else np.nan
        gs["rmse_ratio_vs_INC"] = gs["rmse"] / r0 if r0 > 0 else np.nan
        ratio_rows.append(gs)
    return pd.concat(ratio_rows, ignore_index=True) if ratio_rows else out


def summarize_by_x0(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["setting", "method", "x0_index", "x0_1"]
    for coord in ["x0_2", "x0_3"]:
        if coord in df.columns:
            keys.append(coord)
    for key, g in df.groupby(keys, dropna=False):
        key_dict = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        setting = key_dict["setting"]
        method = key_dict["method"]
        x0_index = key_dict["x0_index"]
        x0_1 = key_dict["x0_1"]
        err = g["error"].to_numpy(float)
        n_clusters, cov_cluster_se, cov_cluster_sd = cluster_coverage_stats(g)
        reps_per_cluster = len(g) / n_clusters if n_clusters > 0 else 1.0
        tuning_seconds = float(g["tuning_seconds"].mean()) if "tuning_seconds" in g else np.nan
        estimation_seconds = float(g["estimation_seconds"].mean()) if "estimation_seconds" in g else np.nan
        row = {
            "setting": setting,
            "method": method,
            "x0_index": int(x0_index),
            "x0_1": float(x0_1),
            "n_reps": int(len(g)),
            "n_unlab_clusters": n_clusters,
            "coverage": float(g["covered"].mean()),
            "coverage_cluster_se": cov_cluster_se,
            "coverage_cluster_sd": cov_cluster_sd,
            "bias": float(np.mean(err)),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "se_mean": float(g["se"].mean()),
            "width": float(g["width"].mean()),
            "emp_sd": float(np.std(g["theta_hat"].to_numpy(float), ddof=1)) if len(g) > 1 else 0.0,
            "omega_mean": float(g["omega"].mean()) if "omega" in g and g["omega"].notna().any() else np.nan,
            "omega_clipped_rate": float(g["omega_clipped_rate"].mean()) if "omega_clipped_rate" in g and g["omega_clipped_rate"].notna().any() else np.nan,
            "omega_sd_mean": float(g["omega_sd"].mean()) if "omega_sd" in g and g["omega_sd"].notna().any() else np.nan,
            "omega_min_mean": float(g["omega_min"].mean()) if "omega_min" in g and g["omega_min"].notna().any() else np.nan,
            "omega_max_mean": float(g["omega_max"].mean()) if "omega_max" in g and g["omega_max"].notna().any() else np.nan,
            "sigma2_Y_mean": float(g["sigma2_Y"].mean()) if "sigma2_Y" in g and g["sigma2_Y"].notna().any() else np.nan,
            "sigma2_Y_minus_f_mean": float(g["sigma2_Y_minus_f"].mean()) if "sigma2_Y_minus_f" in g and g["sigma2_Y_minus_f"].notna().any() else np.nan,
            "sigma2_f_mean": float(g["sigma2_f"].mean()) if "sigma2_f" in g and g["sigma2_f"].notna().any() else np.nan,
            "tuning_seconds_mean_per_unlab_draw": tuning_seconds,
            "estimation_seconds_mean": estimation_seconds,
            "amortized_procedure_seconds_mean": tuning_seconds / reps_per_cluster + estimation_seconds,
        }
        if "x0_2" in key_dict:
            row["x0_2"] = float(key_dict["x0_2"])
        if "x0_3" in key_dict:
            row["x0_3"] = float(key_dict["x0_3"])
        rows.append(row)
    out = pd.DataFrame(rows)
    ratio_rows = []
    for (setting, x0_index), gs in out.groupby(["setting", "x0_index"], dropna=False):
        inc = gs[gs["method"] == "INC"]
        if inc.empty:
            ratio_rows.append(gs)
            continue
        w0 = float(inc["width"].iloc[0])
        r0 = float(inc["rmse"].iloc[0])
        gs = gs.copy()
        gs["width_ratio_vs_INC"] = gs["width"] / w0 if w0 > 0 else np.nan
        gs["rmse_ratio_vs_INC"] = gs["rmse"] / r0 if r0 > 0 else np.nan
        ratio_rows.append(gs)
    return pd.concat(ratio_rows, ignore_index=True) if ratio_rows else out


def compact_tuning_decisions(df: pd.DataFrame, k_max_frac: float) -> pd.DataFrame:
    keys = ["setting", "method", "x0_index", "x0_1", "unlab_rep"]
    for coord in ["x0_2", "x0_3"]:
        if coord in df.columns:
            keys.insert(keys.index("unlab_rep"), coord)
    cols = keys + [
        "n_label",
        "n_unlab",
        "tuning_status",
        "lambda_factor",
        "h_factor",
        "k_target_mean",
        "R_MB_mean",
        "R_GH_mean",
        "bias_score_mean",
        "bias_budget_mean",
        "gh_gamma_eff_mean",
        "edge_score_mean",
        "A_score_mean",
        "M2_lambda_min_mean",
        "pc_ratio_mean",
        "J_w_mean",
        "sw_proxy_mean",
        "op_score",
        "loc_score",
        "tune1_k_target",
        "tune2_k_target",
    ]
    keep = [c for c in cols if c in df.columns]
    tune = df.drop_duplicates(keys)[keep].copy()
    tune["fallback"] = tune["tuning_status"].astype(str).str.contains("fallback", regex=False)
    fold_m = (tune["n_unlab"].astype(int) // 2).clip(lower=1)
    tune["k_max"] = np.floor(float(k_max_frac) * fold_m).clip(2).astype(int)
    tune["boundary"] = (
        (tune["tune1_k_target"].astype(float) >= tune["k_max"].astype(float))
        | (tune["tune2_k_target"].astype(float) >= tune["k_max"].astype(float))
    )
    return tune


def summarize_tuning_decisions(df: pd.DataFrame, k_max_frac: float) -> pd.DataFrame:
    tune = compact_tuning_decisions(df, k_max_frac)
    value_cols = [
        "lambda_factor",
        "h_factor",
        "k_target_mean",
        "R_MB_mean",
        "R_GH_mean",
        "gh_gamma_eff_mean",
        "edge_score_mean",
        "A_score_mean",
        "M2_lambda_min_mean",
        "pc_ratio_mean",
        "J_w_mean",
        "sw_proxy_mean",
        "op_score",
        "loc_score",
    ]
    rows = []
    for (setting, method), g in tune.groupby(["setting", "method"], dropna=False):
        row = {
            "setting": setting,
            "method": method,
            "tuning_decisions": int(len(g)),
            "fallback_rate": float(g["fallback"].mean()),
            "boundary_rate": float(g["boundary"].mean()),
        }
        for col in value_cols:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_one_setting(
    args,
    setting_name: str,
    n_label: int,
    n_unlab: int,
    sigma_eps_values: list[float],
    predictor_qualities: list[float],
    x0_region: str,
):
    rows = []
    x0_list = make_x0_grid(args.x0_num, x0_region)
    indexed_x0 = subset_x0_grid(x0_list, args.x0_indices)
    methods_to_run: list[tuple[str, str, JointTuningConfig]] = []
    base_cfg = make_cfg(args, method="INC")
    if "INC" in args.methods:
        methods_to_run.append(("INC", "INC", base_cfg))
    for r in args.pc_rs:
        if "PC" in args.methods:
            label = f"PC_r{r:g}"
            methods_to_run.append((label, "PC", make_cfg(args, method="PC", pc_r=r)))
    for g in args.mb_gammas:
        if "MB" in args.methods:
            label = f"MB_g{g:g}"
            methods_to_run.append((label, "MB", make_cfg(args, method="MB", mb_gamma=g)))
    if "GH" in args.methods:
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
                        if args.gh_adaptive and args.gh_adaptive_rule == "log_ratio":
                            label = f"GH_log{args.gh_c0:g}"
                        else:
                            label = f"GH_ad{args.gh_gamma0:g}" if args.gh_adaptive else f"GH_g{g:g}"
                    else:
                        screen_label = {"p1_label": "P1", "p2_log": "P2", "p3_full": "P3"}.get(str(screen).lower(), str(screen))
                        label = "PPCI" if str(screen).lower() == "p1_label" and len(args.c_biases) == 1 else f"GH_{screen_label}_c{c_bias:g}"
                    if args.gh_edge_rho > 0:
                        label += f"_E{args.gh_edge_rho:g}"
                    if np.isfinite(args.gh_a_tau):
                        label += f"_A{args.gh_a_tau:g}"
                    if np.isfinite(args.gh_pc_r):
                        label += f"_PC{args.gh_pc_r:g}"
                    methods_to_run.append((label, "GH", make_cfg(args, method="GH", gh_gamma=g)))
    # Optional fixed 3/n diagnostic uses INC h and fixed lambda 3/n, no separate tuning family.
    include_fix3 = "FIX3N" in args.methods

    for x0_index, x0_raw in indexed_x0:
        theta0 = float(m_true_simulation(x0_raw.reshape(1, -1))[0])
        x0_scaled = standardize_unif01(x0_raw.reshape(1, -1))[0]
        for urep in range(args.unlab_reps):
            seed_u = args.seed + stable_setting_offset(setting_name) + 10000 * x0_index + 1000 * urep
            rng_u = np.random.default_rng(seed_u)
            X_u_raw, X_u, _ = generate_simulation_unlabeled(
                rng_u, n_unlab, predictor_quality=predictor_qualities[0]
            )
            tuning_start = time.perf_counter()
            tuned = build_twofold_joint_cached(
                X_u,
                x0_scaled,
                n_label,
                seed_u + 17,
                methods_to_run,
                include_fix3=include_fix3,
                base_cfg=base_cfg,
            )
            tuning_seconds = float(time.perf_counter() - tuning_start)

            for brep in range(args.label_reps):
                seed_l = args.seed + 5000000 + 200000 * x0_index + 10000 * urep + brep
                rng_l = np.random.default_rng(seed_l)
                X_l_raw = rng_l.uniform(0.0, 1.0, size=(n_label, 3))
                X_l = standardize_unif01(X_l_raw)
                m_l = m_true_simulation(X_l_raw)
                eps_standard = rng_l.normal(size=n_label)
                for sigma_eps in sigma_eps_values:
                    Y_l = m_l + float(sigma_eps) * eps_standard
                    for predictor_quality in predictor_qualities:
                        f_l = simulation_predictor(X_l_raw, predictor_quality)
                        f_u = simulation_predictor(X_u_raw, predictor_quality)
                        scenario = f"{setting_name}_q{predictor_quality:g}_s{sigma_eps:g}"
                        ctx = {
                            "setting": scenario,
                            "setting_family": setting_name,
                            "n_label": n_label,
                            "n_unlab": n_unlab,
                            "sigma_eps": sigma_eps,
                            "predictor_quality": predictor_quality,
                            "x0_region": x0_region,
                            "x0_index": x0_index,
                            "x0_1": float(x0_raw[0]),
                            "x0_2": float(x0_raw[1]) if len(x0_raw) > 1 else np.nan,
                            "x0_3": float(x0_raw[2]) if len(x0_raw) > 2 else np.nan,
                            "unlab_rep": urep,
                            "label_rep": brep,
                            "tuning_seconds": tuning_seconds,
                        }
                        for label, packed in tuned.items():
                            estimate_start = time.perf_counter()
                            w1, w2, w_u, tr1, tr2 = packed
                            w_l = 0.5 * (w1(X_l) + w2(X_l))
                            res = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u, method=label)
                            attach_tuning(res, tr1, tr2, label)
                            ctx_method = dict(ctx, estimation_seconds=float(time.perf_counter() - estimate_start))
                            rows.append(row_from_result(res, theta0, ctx_method, tr1, tr2))
                            if args.include_ppci_plus and label == "PPCI":
                                plus_start = time.perf_counter()
                                plus = ppci_plus_mean_from_weight_values(
                                    Y_l,
                                    f_l,
                                    f_u,
                                    w_l,
                                    w_u,
                                    rng=np.random.default_rng(seed_l + 73),
                                    omega_ridge=args.omega_ridge,
                                    omega_folds=args.omega_folds,
                                    method="PPCI++",
                                )
                                attach_tuning(plus, tr1, tr2, "PPCI++")
                                ctx_plus = dict(ctx, estimation_seconds=float(time.perf_counter() - plus_start))
                                rows.append(row_from_result(plus, theta0, ctx_plus, tr1, tr2))
                        if args.include_lo_ppi:
                            for source_label in lo_source_labels(tuned, args):
                                lo_start = time.perf_counter()
                                w1, w2, _, tr1, tr2 = tuned[source_label]
                                w_l = 0.5 * (w1(X_l) + w2(X_l))
                                lo = lo_mean_from_weights(X_l, Y_l, w_l)
                                lo.method = lo_method_label(source_label)
                                ctx_lo = dict(ctx, estimation_seconds=float(time.perf_counter() - lo_start))
                                rows.append(row_from_result(lo, theta0, ctx_lo, tr1, tr2))
                            ppi_start = time.perf_counter()
                            ppi = ppi_global_mean(Y_l, f_l, f_u)
                            ppi.method = "PPI"
                            ctx_ppi = dict(ctx, tuning_seconds=0.0, estimation_seconds=float(time.perf_counter() - ppi_start))
                            rows.append(row_from_result(ppi, theta0, ctx_ppi))
        print(f"[done] setting={setting_name}, x0_index={x0_index}, theta0={theta0:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="PPCI conditional-mean simulation with P1 covariate-only tuning.")
    ap.add_argument("--output-dir", default="runs/simulation")
    ap.add_argument("--seed", type=int, default=12100)
    ap.add_argument("--settings", default="base", help="Comma-separated: base, smalln, largen, moreN, moreN2, smallN, betterf, worsef, highnoise, lownoise, interior, edge")
    ap.add_argument("--methods", default="GH", help="Comma-separated subset of INC,PC,MB,GH,FIX3N")
    ap.add_argument("--n-label", type=int, default=200)
    ap.add_argument("--n-unlab", type=int, default=10000)
    ap.add_argument("--x0-num", type=int, default=10)
    ap.add_argument("--x0-region", default="cube:0.7:0.85:10")
    ap.add_argument("--x0-indices", default="", help="Optional comma-separated x0 indices for sharded runs.")
    ap.add_argument("--unlab-reps", type=int, default=50)
    ap.add_argument("--label-reps", type=int, default=20)
    ap.add_argument("--sigma-eps", type=float, default=1.0)
    ap.add_argument("--predictor-quality", type=float, default=0.9, help="q in f_q(X)=q m(X)+(1-q)s(X).")
    ap.add_argument("--predictor-qualities", default="", help="Optional comma-separated q ladder evaluated with shared weights.")
    ap.add_argument("--sigma-eps-values", default="", help="Optional comma-separated noise-SD ladder evaluated with shared weights.")
    ap.add_argument("--include-ppci-plus", action="store_true", help="Also run labeled-cross-fitted data-driven PPCI++.")
    ap.add_argument("--omega-ridge", type=float, default=1e-6)
    ap.add_argument("--omega-folds", type=int, default=5)
    ap.add_argument("--pc-rs", default="1.04,1.08")
    ap.add_argument("--mb-gammas", default="0.20,0.25")
    ap.add_argument("--gh-gammas", default="0.25,0.30,0.35")
    ap.add_argument("--gh-adaptive", action="store_true")
    ap.add_argument("--gh-adaptive-rule", default="legacy", choices=["legacy", "log_ratio"])
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
    ap.add_argument("--c-biases", default="0.10", help="Comma-separated c_bias values for A_bias.")
    ap.add_argument("--k-min-floor", type=int, default=50)
    ap.add_argument("--k-max-frac", type=float, default=0.80)
    ap.add_argument("--k-growth", type=float, default=1.50)
    ap.add_argument("--lambda-factor-min", type=float, default=0.02)
    ap.add_argument("--lambda-factor-max", type=float, default=60.0)
    ap.add_argument("--lambda-grid-size", type=int, default=35)
    ap.add_argument("--lambda-grid-mode", default="shrinking", choices=["n", "shrinking"])
    ap.add_argument("--tau-op", type=float, default=12.0)
    ap.add_argument("--tau-loc", type=float, default=4.0)
    ap.add_argument("--min-abs-j", type=float, default=1e-6)
    ap.add_argument("--kernel", default="matern52")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "torch", "gpu", "cuda"])
    ap.add_argument("--gpu-id", default="auto")
    ap.add_argument("--include-lo-ppi", action="store_true", default=True)
    ap.add_argument("--no-lo-ppi", dest="include_lo_ppi", action="store_false")
    ap.add_argument("--lo-weight-method", default="first", choices=["first", "all", "inc"], help="Which tuned weights to use for LO when --include-lo-ppi is set.")
    ap.add_argument("--save-replicates", action="store_true", help="Write replicate-level CSV. Off by default to keep server runs compact.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.settings = "base"
        args.x0_num = 2
        args.x0_region = "cube:0.7:0.85:2"
        args.unlab_reps = 1
        args.label_reps = 3
        args.n_label = min(args.n_label, 80)
        args.n_unlab = min(args.n_unlab, 300)
        args.lambda_grid_size = min(args.lambda_grid_size, 11)
        args.pc_rs = "1.04"
        args.mb_gammas = "0.25"
        args.gh_gammas = "0.30"

    args.methods = parse_methods(args.methods)
    args.x0_indices = parse_ints(args.x0_indices) if str(args.x0_indices).strip() else []
    args.pc_rs = parse_floats(args.pc_rs)
    args.mb_gammas = parse_floats(args.mb_gammas)
    args.gh_gammas = parse_floats(args.gh_gammas)
    args.h_factors = parse_floats(args.h_factors)
    args.bias_screens = [s.strip() for s in str(args.bias_screens).split(",") if s.strip()]
    args.c_biases = parse_floats(args.c_biases)
    args.predictor_qualities = parse_floats(args.predictor_qualities) if str(args.predictor_qualities).strip() else [args.predictor_quality]
    args.sigma_eps_values = parse_floats(args.sigma_eps_values) if str(args.sigma_eps_values).strip() else [args.sigma_eps]
    args.backend_resolved = configure_backend(args.backend, args.gpu_id)
    print(f"[backend] {args.backend_resolved}", flush=True)

    setting_specs = []
    for s in parse_methods(args.settings):
        key = s.lower()
        n, N = args.n_label, args.n_unlab
        sigma_eps_values, predictor_qualities = list(args.sigma_eps_values), list(args.predictor_qualities)
        x0_region = args.x0_region
        if s == "smallN":
            N = 500
        elif key == "base":
            pass
        elif key == "smalln":
            n = 100
        elif key == "largen":
            n = 300
        elif key == "moren":
            N = 2000
        elif key == "moren2":
            N = 4000
        elif key == "smalln_unlab":
            N = 500
        elif key == "betterf":
            predictor_qualities = [0.98]
        elif key == "worsef":
            predictor_qualities = [0.25]
        elif key == "highnoise":
            sigma_eps_values = [3.0]
        elif key == "lownoise":
            sigma_eps_values = [1.0]
        elif key == "interior":
            x0_region = "interior"
        elif key == "edge":
            x0_region = "edge"
        else:
            raise ValueError(f"Unknown setting: {s}")
        setting_specs.append((s, n, N, sigma_eps_values, predictor_qualities, x0_region))

    out_dir = ensure_dir(args.output_dir)
    write_run_manifest(out_dir / "run_manifest.json", args)
    all_rows = []
    for spec in setting_specs:
        all_rows.extend(run_one_setting(args, *spec))

    rep = pd.DataFrame(all_rows)
    summary = summarize_long(rep)
    summary_path = out_dir / "summary_by_setting.csv"
    summary.to_csv(summary_path, index=False)
    summary_x0 = summarize_by_x0(rep)
    summary_x0_path = out_dir / "summary_by_x0.csv"
    summary_x0.to_csv(summary_x0_path, index=False)
    tuning_decisions = compact_tuning_decisions(rep, args.k_max_frac)
    tuning_decisions_path = out_dir / "tuning_decisions.csv"
    tuning_decisions.to_csv(tuning_decisions_path, index=False)
    tuning = summarize_tuning_decisions(rep, args.k_max_frac)
    tuning_path = out_dir / "tuning_diagnostics.csv"
    tuning.to_csv(tuning_path, index=False)
    # Aggregate across settings with equal weight per setting.
    agg_rows = []
    for method, g in summary.groupby("method"):
        agg_rows.append({
            "method": method,
            "n_settings": int(g["setting"].nunique()),
            "coverage_mean": float(g["coverage"].mean()),
            "coverage_min_across_settings": float(g["coverage"].min()),
            "coverage_cluster_se_mean": float(g["coverage_cluster_se"].mean()) if "coverage_cluster_se" in g and g["coverage_cluster_se"].notna().any() else np.nan,
            "coverage_cluster_se_max": float(g["coverage_cluster_se"].max()) if "coverage_cluster_se" in g and g["coverage_cluster_se"].notna().any() else np.nan,
            "width_ratio_mean": float(g.get("width_ratio_vs_INC", pd.Series([np.nan])).mean()),
            "rmse_ratio_mean": float(g.get("rmse_ratio_vs_INC", pd.Series([np.nan])).mean()),
            "lambda_factor_mean": float(g["lambda_factor_mean"].mean()),
            "k_target_mean": float(g["k_target_mean"].mean()),
            "bias_score_mean": float(g["bias_score_mean"].mean()) if "bias_score_mean" in g else np.nan,
            "bias_budget_mean": float(g["bias_budget_mean"].mean()) if "bias_budget_mean" in g else np.nan,
            "gh_gamma_eff_mean": float(g["gh_gamma_eff_mean"].mean()) if "gh_gamma_eff_mean" in g else np.nan,
            "edge_score_mean": float(g["edge_score_mean"].mean()) if "edge_score_mean" in g else np.nan,
            "A_score_mean": float(g["A_score_mean"].mean()) if "A_score_mean" in g else np.nan,
            "fallback_rate_mean": float(g["fallback_rate"].mean()),
        })
    agg = pd.DataFrame(agg_rows).sort_values(["method"])
    agg_path = out_dir / "aggregate.csv"
    agg.to_csv(agg_path, index=False)
    if args.save_replicates:
        rep_path = out_dir / "replicates.csv"
        rep.to_csv(rep_path, index=False)
        print(f"Saved {rep_path}")
    else:
        print("Skipped replicate-level CSV; use --save-replicates to write it.")
    print(f"Saved {summary_path}")
    print(f"Saved {summary_x0_path}")
    print(f"Saved {tuning_decisions_path}")
    print(f"Saved {tuning_path}")
    print(f"Saved {agg_path}")


if __name__ == "__main__":
    main()
