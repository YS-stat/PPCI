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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ppci_condmean.data import census_sex_subset, load_census_npz, nw_oracle_mean
from ppci_condmean.estimator import ppci_mean_from_weight_values
from ppci_condmean.gpu import configure_backend
from ppci_condmean.joint_tuning import (
    JointTuningConfig,
    collect_joint_candidate_cache,
    select_joint_from_cache,
    weight_from_joint_cache,
)
from ppci_condmean.utils import ensure_dir, standardize_apply, write_run_manifest


def parse_ints(value: str) -> list[int]:
    return [int(float(x)) for x in str(value).split(",") if x.strip()]


def twofold_weights(X_u: np.ndarray, x0: np.ndarray, n_label: int, seed: int, cfg: JointTuningConfig):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_u))
    i1 = np.sort(perm[: len(X_u) // 2])
    i2 = np.sort(perm[len(X_u) // 2 :])
    cache1 = collect_joint_candidate_cache(X_u[i1], x0, n=n_label, cfg=cfg)
    cache2 = collect_joint_candidate_cache(X_u[i2], x0, n=n_label, cfg=cfg)
    tr1 = select_joint_from_cache(cache1, "GH", cfg=cfg)
    tr2 = select_joint_from_cache(cache2, "GH", cfg=cfg)
    w1 = weight_from_joint_cache(cache1, tr1, cfg)
    w2 = weight_from_joint_cache(cache2, tr2, cfg)
    w_u = np.empty(len(X_u), dtype=float)
    w_u[i1] = w2(X_u[i1])
    w_u[i2] = w1(X_u[i2])
    return w1, w2, w_u, tr1, tr2


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for n_unlab, group in rows.groupby("n_unlab"):
        error = group["error"].to_numpy(float)
        cluster_cov = group.groupby("unlab_rep")["covered"].mean()
        out.append({
            "n_unlab": int(n_unlab),
            "n_reps": len(group),
            "n_unlab_clusters": cluster_cov.size,
            "coverage": group["covered"].mean(),
            "coverage_cluster_se": cluster_cov.std(ddof=1) / np.sqrt(cluster_cov.size),
            "bias": error.mean(),
            "rmse": np.sqrt(np.mean(error**2)),
            "emp_sd": group["theta_hat"].std(ddof=1),
            "se_mean": group["se"].mean(),
            "width": group["width"].mean(),
            "sigma2_Y_mean": group["sigma2_Y"].mean(),
            "sigma2_Y_minus_f_mean": group["sigma2_Y_minus_f"].mean(),
            "sigma2_f_mean": group["sigma2_f"].mean(),
            "h_mean": group["h_mean"].mean(),
            "lambda_mean": group["lambda_mean"].mean(),
            "fallback_rate": group["fallback"].mean(),
            "tuning_seconds_mean": group["tuning_seconds"].mean(),
            "estimation_seconds_mean": group["estimation_seconds"].mean(),
            "amortized_procedure_seconds_mean": group["amortized_procedure_seconds"].mean(),
        })
    return pd.DataFrame(out).sort_values("n_unlab")


def result_record(
    result,
    n_unlab: int,
    unlab_rep: int,
    label_rep: int,
    theta0: float,
    tr1,
    tr2,
    tuning_seconds: float,
    estimation_seconds: float,
    label_reps: int,
) -> dict:
    return {
        "n_unlab": n_unlab,
        "unlab_rep": unlab_rep,
        "label_rep": label_rep,
        "theta0": theta0,
        "theta_hat": result.theta_hat,
        "error": result.theta_hat - theta0,
        "covered": result.ci_low <= theta0 <= result.ci_high,
        "se": result.se,
        "width": result.ci_high - result.ci_low,
        "sigma2_Y": result.sigma2_Y,
        "sigma2_Y_minus_f": result.sigma2_Y_minus_f,
        "sigma2_f": result.sigma2_f,
        "h_mean": 0.5 * (tr1.h + tr2.h),
        "lambda_mean": 0.5 * (tr1.lam + tr2.lam),
        "fallback": "fallback" in tr1.status or "fallback" in tr2.status,
        "tuning_seconds": tuning_seconds,
        "estimation_seconds": estimation_seconds,
        "amortized_procedure_seconds": tuning_seconds / label_reps + estimation_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Income PPCI coverage/width sweep over unlabeled sample size.")
    parser.add_argument("--data", default="data/census_income/census_income.npz")
    parser.add_argument("--output-dir", default="runs/income_unlabeled_sweep")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--age", type=int, default=70)
    parser.add_argument("--sex", type=int, default=1)
    parser.add_argument("--n-label", type=int, default=300)
    parser.add_argument("--n-unlab-values", default="500,1500,2500,3500,4500,5500,6500,7500,8500,9500,10500")
    parser.add_argument("--unlab-reps", type=int, default=50)
    parser.add_argument("--unlab-rep-start", type=int, default=0, help="Global first cluster index, for sharding paired-N runs.")
    parser.add_argument("--label-reps", type=int, default=20)
    parser.add_argument("--paired-nested", action="store_true", default=True, help="Use nested unlabeled prefixes and shared labeled draws across N.")
    parser.add_argument("--independent-N", dest="paired_nested", action="store_false")
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "torch", "gpu", "cuda"])
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--save-replicates", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_label = min(args.n_label, 80)
        args.n_unlab_values = "200,300"
        args.unlab_reps = 2
        args.label_reps = 3

    backend = configure_backend(args.backend, args.gpu_id)
    cfg = JointTuningConfig(
        h_grid_mode="median_grid",
        h_factors=(1.0, 1.2, 1.4),
        lambda_factor_min=0.1,
        lambda_factor_max=1000.0,
        lambda_grid_size=41 if not args.smoke else 9,
        lambda_grid_mode="shrinking",
        tau_op=12.0,
        tau_loc=4.0,
        bias_screen="p1_label",
        c_bias=22.0,
        constraint_fallback="least_violation",
        backend=backend,
    )
    print(f"[backend] {backend}", flush=True)

    X, Y, f = load_census_npz(args.data)
    X_raw, X_std, Y_s, f_s, mean, std = census_sex_subset(X, Y, f, args.sex)
    x0_raw = np.array([float(args.age), float(args.sex)])
    x0 = standardize_apply(x0_raw.reshape(1, -1), mean, std)[0]
    theta0 = nw_oracle_mean(X_std, Y_s, x0)
    exact = np.all(X_raw == x0_raw.reshape(1, -1), axis=1)
    pool = np.arange(len(Y_s))[~exact]

    n_values = sorted(set(parse_ints(args.n_unlab_values)))
    if not n_values:
        raise ValueError("At least one unlabeled sample size is required.")
    if len(pool) < args.n_label + max(n_values):
        raise ValueError(f"Not enough observations for N={max(n_values)}.")

    rows = []
    if args.paired_nested:
        for urep in range(args.unlab_rep_start, args.unlab_rep_start + args.unlab_reps):
            seed_u = args.seed + 10000 * urep
            rng_u = np.random.default_rng(seed_u)
            un_idx_max = rng_u.choice(pool, size=max(n_values), replace=False)
            label_pool = np.setdiff1d(pool, un_idx_max, assume_unique=False)
            label_indices = [
                np.random.default_rng(seed_u + 5000000 + lrep).choice(
                    label_pool, size=args.n_label, replace=False
                )
                for lrep in range(args.label_reps)
            ]
            for n_unlab in n_values:
                un_idx = un_idx_max[:n_unlab]
                X_u, f_u = X_std[un_idx], f_s[un_idx]
                started = time.perf_counter()
                w1, w2, w_u, tr1, tr2 = twofold_weights(X_u, x0, args.n_label, seed_u + 17, cfg)
                tuning_seconds = float(time.perf_counter() - started)
                for lrep, lab_idx in enumerate(label_indices):
                    X_l, Y_l, f_l = X_std[lab_idx], Y_s[lab_idx], f_s[lab_idx]
                    estimation_start = time.perf_counter()
                    w_l = 0.5 * (w1(X_l) + w2(X_l))
                    result = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u)
                    estimation_seconds = float(time.perf_counter() - estimation_start)
                    rows.append(result_record(
                        result, n_unlab, urep, lrep, theta0, tr1, tr2,
                        tuning_seconds, estimation_seconds, args.label_reps,
                    ))
            print(f"[done] paired cluster={urep}", flush=True)
    else:
        for n_unlab in n_values:
            for local_urep in range(args.unlab_reps):
                urep = args.unlab_rep_start + local_urep
                seed_u = args.seed + 1000000 * n_unlab + 10000 * urep
                rng_u = np.random.default_rng(seed_u)
                un_idx = rng_u.choice(pool, size=n_unlab, replace=False)
                label_pool = np.setdiff1d(pool, un_idx, assume_unique=False)
                X_u, f_u = X_std[un_idx], f_s[un_idx]
                started = time.perf_counter()
                w1, w2, w_u, tr1, tr2 = twofold_weights(X_u, x0, args.n_label, seed_u + 17, cfg)
                tuning_seconds = float(time.perf_counter() - started)
                for lrep in range(args.label_reps):
                    lab_idx = np.random.default_rng(seed_u + 5000000 + lrep).choice(
                        label_pool, size=args.n_label, replace=False
                    )
                    X_l, Y_l, f_l = X_std[lab_idx], Y_s[lab_idx], f_s[lab_idx]
                    estimation_start = time.perf_counter()
                    w_l = 0.5 * (w1(X_l) + w2(X_l))
                    result = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u)
                    estimation_seconds = float(time.perf_counter() - estimation_start)
                    rows.append(result_record(
                        result, n_unlab, urep, lrep, theta0, tr1, tr2,
                        tuning_seconds, estimation_seconds, args.label_reps,
                    ))
            print(f"[done] N={n_unlab}", flush=True)

    out_dir = ensure_dir(args.output_dir)
    write_run_manifest(out_dir / "run_manifest.json", args, extra={"backend_resolved": backend})
    replicate = pd.DataFrame(rows)
    summary = summarize(replicate)
    summary.to_csv(out_dir / "summary_by_N.csv", index=False)
    if args.save_replicates or args.paired_nested:
        replicate.to_csv(out_dir / "replicates.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
