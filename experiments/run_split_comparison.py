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

from ppci_condmean.data import (
    generate_simulation_labeled,
    generate_simulation_unlabeled,
    m_true_simulation,
    standardize_unif01,
)
from ppci_condmean.estimator import fit_ppci_mean
from ppci_condmean.gpu import configure_backend
from ppci_condmean.joint_tuning import JointTuningConfig
from ppci_condmean.utils import ensure_dir, write_run_manifest


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in str(value).split(",") if x.strip()]


def synchronize(backend: str) -> None:
    if backend == "torch":
        import torch

        torch.cuda.synchronize()


def reset_gpu_peak(backend: str) -> None:
    if backend == "torch":
        import torch

        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb(backend: str) -> float:
    if backend != "torch":
        return np.nan
    import torch

    return float(torch.cuda.max_memory_allocated() / (1024.0**2))


def result_row(result, split: str, x0_index: int, rep: int, theta0: float, elapsed: float, peak_mb: float) -> dict:
    return {
        "split": split,
        "method": result.method,
        "x0_index": x0_index,
        "rep": rep,
        "theta0": theta0,
        "theta_hat": result.theta_hat,
        "error": result.theta_hat - theta0,
        "covered": result.ci_low <= theta0 <= result.ci_high,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "se": result.se,
        "width": result.ci_high - result.ci_low,
        "sigma2_Y": result.sigma2_Y,
        "sigma2_Y_minus_f": result.sigma2_Y_minus_f,
        "sigma2_f": result.sigma2_f,
        "procedure_seconds": elapsed,
        "gpu_peak_mb": peak_mb,
        "h_mean": result.h_mean,
        "lambda_mean": result.lambda_mean,
        "h_factor": result.h_factor,
        "lambda_factor": result.lambda_factor,
        "op_score": result.op_score,
        "loc_score": result.loc_score,
        "fallback": "fallback" in str(result.tuning_status),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, method), group in df.groupby(["split", "method"], dropna=False):
        error = group["error"].to_numpy(float)
        rows.append({
            "split": split,
            "method": method,
            "n_reps": len(group),
            "coverage": group["covered"].mean(),
            "bias": error.mean(),
            "abs_bias": abs(error.mean()),
            "rmse": np.sqrt(np.mean(error**2)),
            "emp_sd": group["theta_hat"].std(ddof=1),
            "se_mean": group["se"].mean(),
            "width": group["width"].mean(),
            "sigma2_Y_mean": group["sigma2_Y"].mean(),
            "sigma2_Y_minus_f_mean": group["sigma2_Y_minus_f"].mean(),
            "sigma2_f_mean": group["sigma2_f"].mean(),
            "procedure_seconds_mean": group["procedure_seconds"].mean(),
            "procedure_seconds_median": group["procedure_seconds"].median(),
            "gpu_peak_mb_mean": group["gpu_peak_mb"].mean(),
            "h_mean": group["h_mean"].mean(),
            "lambda_mean": group["lambda_mean"].mean(),
            "h_factor_mean": group["h_factor"].mean(),
            "lambda_factor_mean": group["lambda_factor"].mean(),
            "op_score_mean": group["op_score"].mean(),
            "loc_score_mean": group["loc_score"].mean(),
            "fallback_rate": group["fallback"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired two-fold versus no-split PPCI comparison.")
    parser.add_argument("--output-dir", default="runs/split_comparison")
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--n-label", type=int, default=200)
    parser.add_argument("--n-unlab", type=int, default=10000)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--x0-values", default="0.70,0.75,0.80,0.825,0.85")
    parser.add_argument("--x0-index-offset", type=int, default=0, help="Global x0 index of the first value, for reproducible sharding.")
    parser.add_argument("--sigma-eps", type=float, default=1.0)
    parser.add_argument("--predictor-quality", type=float, default=0.9)
    parser.add_argument("--h-factors", default="0.8,1.0,1.15,1.2")
    parser.add_argument("--lambda-factor-min", type=float, default=0.02)
    parser.add_argument("--lambda-factor-max", type=float, default=60.0)
    parser.add_argument("--lambda-grid-size", type=int, default=35)
    parser.add_argument("--tau-op", type=float, default=12.0)
    parser.add_argument("--tau-loc", type=float, default=4.0)
    parser.add_argument("--c-bias", type=float, default=0.10)
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "torch", "gpu", "cuda"])
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_label = min(args.n_label, 60)
        args.n_unlab = min(args.n_unlab, 200)
        args.reps = 2
        args.x0_values = "0.75"
        args.lambda_grid_size = min(args.lambda_grid_size, 9)

    backend = configure_backend(args.backend, args.gpu_id)
    cfg = JointTuningConfig(
        h_grid_mode="median_grid",
        h_factors=tuple(parse_floats(args.h_factors)),
        lambda_factor_min=args.lambda_factor_min,
        lambda_factor_max=args.lambda_factor_max,
        lambda_grid_size=args.lambda_grid_size,
        lambda_grid_mode="shrinking",
        tau_op=args.tau_op,
        tau_loc=args.tau_loc,
        bias_screen="p1_label",
        c_bias=args.c_bias,
        constraint_fallback="least_violation",
        backend=backend,
    )
    print(f"[backend] {backend}", flush=True)

    rows = []
    for local_index, value in enumerate(parse_floats(args.x0_values)):
        x0_index = args.x0_index_offset + local_index
        x0_raw = np.repeat(value, 3)
        x0 = standardize_unif01(x0_raw.reshape(1, -1))[0]
        theta0 = float(m_true_simulation(x0_raw.reshape(1, -1))[0])
        for rep in range(args.reps):
            rng_l = np.random.default_rng(args.seed + 100000 * x0_index + 2 * rep)
            rng_u = np.random.default_rng(args.seed + 100000 * x0_index + 2 * rep + 1)
            _, X_l, Y_l, f_l = generate_simulation_labeled(
                rng_l, args.n_label, args.sigma_eps, args.predictor_quality
            )
            _, X_u, f_u = generate_simulation_unlabeled(
                rng_u, args.n_unlab, args.predictor_quality
            )
            for split in ("twofold", "nosplit"):
                reset_gpu_peak(backend)
                synchronize(backend)
                started = time.perf_counter()
                results = fit_ppci_mean(
                    X_l, Y_l, f_l, X_u, f_u, x0,
                    split=split, seed=args.seed + 97 * rep, tuning_cfg=cfg,
                )
                synchronize(backend)
                elapsed = float(time.perf_counter() - started)
                peak_mb = gpu_peak_mb(backend)
                for result in results[:2]:
                    rows.append(result_row(result, split, x0_index, rep, theta0, elapsed, peak_mb))
        print(f"[done] x0_index={x0_index}, x0={value:g}", flush=True)

    out_dir = ensure_dir(args.output_dir)
    write_run_manifest(out_dir / "run_manifest.json", args, extra={"backend_resolved": backend})
    replicate = pd.DataFrame(rows)
    replicate.to_csv(out_dir / "replicates.csv", index=False)
    summary = summarize(replicate)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
