#!/usr/bin/env python3
"""Rescore real-data intervals under alternative full-data reference smoothers.

The PPCI/LO/PPI estimates and intervals are held fixed. Only the finite-
population reference target used to compute coverage, bias, and RMSE changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ppci_condmean.data import (
    census_sex_subset,
    load_blogfeedback_raw,
    load_census_npz,
)
from ppci_condmean.utils import standardize_apply


REFERENCE_ORDER = [
    "matern52_0.8h0",
    "matern52_h0",
    "matern52_1.2h0",
    "gaussian_ess_matched",
]


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def kernel_weights(distances: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    h = max(float(bandwidth), 1e-12)
    if kernel == "matern52":
        t = np.sqrt(5.0) * d / h
        return (1.0 + t + t * t / 3.0) * np.exp(-t)
    if kernel == "gaussian":
        return np.exp(-0.5 * (d / h) ** 2)
    raise ValueError(f"Unknown kernel: {kernel}")


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w) ** 2 / max(float(np.sum(w * w)), 1e-30))


def weighted_mean(y: np.ndarray, weights: np.ndarray) -> float:
    den = float(np.sum(weights))
    if den <= 1e-14:
        raise ValueError("Reference smoother has zero total weight.")
    return float(np.dot(np.asarray(y, dtype=float), weights) / den)


def ess_matched_gaussian_bandwidth(distances: np.ndarray, target_ess: float, h0: float) -> float:
    low = max(float(h0) * 1e-4, 1e-12)
    high = max(float(h0), low * 2.0)
    while effective_sample_size(kernel_weights(distances, high, "gaussian")) < target_ess:
        high *= 2.0
        if high > max(float(h0), 1.0) * 1e6:
            raise RuntimeError("Could not bracket ESS-matched Gaussian bandwidth.")
    for _ in range(80):
        mid = 0.5 * (low + high)
        if effective_sample_size(kernel_weights(distances, mid, "gaussian")) < target_ess:
            low = mid
        else:
            high = mid
    return float(high)


def reference_rows(
    X: np.ndarray,
    y: np.ndarray,
    x0: np.ndarray,
    target_context: dict,
) -> list[dict]:
    distances = np.linalg.norm(np.asarray(X, dtype=float) - np.asarray(x0, dtype=float), axis=1)
    h0 = max(float(np.median(distances)), 1e-12)
    baseline_weights = kernel_weights(distances, h0, "matern52")
    baseline_ess = effective_sample_size(baseline_weights)
    gaussian_h = ess_matched_gaussian_bandwidth(distances, baseline_ess, h0)
    specs = [
        ("matern52_0.8h0", "matern52", 0.8 * h0),
        ("matern52_h0", "matern52", h0),
        ("matern52_1.2h0", "matern52", 1.2 * h0),
        ("gaussian_ess_matched", "gaussian", gaussian_h),
    ]
    rows = []
    for reference, kernel, bandwidth in specs:
        weights = kernel_weights(distances, bandwidth, kernel)
        rows.append(
            {
                **target_context,
                "reference": reference,
                "kernel": kernel,
                "bandwidth": float(bandwidth),
                "bandwidth_over_h0": float(bandwidth / h0),
                "ess": effective_sample_size(weights),
                "theta0": weighted_mean(y, weights),
            }
        )
    return rows


def income_reference_table(data_path: Path, sexes: list[int], ages: list[int]) -> pd.DataFrame:
    X, y, f = load_census_npz(data_path)
    rows = []
    for sex in sexes:
        X_raw, X_std, y_sex, _, mean, std = census_sex_subset(X, y, f, sex)
        for age in ages:
            x0_raw = np.array([[float(age), float(sex)]])
            x0 = standardize_apply(x0_raw, mean, std)[0]
            rows.extend(
                reference_rows(
                    X_std,
                    y_sex,
                    x0,
                    {"dataset": "income", "sex": int(sex), "age": int(age)},
                )
            )
    return pd.DataFrame(rows)


def blog_reference_table(
    data_path: Path,
    seed: int,
    n_x0: int,
    x0_indices: list[int],
    ppci_fraction: float,
) -> pd.DataFrame:
    X, y, _, _ = load_blogfeedback_raw(data_path)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    idx_x0 = rng.choice(idx_all, size=min(n_x0, len(y) // 10), replace=False)
    keep = np.ones(len(y), dtype=bool)
    keep[idx_x0] = False
    idx_rem = idx_all[keep]
    _, X_ppci, _, y_ppci = train_test_split(
        X[idx_rem],
        y[idx_rem],
        test_size=ppci_fraction,
        random_state=seed,
    )
    rows = []
    for x0_index in x0_indices:
        if not 0 <= x0_index < len(idx_x0):
            raise ValueError(f"BlogFeedback x0 index {x0_index} is outside [0, {len(idx_x0) - 1}].")
        rows.extend(
            reference_rows(
                X_ppci,
                y_ppci,
                X[idx_x0[x0_index]],
                {"dataset": "blogfeedback", "x0_index": int(x0_index)},
            )
        )
    return pd.DataFrame(rows)


def read_replicates(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["replicate_source"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    required = {"method", "theta_hat", "ci_low", "ci_high"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Replicate files are missing columns: {sorted(missing)}")
    return out


def target_keys(dataset: str) -> list[str]:
    return ["sex", "age"] if dataset == "income" else ["x0_index"]


def method_family(method: str) -> str:
    method = str(method)
    if method.startswith("GH_P1") or method == "PPCI":
        return "PPCI"
    if method.startswith("LO_") or method == "LO":
        return "LO"
    if method.startswith("PPI"):
        return "PPI"
    return method


def summarize_target_shifts(
    references: pd.DataFrame,
    replicates: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = target_keys(dataset)
    baseline = references[references["reference"] == "matern52_h0"][keys + ["theta0"]].rename(
        columns={"theta0": "theta0_baseline"}
    )
    shifts = references.merge(baseline, on=keys, validate="many_to_one")
    shifts["shift"] = shifts["theta0"] - shifts["theta0_baseline"]
    shifts["abs_shift"] = shifts["shift"].abs()
    if not replicates.empty:
        width = replicates[replicates["method"].map(method_family) == "PPCI"].copy()
        if width.empty:
            raise ValueError("Could not identify PPCI rows in the replicate files.")
        width["width"] = width["ci_high"] - width["ci_low"]
        width = width.groupby(keys, as_index=False)["width"].mean().rename(columns={"width": "ppci_width"})
        shifts = shifts.merge(width, on=keys, how="left", validate="many_to_one")
        shifts["abs_shift_over_ppci_width"] = shifts["abs_shift"] / shifts["ppci_width"]
    group = shifts.groupby("reference", sort=False, observed=True)
    summary = group.agg(
        n_targets=("theta0", "size"),
        mean_signed_shift=("shift", "mean"),
        mean_abs_shift=("abs_shift", "mean"),
        max_abs_shift=("abs_shift", "max"),
        mean_ess=("ess", "mean"),
        mean_bandwidth_over_h0=("bandwidth_over_h0", "mean"),
    ).reset_index()
    if "abs_shift_over_ppci_width" in shifts:
        scaled = group["abs_shift_over_ppci_width"].agg(["mean", "max"]).reset_index()
        scaled = scaled.rename(
            columns={"mean": "mean_abs_shift_over_ppci_width", "max": "max_abs_shift_over_ppci_width"}
        )
        summary = summary.merge(scaled, on="reference", validate="one_to_one")
    return shifts, summary


def rescore(
    references: pd.DataFrame,
    replicates: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if replicates.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = target_keys(dataset)
    target_rows = []
    cluster_col = next(
        (c for c in ["unlab_rep", "unlab_rep_index", "unlab_draw", "unlabeled_rep"] if c in replicates),
        None,
    )
    for reference in REFERENCE_ORDER:
        theta = references[references["reference"] == reference][keys + ["theta0"]]
        scored = replicates.merge(theta, on=keys, suffixes=("_old", ""), validate="many_to_one")
        scored["covered_rescored"] = (scored["ci_low"] <= scored["theta0"]) & (
            scored["theta0"] <= scored["ci_high"]
        )
        scored["error_rescored"] = scored["theta_hat"] - scored["theta0"]
        scored["width_rescored"] = scored["ci_high"] - scored["ci_low"]
        for key, group in scored.groupby(keys + ["method"], dropna=False):
            key_values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(keys + ["method"], key_values))
            errors = group["error_rescored"].to_numpy(float)
            row.update(
                {
                    "dataset": dataset,
                    "reference": reference,
                    "method_family": method_family(row["method"]),
                    "n_reps": int(len(group)),
                    "coverage": float(group["covered_rescored"].mean()),
                    "bias": float(np.mean(errors)),
                    "abs_bias": float(abs(np.mean(errors))),
                    "rmse": float(np.sqrt(np.mean(errors * errors))),
                    "width": float(group["width_rescored"].mean()),
                    "theta0": float(group["theta0"].iloc[0]),
                    "theta_hat_mean": float(group["theta_hat"].mean()),
                }
            )
            if cluster_col is not None:
                cluster_coverage = group.groupby(cluster_col)["covered_rescored"].mean()
                row["n_unlab_clusters"] = int(cluster_coverage.size)
                row["coverage_cluster_se"] = (
                    float(cluster_coverage.std(ddof=1) / np.sqrt(cluster_coverage.size))
                    if cluster_coverage.size > 1
                    else np.nan
                )
            target_rows.append(row)
    by_target = pd.DataFrame(target_rows)
    aggregate_rows = []
    for (reference, family), group in by_target.groupby(["reference", "method_family"], sort=False):
        row = {
            "dataset": dataset,
            "reference": reference,
            "method": family,
            "internal_methods": ",".join(sorted(group["method"].astype(str).unique())),
            "n_targets": int(len(group)),
            "coverage_mean": float(group["coverage"].mean()),
            "coverage_min": float(group["coverage"].min()),
            "coverage_max": float(group["coverage"].max()),
            "bias_mean": float(group["bias"].mean()),
            "abs_bias_mean": float(group["abs_bias"].mean()),
            "rmse_mean": float(group["rmse"].mean()),
            "width_mean": float(group["width"].mean()),
        }
        if "coverage_cluster_se" in group:
            row["coverage_cluster_se_mean"] = float(group["coverage_cluster_se"].mean())
            row["coverage_cluster_se_max"] = float(group["coverage_cluster_se"].max())
        aggregate_rows.append(row)
    return by_target, pd.DataFrame(aggregate_rows)


def rescore_formal_summary(
    references: pd.DataFrame,
    summary: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = target_keys(dataset)
    required = set(keys + ["method", "n_reps", "theta_hat_mean", "emp_sd", "width"])
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Formal summary is missing columns: {sorted(missing)}")
    rows = []
    for reference in REFERENCE_ORDER:
        theta = references[references["reference"] == reference][keys + ["theta0"]]
        scored = summary.merge(theta, on=keys, validate="many_to_one")
        scored["dataset"] = dataset
        scored["reference"] = reference
        scored["method_family"] = scored["method"].map(method_family)
        scored["bias_rescored"] = scored["theta_hat_mean"] - scored["theta0"]
        n_reps = scored["n_reps"].to_numpy(float)
        variance_mle = scored["emp_sd"].to_numpy(float) ** 2 * np.maximum(n_reps - 1.0, 0.0) / np.maximum(
            n_reps, 1.0
        )
        scored["rmse_rescored"] = np.sqrt(variance_mle + scored["bias_rescored"].to_numpy(float) ** 2)
        rows.append(scored)
    by_target = pd.concat(rows, ignore_index=True)
    aggregate_rows = []
    for (reference, family), group in by_target.groupby(["reference", "method_family"], sort=False):
        aggregate_rows.append(
            {
                "dataset": dataset,
                "reference": reference,
                "method": family,
                "internal_methods": ",".join(sorted(group["method"].astype(str).unique())),
                "n_targets": int(len(group)),
                "n_reps_per_target_min": int(group["n_reps"].min()),
                "bias_mean": float(group["bias_rescored"].mean()),
                "abs_bias_mean": float(group["bias_rescored"].abs().mean()),
                "rmse_mean": float(group["rmse_rescored"].mean()),
                "width_mean": float(group["width"].mean()),
            }
        )
    return by_target, pd.DataFrame(aggregate_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore fixed real-data intervals under alternative reference smoothers."
    )
    parser.add_argument("--dataset", choices=["income", "blogfeedback"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicate-files", nargs="*", default=[])
    parser.add_argument("--formal-summary-file", default="")
    parser.add_argument("--income-sexes", default="1,2")
    parser.add_argument("--income-ages", default="70,80,90,100")
    parser.add_argument("--blog-x0-indices", default="0,7,14,21,28,35,42,49")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-blog-x0", type=int, default=50)
    parser.add_argument("--ppci-fraction", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_paths = [Path(p) for p in args.replicate_files]
    replicates = read_replicates(replicate_paths)

    if args.dataset == "income":
        references = income_reference_table(
            Path(args.data),
            parse_ints(args.income_sexes),
            parse_ints(args.income_ages),
        )
    else:
        references = blog_reference_table(
            Path(args.data),
            args.seed,
            args.n_blog_x0,
            parse_ints(args.blog_x0_indices),
            args.ppci_fraction,
        )

    references["reference"] = pd.Categorical(
        references["reference"], categories=REFERENCE_ORDER, ordered=True
    )
    references = references.sort_values(target_keys(args.dataset) + ["reference"])
    references.to_csv(output_dir / "reference_targets.csv", index=False)

    shifts, shift_summary = summarize_target_shifts(references, replicates, args.dataset)
    shifts.to_csv(output_dir / "target_shifts.csv", index=False)
    shift_summary.to_csv(output_dir / "target_shift_summary.csv", index=False)

    by_target, aggregate = rescore(references, replicates, args.dataset)
    if not by_target.empty:
        by_target.to_csv(output_dir / "rescored_by_target.csv", index=False)
        aggregate.to_csv(output_dir / "rescored_aggregate.csv", index=False)

    if str(args.formal_summary_file).strip():
        formal_summary = pd.read_csv(args.formal_summary_file)
        formal_by_target, formal_aggregate = rescore_formal_summary(references, formal_summary, args.dataset)
        formal_by_target.to_csv(output_dir / "formal_rescored_by_target.csv", index=False)
        formal_aggregate.to_csv(output_dir / "formal_rescored_aggregate.csv", index=False)

    config = vars(args).copy()
    config["replicate_files"] = [str(p) for p in replicate_paths]
    config["reference_order"] = REFERENCE_ORDER
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print("\nReference-target shifts")
    print(shift_summary.to_string(index=False))
    if not aggregate.empty:
        print("\nRescored repeated-sampling metrics")
        print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
