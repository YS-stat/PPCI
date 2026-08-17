#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_if_present(group: pd.DataFrame, column: str) -> float:
    if column not in group or not group[column].notna().any():
        return np.nan
    return float(group[column].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint BlogFeedback x0 shards.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-x0", type=int, default=50)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()

    shards = [Path(value) for value in args.shards]
    summary_paths = [path / "summary_by_x0.csv" for path in shards]
    missing = [str(path) for path in summary_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete BlogFeedback shards: {missing}")

    summary = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    keys = ["x0_index", "method"]
    if summary.duplicated(keys).any():
        raise ValueError("BlogFeedback shard target assignments overlap.")
    n_x0 = int(summary["x0_index"].nunique())
    if args.expected_x0 > 0 and n_x0 != args.expected_x0:
        raise ValueError(f"Expected {args.expected_x0} x0 values, found {n_x0}.")
    summary = summary.sort_values(keys)

    rows = []
    for method, group in summary.groupby("method", dropna=False):
        row = {
            "method": method,
            "n_x0": int(group["x0_index"].nunique()),
            "coverage_mean": float(group["coverage"].mean()),
            "coverage_min_x0": float(group["coverage"].min()),
            "coverage_max_x0": float(group["coverage"].max()),
            "coverage_cluster_se_mean": mean_if_present(group, "coverage_cluster_se"),
            "coverage_cluster_se_max": float(group["coverage_cluster_se"].max()) if "coverage_cluster_se" in group else np.nan,
            "bias_mean": float(group["bias"].mean()),
            "abs_bias_mean": mean_if_present(group, "abs_bias"),
            "rmse_mean": float(group["rmse"].mean()),
            "rmse_max_x0": float(group["rmse"].max()),
            "se_mean": mean_if_present(group, "se_mean"),
            "emp_sd_mean": mean_if_present(group, "emp_sd"),
            "width_mean": float(group["width"].mean()),
            "sigma2_Y_mean": mean_if_present(group, "sigma2_Y_mean"),
            "sigma2_Y_minus_f_mean": mean_if_present(group, "sigma2_Y_minus_f_mean"),
            "sigma2_f_mean": mean_if_present(group, "sigma2_f_mean"),
            "fallback_rate_mean": mean_if_present(group, "fallback_rate"),
            "h_factor_mean": mean_if_present(group, "h_factor_mean"),
            "lambda_factor_mean": mean_if_present(group, "lambda_factor_mean"),
            "bias_score_mean": mean_if_present(group, "bias_score_mean"),
            "bias_budget_mean": mean_if_present(group, "bias_budget_mean"),
            "nw_corr_mean": mean_if_present(group, "nw_corr_mean"),
            "nw_relative_difference_mean": mean_if_present(group, "nw_relative_difference_mean"),
            "negative_weight_fraction_mean": mean_if_present(group, "negative_weight_fraction_mean"),
            "M_lambda_over_eigmax_mean": mean_if_present(group, "M_lambda_over_eigmax_mean"),
            "tuning_seconds_mean_per_unlab_draw": mean_if_present(group, "tuning_seconds_mean_per_unlab_draw"),
            "estimation_seconds_mean": mean_if_present(group, "estimation_seconds_mean"),
            "amortized_procedure_seconds_mean": mean_if_present(group, "amortized_procedure_seconds_mean"),
        }
        for column in group.columns:
            if column.startswith("width_ratio_vs_") or column.startswith("rmse_ratio_vs_"):
                row[f"{column}_mean"] = mean_if_present(group, column)
        rows.append(row)
    aggregate = pd.DataFrame(rows).sort_values("method")

    tuning_paths = [path / "tuning_decisions.csv" for path in shards]
    tuning = pd.concat([pd.read_csv(path) for path in tuning_paths], ignore_index=True)
    tuning_keys = ["x0_index", "method", "unlab_rep"]
    if tuning.duplicated(tuning_keys).any():
        raise ValueError("BlogFeedback tuning decisions overlap across shards.")
    tuning = tuning.sort_values(tuning_keys)
    tuning_rows = []
    for method, group in tuning.groupby("method", dropna=False):
        row = {"method": method, "tuning_decisions": int(len(group))}
        for column in group.select_dtypes(include="number").columns:
            if column not in {"x0_index", "unlab_rep"}:
                row[f"{column}_mean"] = float(group[column].mean())
        if "fallback" in group:
            row["fallback_rate"] = float(group["fallback"].mean())
        tuning_rows.append(row)

    manifests = []
    for shard in shards:
        path = shard / "run_manifest.json"
        manifests.append({"shard": str(shard), "manifest": json.loads(path.read_text())})
    source_hashes = {item["manifest"].get("source_sha256") for item in manifests}
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError(f"BlogFeedback shards have inconsistent source hashes: {source_hashes}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary_by_x0.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    tuning.to_csv(output / "tuning_decisions.csv", index=False)
    pd.DataFrame(tuning_rows).sort_values("method").to_csv(output / "tuning_by_method.csv", index=False)
    (output / "shard_manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
