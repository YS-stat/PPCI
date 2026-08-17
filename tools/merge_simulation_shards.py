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


def first_mean_if_present(group: pd.DataFrame, *columns: str) -> float:
    for column in columns:
        value = mean_if_present(group, column)
        if np.isfinite(value):
            return value
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compact Simulation shard outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()

    shard_dirs = [Path(path) for path in args.shards]
    manifests = [json.loads((path / "run_manifest.json").read_text()) for path in shard_dirs]
    source_hashes = {manifest.get("source_sha256") for manifest in manifests}
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError(f"Simulation shards have inconsistent source hashes: {source_hashes}")
    missing = [str(path) for path in shard_dirs if not (path / "summary_by_x0.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete shards: {missing}")

    by_x0 = pd.concat(
        [pd.read_csv(path / "summary_by_x0.csv") for path in shard_dirs], ignore_index=True
    ).sort_values(["setting", "method", "x0_index"])
    duplicated = by_x0.duplicated(["setting", "method", "x0_index"])
    if duplicated.any():
        raise ValueError("Shard target assignments overlap.")

    rows = []
    for (setting, method), group in by_x0.groupby(["setting", "method"], dropna=False):
        rows.append({
            "setting": setting,
            "method": method,
            "n_x0": group["x0_index"].nunique(),
            "n_reps": int(group["n_reps"].sum()),
            "coverage_mean": group["coverage"].mean(),
            "coverage_min_x0": group["coverage"].min(),
            "coverage_max_x0": group["coverage"].max(),
            "bias_mean": group["bias"].mean(),
            "abs_bias_mean": group["bias"].abs().mean(),
            "rmse_mean": group["rmse"].mean(),
            "rmse_max_x0": group["rmse"].max(),
            "se_mean": group["se_mean"].mean(),
            "emp_sd_mean": group["emp_sd"].mean(),
            "emp_sd_over_se_mean": group["emp_sd"].mean() / group["se_mean"].mean() if group["se_mean"].mean() > 0 else np.nan,
            "width_mean": group["width"].mean(),
            "omega_mean": mean_if_present(group, "omega_mean"),
            "omega_clipped_rate": mean_if_present(group, "omega_clipped_rate"),
            "omega_sd_mean": mean_if_present(group, "omega_sd_mean"),
            "omega_min_mean": mean_if_present(group, "omega_min_mean"),
            "omega_max_mean": mean_if_present(group, "omega_max_mean"),
            "sigma2_Y_mean": mean_if_present(group, "sigma2_Y_mean"),
            "sigma2_Y_minus_f_mean": mean_if_present(group, "sigma2_Y_minus_f_mean"),
            "sigma2_f_mean": mean_if_present(group, "sigma2_f_mean"),
            "tuning_seconds_mean_per_unlab_draw": first_mean_if_present(
                group, "tuning_seconds_mean_per_unlab_draw", "tuning_seconds_mean"
            ),
            "estimation_seconds_mean": mean_if_present(group, "estimation_seconds_mean"),
            "amortized_procedure_seconds_mean": mean_if_present(group, "amortized_procedure_seconds_mean"),
        })
    aggregate = pd.DataFrame(rows).sort_values(["setting", "method"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_x0.to_csv(out_dir / "summary_by_x0.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_equal_x0.csv", index=False)
    (out_dir / "shard_manifests.json").write_text(
        json.dumps(
            [{"shard": str(path), "manifest": manifest} for path, manifest in zip(shard_dirs, manifests)],
            indent=2,
        ),
        encoding="utf-8",
    )

    for filename in ("tuning_decisions.csv", "tuning_diagnostics.csv"):
        paths = [path / filename for path in shard_dirs]
        if all(path.exists() for path in paths):
            pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(
                out_dir / filename, index=False
            )
    print(f"Merged {len(shard_dirs)} shards and {by_x0['x0_index'].nunique()} targets into {out_dir}")


if __name__ == "__main__":
    main()
