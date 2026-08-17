#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split-comparison replicate shards.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()
    shards = [Path(shard) for shard in args.shards]
    manifests = [json.loads((shard / "run_manifest.json").read_text()) for shard in shards]
    source_hashes = {manifest.get("source_sha256") for manifest in manifests}
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError(f"Split-comparison shards have inconsistent source hashes: {source_hashes}")
    paths = [shard / "replicates.csv" for shard in shards]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete shards: {missing}")
    replicate = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if replicate.duplicated(["split", "method", "x0_index", "rep"]).any():
        raise ValueError("Split-comparison shard assignments overlap.")

    rows = []
    for (split, method), group in replicate.groupby(["split", "method"], dropna=False):
        error = group["error"].to_numpy(float)
        rows.append({
            "split": split,
            "method": method,
            "n_x0": group["x0_index"].nunique(),
            "n_reps": len(group),
            "coverage": group["covered"].mean(),
            "bias": error.mean(),
            "abs_bias": abs(error.mean()),
            "rmse": np.sqrt(np.mean(error**2)),
            "emp_sd": group["theta_hat"].std(ddof=1),
            "se_mean": group["se"].mean(),
            "emp_sd_over_se_mean": group["theta_hat"].std(ddof=1) / group["se"].mean(),
            "width": group["width"].mean(),
            "sigma2_Y_mean": group["sigma2_Y"].mean(),
            "sigma2_Y_minus_f_mean": group["sigma2_Y_minus_f"].mean(),
            "sigma2_f_mean": group["sigma2_f"].mean(),
            "procedure_seconds_mean": group["procedure_seconds"].mean(),
            "procedure_seconds_median": group["procedure_seconds"].median(),
            "procedure_seconds_p90": group["procedure_seconds"].quantile(0.9),
            "gpu_peak_mb_mean": group["gpu_peak_mb"].mean(),
            "h_mean": group["h_mean"].mean(),
            "lambda_mean": group["lambda_mean"].mean(),
            "h_factor_mean": group["h_factor"].mean(),
            "lambda_factor_mean": group["lambda_factor"].mean(),
            "op_score_mean": group["op_score"].mean(),
            "loc_score_mean": group["loc_score"].mean(),
            "fallback_rate": group["fallback"].mean(),
        })
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shard_manifests.json").write_text(
        json.dumps(
            [{"shard": str(shard), "manifest": manifest} for shard, manifest in zip(shards, manifests)],
            indent=2,
        ),
        encoding="utf-8",
    )
    replicate.to_csv(out_dir / "replicates.csv", index=False)
    summary = pd.DataFrame(rows).sort_values(["split", "method"])
    summary.to_csv(out_dir / "summary.csv", index=False)
    paired_rows = []
    index = ["method", "x0_index", "rep"]
    paired = replicate.pivot(index=index, columns="split")
    if {"twofold", "nosplit"}.issubset(set(paired.columns.get_level_values(1))):
        for method, block in paired.groupby(level="method"):
            row = {"method": method, "n_pairs": int(len(block))}
            for column in ("theta_hat", "se", "width", "covered"):
                difference = (
                    block[(column, "nosplit")].astype(float)
                    - block[(column, "twofold")].astype(float)
                )
                row[f"{column}_difference_mean_nosplit_minus_twofold"] = float(difference.mean())
                row[f"{column}_absolute_difference_mean"] = float(difference.abs().mean())
                row[f"{column}_absolute_difference_max"] = float(difference.abs().max())
            row["coverage_discordant_pairs"] = int(
                (block[("covered", "nosplit")] != block[("covered", "twofold")]).sum()
            )
            row["procedure_time_ratio_nosplit_over_twofold"] = float(
                block[("procedure_seconds", "nosplit")].mean()
                / block[("procedure_seconds", "twofold")].mean()
            )
            row["gpu_peak_ratio_nosplit_over_twofold"] = float(
                block[("gpu_peak_mb", "nosplit")].mean()
                / block[("gpu_peak_mb", "twofold")].mean()
            )
            paired_rows.append(row)
    pd.DataFrame(paired_rows).sort_values("method").to_csv(
        out_dir / "paired_differences.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
