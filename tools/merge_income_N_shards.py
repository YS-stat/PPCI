#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint Income unlabeled-N sweep shards.")
    parser.add_argument("--output", required=True)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()
    shards = [Path(shard) for shard in args.shards]
    manifests = [json.loads((shard / "run_manifest.json").read_text()) for shard in shards]
    source_hashes = {manifest.get("source_sha256") for manifest in manifests}
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError(f"Income-N shards have inconsistent source hashes: {source_hashes}")
    replicate_paths = [path / "replicates.csv" for path in shards]
    if all(path.exists() for path in replicate_paths):
        replicate = pd.concat([pd.read_csv(path) for path in replicate_paths], ignore_index=True)
        keys = ["n_unlab", "unlab_rep", "label_rep"]
        if replicate.duplicated(keys).any():
            raise ValueError("Income-N replicate shards overlap.")
        summary = summarize(replicate)
    else:
        paths = [path / "summary_by_N.csv" for path in shards]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete shards: {missing}")
        summary = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        if summary["n_unlab"].duplicated().any():
            raise ValueError("Income-N summary shards overlap; retain replicates for cluster-sharded runs.")
        summary = summary.sort_values("n_unlab")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    if "replicate" in locals():
        replicate.sort_values(["n_unlab", "unlab_rep", "label_rep"]).to_csv(
            output.with_name("replicates.csv"), index=False
        )
    output.with_name("shard_manifests.json").write_text(
        json.dumps(
            [{"shard": str(shard), "manifest": manifest} for shard, manifest in zip(shards, manifests)],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
