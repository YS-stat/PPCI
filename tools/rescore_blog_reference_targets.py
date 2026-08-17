#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ppci_condmean.data import load_blogfeedback_raw, nw_oracle_mean


def parse_floats(value: str) -> list[float]:
    return [float(x) for x in str(value).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore BlogFeedback intervals against reference-target sensitivities.")
    parser.add_argument("replicates")
    parser.add_argument("--data", default="data/blogfeedback/blogfeedback.zip")
    parser.add_argument("--output", default="reference_target_sensitivity.csv")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-x0", type=int, default=50)
    parser.add_argument("--ppci-fraction", type=float, default=0.3)
    parser.add_argument("--max-raw-rows", type=int, default=0)
    parser.add_argument("--bandwidth-multipliers", default="0.75,1.0,1.25")
    args = parser.parse_args()

    replicate = pd.read_csv(args.replicates)
    X, Y, _, _ = load_blogfeedback_raw(
        args.data, nrows=None if args.max_raw_rows <= 0 else args.max_raw_rows
    )
    rng = np.random.default_rng(args.seed)
    idx_all = np.arange(len(Y))
    idx_x0 = rng.choice(idx_all, size=min(args.n_x0, len(Y) // 10), replace=False)
    idx_remaining = idx_all[~np.isin(idx_all, idx_x0)]
    idx_reference, _ = train_test_split(
        idx_remaining, test_size=args.ppci_fraction, random_state=args.seed
    )

    targets = []
    for x0_index, data_index in enumerate(idx_x0):
        x0 = X[data_index]
        for reference_name, indices in (("full", idx_all), ("disjoint_train", idx_reference)):
            distances = np.linalg.norm(X[indices] - x0.reshape(1, -1), axis=1)
            h0 = max(float(np.median(distances)), 1e-8)
            for multiplier in parse_floats(args.bandwidth_multipliers):
                theta = nw_oracle_mean(X[indices], Y[indices], x0, h=multiplier * h0)
                targets.append({
                    "x0_index": x0_index,
                    "reference": reference_name,
                    "bandwidth_multiplier": multiplier,
                    "theta_reference": theta,
                })
    target_table = pd.DataFrame(targets)
    scored = replicate.merge(target_table, on="x0_index", how="inner", validate="many_to_many")
    scored["error_reference"] = scored["theta_hat"] - scored["theta_reference"]
    scored["covered_reference"] = (
        (scored["ci_low"] <= scored["theta_reference"])
        & (scored["theta_reference"] <= scored["ci_high"])
    )

    rows = []
    for key, group in scored.groupby(["reference", "bandwidth_multiplier", "method"]):
        error = group["error_reference"].to_numpy(float)
        by_x0 = group.groupby("x0_index").agg(
            coverage=("covered_reference", "mean"),
            bias=("error_reference", "mean"),
            rmse=("error_reference", lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))),
        )
        rows.append({
            "reference": key[0],
            "bandwidth_multiplier": key[1],
            "method": key[2],
            "n_x0": by_x0.shape[0],
            "coverage_mean": by_x0["coverage"].mean(),
            "coverage_min_x0": by_x0["coverage"].min(),
            "coverage_max_x0": by_x0["coverage"].max(),
            "bias_mean": by_x0["bias"].mean(),
            "abs_bias_mean": by_x0["bias"].abs().mean(),
            "rmse_mean": by_x0["rmse"].mean(),
        })
    summary = pd.DataFrame(rows).sort_values(["reference", "bandwidth_multiplier", "method"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
