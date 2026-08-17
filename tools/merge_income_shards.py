#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_income import aggregate_from_targets, summarize_tuning_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint Income age/sex shards.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-targets", type=int, default=62)
    parser.add_argument("shards", nargs="+")
    args = parser.parse_args()

    shards = [Path(value) for value in args.shards]
    manifests = [json.loads((shard / "run_manifest.json").read_text()) for shard in shards]
    source_hashes = {manifest.get("source_sha256") for manifest in manifests}
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError(f"Income shards have inconsistent source hashes: {source_hashes}")

    targets = pd.concat(
        [pd.read_csv(shard / "summary_by_target.csv") for shard in shards],
        ignore_index=True,
    )
    keys = ["sex", "age", "method"]
    if targets.duplicated(keys).any():
        raise ValueError("Income shard target assignments overlap.")
    n_targets = targets[["sex", "age"]].drop_duplicates().shape[0]
    if args.expected_targets > 0 and n_targets != args.expected_targets:
        raise ValueError(f"Expected {args.expected_targets} Income targets, found {n_targets}.")
    targets = targets.sort_values(keys)

    aggregate = aggregate_from_targets(targets)
    by_sex_blocks = []
    for sex, group in targets.groupby("sex", sort=True):
        block = aggregate_from_targets(group)
        block.insert(0, "sex", sex)
        by_sex_blocks.append(block)
    by_sex = pd.concat(by_sex_blocks, ignore_index=True)

    tuning = pd.concat(
        [pd.read_csv(shard / "tuning_decisions.csv") for shard in shards],
        ignore_index=True,
    )
    tuning_keys = ["sex", "age", "method", "unlab_rep"]
    if tuning.duplicated(tuning_keys).any():
        raise ValueError("Income tuning decisions overlap across shards.")
    tuning = tuning.sort_values(tuning_keys)
    tuning_by_target = summarize_tuning_decisions(tuning, ["sex", "age"])
    tuning_by_method = summarize_tuning_decisions(tuning, [])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "summary_by_target.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    by_sex.to_csv(output / "aggregate_by_sex.csv", index=False)
    tuning.to_csv(output / "tuning_decisions.csv", index=False)
    tuning_by_target.to_csv(output / "tuning_by_target.csv", index=False)
    tuning_by_method.to_csv(output / "tuning_by_method.csv", index=False)
    (output / "shard_manifests.json").write_text(
        json.dumps(
            [{"shard": str(shard), "manifest": manifest} for shard, manifest in zip(shards, manifests)],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
