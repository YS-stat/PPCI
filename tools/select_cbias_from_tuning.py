#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


CBIAS_RE = re.compile(r"(?:^|_)c(?P<value>[0-9]+(?:\.[0-9]+)?)$")


def parse_cbias(method: str) -> float | None:
    match = CBIAS_RE.search(str(method))
    return float(match.group("value")) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Outcome-blind c_bias calibration from tuning feasibility.")
    parser.add_argument("tuning_decisions")
    parser.add_argument("--max-fallback-rate", type=float, default=0.05)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    table = pd.read_csv(args.tuning_decisions)
    if "fallback" not in table:
        if "tuning_status" not in table:
            raise ValueError("Input must contain fallback or tuning_status.")
        table["fallback"] = table["tuning_status"].astype(str).str.contains("fallback", regex=False)
    table["c_bias"] = table["method"].map(parse_cbias)
    table = table[table["c_bias"].notna()].copy()
    if table.empty:
        raise ValueError("No method labels ending in _c<value> were found.")
    # A shared covariate draw can be reused across outcome-noise/predictor scenarios.
    # Count that tuning decision once; outcome scenario labels must not affect calibration.
    identity = [column for column in ("c_bias", "x0_index", "unlab_rep") if column in table]
    if len(identity) >= 2:
        table = table.drop_duplicates(identity)

    summary = table.groupby("c_bias", as_index=False).agg(
        tuning_decisions=("fallback", "size"), fallback_rate=("fallback", "mean")
    ).sort_values("c_bias")
    feasible = summary[summary["fallback_rate"] <= args.max_fallback_rate]
    if feasible.empty:
        raise RuntimeError("No c_bias candidate meets the fallback-rate threshold.")
    selected = float(feasible.iloc[0]["c_bias"])
    payload = {
        "selection_rule": "smallest c_bias with fallback_rate <= max_fallback_rate",
        "max_fallback_rate": args.max_fallback_rate,
        "selected_c_bias": selected,
        "diagnostics": summary.to_dict(orient="records"),
        "uses_outcomes": False,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
