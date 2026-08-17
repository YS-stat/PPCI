# Validated Compact Results

This directory contains the compact, plot-ready summaries from the final
paper-scale runs. Replicate-level tables, per-candidate tuning tables, kernel
matrices, eigensystems, and fitted weight vectors are intentionally omitted.

## Main Experiments

- `simulation/summary_by_x0.csv`: PPCI, LO, and global PPI at each of the 343
  Simulation target points.
- `simulation/aggregate.csv`: equal-target aggregate Simulation metrics.
- `simulation/tuning_diagnostics.csv`: selected-grid and screen diagnostics.
- `income/summary_by_target.csv`: PPCI, LO, and global PPI by age and sex.
- `income/aggregate_by_sex.csv`: sex-specific aggregate Income metrics.
- `income/sigma_by_age.csv`: the three variance components used in the Income
  predictor-role figure.
- `blogfeedback/summary_by_x0.csv`: PPCI, LO, and global PPI at 50 target
  points under the corrected held-out reference population.
- `blogfeedback/aggregate.csv`: aggregate BlogFeedback metrics.

## Additional Experiments

- `predictor_quality/`: PPCI/PPCI++/LO/PPI summaries for
  `q in {0.9,0.5,0}`.
- `income/unlabeled_size_summary.csv`: the paired-nested trajectory over
  `N=500,1500,...,10500`.
- `split_comparison/`: paired two-fold and no-split accuracy, runtime, and GPU
  memory summaries.
- `simulation/sensitivity/`: P1 `c_bias`, upper-bandwidth, and lambda-envelope
  sensitivity summaries.
- `nw/`: compact analytic, quadratic-mechanism, and same-bandwidth NW/RKHS
  localization summaries.

The `shard_manifests.json` files retain the exact commands, arguments,
configuration hashes, runtime versions, and executable-source hash. All formal
PPCI results in this release use source fingerprint
`cadbd8b7d3f84e630b0690c9baf0c8c99f8f38e5adeed38a645d932debb7fcc8`.

The archived method labels `GH_P1_c*` and `LO_GH_P1_c*w` are internal run
identifiers for, respectively, PPCI with the P1 screen and LO evaluated with
the same selected localization weights. They are retained verbatim so the
released CSVs remain identical to the validated merged outputs.

Coverage Monte Carlo standard errors are clustered by independent unlabeled
sample. Method comparisons within a scenario use paired data draws whenever
the method definition permits it.
