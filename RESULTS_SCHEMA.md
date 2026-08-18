# Result schema

The experiment runners write compact CSV summaries. Large Gram matrices,
weight vectors, and per-candidate eigensystems are never retained.

## Plotting tables

- `summary_by_x0.csv` or `summary_by_target.csv`: one row per method and target
  point. Use these files for coverage, bias, RMSE, standard-error, and width
  figures.
- `summary_by_setting.csv`: one row per Simulation method and predictor/noise
  scenario. Use this for predictor-quality curves.
- `aggregate.csv` or `aggregate_equal_x0.csv`: equal-target summaries for
  manuscript tables and quick checks.
- `summary_by_N.csv`: Income unlabeled-sample-size trajectory.

The optional `replicates.csv` is written only when `--save-replicates` is
specified. It is needed for paired rescoring or diagnostics, not for the main
figures.

## Reference-smoother sensitivity

- `reference_targets.csv`: alternative finite-population reference targets,
  kernel bandwidths, and kernel-weight effective sample sizes.
- `target_shifts.csv` and `target_shift_summary.csv`: signed and absolute
  changes relative to the Matérn-5/2 reference at `h0`.
- `rescored_by_target.csv` and `rescored_aggregate.csv`: representative-target
  coverage, bias, RMSE, and width after changing only the evaluation target.
- `formal_rescored_by_target.csv` and `formal_rescored_aggregate.csv`:
  all-target bias, RMSE, and width reconstructed from the formal 1,000-run
  summaries. Width is invariant across reference targets because the intervals
  themselves are held fixed.

## Accuracy and uncertainty

- `coverage`: empirical fraction of intervals containing the evaluation target.
- `coverage_cluster_se`: Monte Carlo standard error computed across independent
  unlabeled draws, after averaging the labeled replications within each draw.
- `bias`, `abs_bias`, `rmse`, `emp_sd`: signed bias, absolute signed bias, root
  mean squared error, and empirical standard deviation of the point estimator.
- `se_mean`, `width`: average reported standard error and confidence-interval
  width.
- `sigma2_Y_mean`, `sigma2_Y_minus_f_mean`, `sigma2_f_mean`: empirical weighted
  variance components for localized LO and PPCI uncertainty.

## Tuning diagnostics

- `h_factor_mean`, `lambda_factor_mean`: selected median-scaled bandwidth and
  shrinking-grid regularization factors.
- `op_score`, `loc_score`, `bias_score_mean`, `bias_budget_mean`: operator,
  pointwise-leverage, and P1 bias-screen diagnostics.
- `fallback_rate`: fraction of tuning decisions using least normalized violation.
- `nw_corr_mean`, `nw_relative_difference_mean`,
  `negative_weight_fraction_mean`, `M_lambda_over_eigmax_mean`: descriptive
  RKHS localization versus NW and spectral diagnostics. They are not tuning inputs.

## Predictor adaptation

- `omega_mean`, `omega_sd_mean`, `omega_min_mean`, `omega_max_mean`: summaries
  of labeled-cross-fitted PPCI++ coefficients.
- `omega_clipped_rate`: fraction of raw coefficient estimates clipped to
  `[0,1]`.
- PPCI++ uses the same normal critical value as the primary PPCI interval;
  interval variants are not part of the released procedure.

## Runtime

- `tuning_seconds_mean_per_unlab_draw`: one-time weight tuning cost for an
  independent unlabeled draw.
- `estimation_seconds_mean`: estimator cost after weights are available.
- `amortized_procedure_seconds_mean`: tuning cost divided across the labeled
  replications sharing those weights, plus estimation cost.
- Split-comparison files additionally retain `procedure_seconds_*` and
  `gpu_peak_mb_*` for the complete two-fold or no-split procedure.
- `run_manifest.json` records the command, all arguments, resolved backend and
  package versions, a configuration SHA-256, and a source SHA-256 over every
  released Python/environment-specification file. BlogFeedback manifests also
  retain model-training time and verify target exclusion.

All method comparisons within a scenario use paired data draws and the same
covariate-only tuning objects whenever the method definition permits it.
