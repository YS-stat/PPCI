# Revision Release Notes

This package is the reproducibility release for the revised PPCI experiments.
Relative to the earlier public code, it includes the following substantive
updates:

- covariate-only joint selection of bandwidth and regularization using the two
  stability screens, the P1 labeled-scale bias screen, and least-violation
  fallback;
- a fixed deployable Simulation predictor that depends only on covariates;
- paired PPCI, localized-only (LO), and global PPI summaries for the final
  343-target Simulation;
- the corrected BlogFeedback reference population and a predictor-training
  split disjoint from the PPCI inference population;
- point-specific, labeled-cross-fitted PPCI++ with one normal-Wald interval;
- Income unlabeled-sample-size, two-fold/no-split, tuning-sensitivity, and
  NW/RKHS-localization experiments;
- compact plot-ready results, exact shard manifests, six final PDF figures,
  and the formal server package versions.

The formal results were generated from the full server-archive source
fingerprint
`cadbd8b7d3f84e630b0690c9baf0c8c99f8f38e5adeed38a645d932debb7fcc8`.
All 17 tests passed in the formal CUDA environment. The public package omits
large replicate tables, intermediate shards, obsolete archival utilities,
server schedulers, and logs. Its core algorithm and experiment entry points
are byte-identical to the formal archive, and the validated compact summaries
are unchanged from the merged formal outputs.
