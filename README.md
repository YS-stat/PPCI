# Prediction-Powered Conditional Inference (PPCI)

This repository contains the conditional-mean implementation and compact
reproducibility artifacts for the paper
[Prediction-Powered Conditional Inference](https://arxiv.org/pdf/2603.05575).
It includes the controlled Simulation, Census Income, and BlogFeedback
experiments, together with the PPCI++ predictor-quality study, the
unlabeled-sample-size study, the two-fold versus no-split comparison, and the
Nadaraya--Watson (NW) localization studies.

## Method

For each candidate bandwidth `h` and regularization level `lambda`, PPCI
constructs an empirical RKHS localization weight using covariates only. The
paper-facing tuning rule applies the operator and local-leverage stability
screens and the P1 labeled-scale bias screen

```text
sqrt(n * lambda * [D_hat_h(x0; lambda) - Q_hat_h(x0; lambda)]_+
     / Q_hat_h(x0; lambda)) <= c_bias.
```

Among feasible candidates, the procedure minimizes
`sqrt(Q_hat_h) / abs(mean(w_hat))`. If the feasible set is empty, it selects
the candidate with the smallest normalized constraint violation and uses the
same scale proxy as a tie-breaker. The candidate grids are

```text
h = a * median_u ||X_tilde_u - x0||,
lambda = c / {n * log(log(n + exp(exp(1))))}.
```

The tuning step never uses outcomes or predictions. These enter only after
`(h, lambda)` and the two fold-specific localization weights have been fixed.

## Repository Layout

```text
ppci_condmean/       RKHS localization, P1 tuning, estimators, and diagnostics
experiments/         main and reviewer-facing experiment entry points
experiments/nw/      NW/RKHS-localization mechanism experiments
data/                processed Income and BlogFeedback inputs
results/             compact validated summaries used by figures and tables
figures/             six final paper-facing PDF figures
tests/               deterministic unit and smoke tests
tools/               result merging, calibration, and target-rescoring tools
```

Large kernel matrices, eigensystems, fitted weight vectors, and paper-scale
replicate tables are intentionally excluded.

## Installation

The formal results used Python 3.10.12 with NumPy 1.26.4, SciPy 1.13.1,
Pandas 2.2.2, scikit-learn 1.5.0, Matplotlib 3.9.0, Joblib 1.4.2,
LightGBM 4.6.0, and PyTorch 2.5.1 with CUDA 12.1. Install the CPU dependencies
with

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PyTorch is optional. For GPU execution, install the PyTorch build matching the
local CUDA runtime and pass `--backend cuda --gpu-id 0`. CPU execution uses
NumPy and SciPy.

## Verification

Run the deterministic test suite and short CPU smoke experiments with

```bash
python -m unittest discover -s tests -v
python experiments/run_simulation.py --smoke --backend cpu \
  --output-dir runs/smoke_simulation
python experiments/run_income.py --smoke --backend cpu \
  --output-dir runs/smoke_income
python experiments/run_blogfeedback.py --smoke --model extratrees \
  --backend cpu --output-dir runs/smoke_blog
python experiments/run_split_comparison.py --smoke --backend cpu \
  --output-dir runs/smoke_split
python experiments/run_income_unlabeled_sweep.py --smoke --backend cpu \
  --output-dir runs/smoke_income_N
python experiments/nw/run_nw_mechanism_analytic.py \
  --output-dir runs/smoke_nw_analytic
python experiments/nw/run_nw_localization_quadratic_mc.py --smoke --workers 1 \
  --backend cpu --output-dir runs/smoke_nw_mc
```

The release passes all 17 tests in the formal server environment, including
CPU/CUDA parity. The CUDA test is skipped automatically when CUDA is absent.

## Main Paper Configurations

Each main experiment uses 50 independent unlabeled samples and 20 independent
labeled samples per unlabeled sample, yielding 1,000 confidence intervals and
50 independent unlabeled clusters per target point. Coverage Monte Carlo
standard errors are computed across those clusters.

| Experiment | `(n, N, targets)` | `C_h` | `C_lambda` | `(c_op, c_loc, c_bias)` |
|---|---:|---|---|---:|
| Simulation | `(400, 10000, 343)` | `0.8,1.0,1.15,1.2` | 35 log points in `[0.02,60]` | `(12,4,0.10)` |
| Income, sex 1 | `(300, 10000, 31)` | `1.0,1.2,1.4` | 41 log points in `[0.1,1000]` | `(12,4,22)` |
| Income, sex 2 | `(300, 10000, 31)` | `1.0,1.2,1.4` | 41 log points in `[0.1,1000]` | `(10,3,15)` |
| BlogFeedback | `(300, 10000, 50)` | `0.8,0.9,1.0` | 81 log points in `[0.1,10000]` | `(12,4,300)` |

The Simulation target grid is the `7^3` Cartesian grid on `[0.70,0.85]^3`.
It uses the fixed deployable predictor
`f_q(X) = q m(X) + (1-q) s(X)`, where
`s(X) = sqrt(1.5825) sin(6 pi X_1)` and `q=0.9`; predictions never read the
response. BlogFeedback trains LightGBM on a split disjoint from the inference
population, and its empirical reference target is defined on that same held-out
inference population.

### Simulation

```bash
python experiments/run_simulation.py \
  --seed 12100 --settings base \
  --n-label 400 --n-unlab 10000 \
  --x0-num 7 --x0-region cube:0.7:0.85:7 \
  --unlab-reps 50 --label-reps 20 \
  --predictor-qualities 0.9 --sigma-eps-values 1 \
  --h-grid-mode median_grid --h-factors 0.8,1.0,1.15,1.2 \
  --bias-screens p1_label --c-biases 0.10 \
  --lambda-factor-min 0.02 --lambda-factor-max 60 \
  --lambda-grid-size 35 --lambda-grid-mode shrinking \
  --tau-op 12 --tau-loc 4 --constraint-fallback least_violation \
  --include-lo-ppi --backend cuda --gpu-id 0
```

Use `--x0-indices` to shard the 343 target indices across devices and merge
the completed shards with `tools/merge_simulation_shards.py`.

### Census Income

Sex 1 and sex 2 use separately predeclared stability and bias constants.

```bash
python experiments/run_income.py --sexes 1 --seed 15100 \
  --n-label 300 --n-unlab 10000 --unlab-reps 50 --label-reps 20 \
  --h-factors 1.0,1.2,1.4 \
  --lambda-factor-min 0.1 --lambda-factor-max 1000 \
  --lambda-grid-size 41 --c-biases 22 --tau-op 12 --tau-loc 4 \
  --backend cuda --gpu-id 0

python experiments/run_income.py --sexes 2 --seed 15300 \
  --n-label 300 --n-unlab 10000 --unlab-reps 50 --label-reps 20 \
  --h-factors 1.0,1.2,1.4 \
  --lambda-factor-min 0.1 --lambda-factor-max 1000 \
  --lambda-grid-size 41 --c-biases 15 --tau-op 10 --tau-loc 3 \
  --backend cuda --gpu-id 1
```

### BlogFeedback

```bash
python experiments/run_blogfeedback.py \
  --seed 2025 --n-label 300 --n-unlab 10000 --n-x0 50 \
  --unlab-reps 50 --unlab-rep-offset 1000 --label-reps 20 \
  --h-factors 0.8,0.9,1.0 \
  --lambda-factor-min 0.1 --lambda-factor-max 10000 \
  --lambda-grid-size 81 --lambda-grid-mode shrinking \
  --tau-op 12 --tau-loc 4 --c-biases 300 \
  --bias-screens p1_label --constraint-fallback least_violation \
  --model lightgbm --backend cuda --gpu-id 2
```

## Additional Experiments

The predictor-quality experiment compares PPCI, PPCI++, LO, and global PPI at
`q in {0.9,0.5,0}` with `n=500`, `N=10000`, eight target points, and the same
normal critical value for PPCI and PPCI++. PPCI++ estimates a point-specific,
labeled-cross-fitted coefficient `omega(x0)` and clips it to `[0,1]`.

```bash
python experiments/run_simulation.py \
  --n-label 500 --n-unlab 10000 \
  --x0-num 2 --x0-region cube:0.7:0.85:2 \
  --unlab-reps 50 --label-reps 20 \
  --predictor-qualities 0.9,0.5,0 --sigma-eps-values 1 \
  --include-ppci-plus --omega-folds 5 --backend cuda
```

The other appendix studies use `experiments/run_split_comparison.py`,
`experiments/run_income_unlabeled_sweep.py`, and the scripts under
`experiments/nw/`. Their compact outputs and exact configurations are archived
under `results/`.

## Validated Main Results

| Experiment/method | Coverage mean | Coverage min | Coverage max | RMSE | Mean width |
|---|---:|---:|---:|---:|---:|
| Simulation PPCI | 0.9524 | 0.926 | 0.973 | 0.2899 | 1.1111 |
| Simulation LO | 0.9375 | 0.911 | 0.961 | 0.3550 | 1.2871 |
| Income PPCI, sex 1 | 0.9413 | 0.916 | 0.959 | 0.6483 | 2.2802 |
| Income PPCI, sex 2 | 0.9408 | 0.922 | 0.959 | 0.4659 | 1.5356 |
| BlogFeedback PPCI | 0.9510 | 0.935 | 0.963 | 0.0366 | 0.1442 |
| BlogFeedback LO | 0.9482 | 0.933 | 0.963 | 0.0563 | 0.2227 |

All main PPCI tuning decisions are feasible under the three screens; the
least-violation fallback rate is zero. Global PPI estimates a population-level
target and is retained only as an out-of-scope comparator for these conditional
targets.

## Results and Provenance

See [`results/README.md`](results/README.md) for the compact file map and
[`RESULTS_SCHEMA.md`](RESULTS_SCHEMA.md) for metric definitions. Every formal
shard manifest records the command, full arguments, runtime versions,
configuration SHA-256, and executable-source SHA-256. All included formal
results share the full server-archive source fingerprint

```text
cadbd8b7d3f84e630b0690c9baf0c8c99f8f38e5adeed38a645d932debb7fcc8
```

The core algorithm and experiment entry points in this public package are
byte-identical to the corresponding files in that archive. Documentation and
compact-result packaging were updated after the formal runs, while obsolete
internal schedulers and archival utilities were omitted from the public
release.
