# NW versus Kernel RKHS localization PPCI

## Design

`X ~ Unif[-1,1]` and `Y = beta + b X + A X^2 + epsilon`, with `beta=0`, `A=4`, `sigma=1`, `n=200`, `N=5000`, and `b in [0.0, 2.0]`. The auxiliary predictor is the exact conditional mean, and every confidence interval targets `theta(0)=beta`.

Each setting has 500 paired Monte Carlo replicates. Within a replicate all methods share the same labelled and unlabelled samples. NW-CV uses an independent pilot sample of size 200 and leave-one-out regression CV over a predeclared 35-point bandwidth grid. NW-US uses `h_CV * n^(-0.05)`, clipped only to the predeclared grid range.

The actual RKHS localization methods call the project implementation with median-based `G_h=[0.8, 1.0, 1.15, 1.2]`, shrinking `G_lambda`, stability thresholds `12/4`, the P1 labelled-scale screen, and least-normalized-violation fallback. The main table fixes `c_bias=0.18`; `[0.12, 0.18, 0.25]` are reported in `cbias_sensitivity.csv`.

`ORACLE_WSTAR_*` uses `w*(x)=9/4-15x^2/4`. It is a finite-dimensional oracle signed-balance reference, not a weight that general kernel RKHS localization PPCI is required to recover.

## Main Results

| setting   | method                       |   coverage |    bias |   rmse |   mean_width |
|:----------|:-----------------------------|-----------:|--------:|-------:|-------------:|
| b=0       | NW_LO_CV                     |     0.9540 |  0.0294 | 0.1818 |       0.6945 |
| b=0       | NW_LO_US                     |     0.9420 |  0.0136 | 0.2028 |       0.7726 |
| b=0       | NW_PPCI_CV                   |     0.9640 |  0.0302 | 0.1818 |       0.7110 |
| b=0       | NW_PPCI_US                   |     0.9700 |  0.0145 | 0.2027 |       0.7958 |
| b=0       | ORACLE_WSTAR_LO              |     0.9520 | -0.0097 | 0.1487 |       0.6204 |
| b=0       | ORACLE_WSTAR_PPCI            |     0.9460 | -0.0047 | 0.1057 |       0.4251 |
| b=0       | RKHS_LOCALIZATION_LO_GH_P1   |     0.9560 | -0.0001 | 0.1682 |       0.6765 |
| b=0       | RKHS_LOCALIZATION_PPCI_GH_P1 |     0.9700 | -0.0005 | 0.1674 |       0.6808 |
| b=2       | NW_LO_CV                     |     0.9380 |  0.0289 | 0.1861 |       0.7046 |
| b=2       | NW_LO_US                     |     0.9400 |  0.0131 | 0.2062 |       0.7815 |
| b=2       | NW_PPCI_CV                   |     0.9600 |  0.0288 | 0.1833 |       0.7171 |
| b=2       | NW_PPCI_US                   |     0.9700 |  0.0135 | 0.2044 |       0.8014 |
| b=2       | ORACLE_WSTAR_LO              |     0.9560 | -0.0090 | 0.1697 |       0.6967 |
| b=2       | ORACLE_WSTAR_PPCI            |     0.9480 | -0.0046 | 0.1070 |       0.4297 |
| b=2       | RKHS_LOCALIZATION_LO_GH_P1   |     0.9500 |  0.0011 | 0.1736 |       0.6882 |
| b=2       | RKHS_LOCALIZATION_PPCI_GH_P1 |     0.9700 | -0.0005 | 0.1674 |       0.6813 |

## Empirical Relative Variance Reduction

| setting   | pair                    | lo_method                  | ppci_method                  |   empirical_variance_lo |   empirical_variance_ppci |   relative_variance_reduction |   paired_error_correlation |
|:----------|:------------------------|:---------------------------|:-----------------------------|------------------------:|--------------------------:|------------------------------:|---------------------------:|
| b=0       | NW_CV                   | NW_LO_CV                   | NW_PPCI_CV                   |                  0.0322 |                    0.0322 |                        0.0014 |                     0.9823 |
| b=0       | NW_US                   | NW_LO_US                   | NW_PPCI_US                   |                  0.0410 |                    0.0410 |                        0.0013 |                     0.9800 |
| b=0       | RKHS_LOCALIZATION_GH_P1 | RKHS_LOCALIZATION_LO_GH_P1 | RKHS_LOCALIZATION_PPCI_GH_P1 |                  0.0283 |                    0.0281 |                        0.0086 |                     0.9837 |
| b=0       | ORACLE_WSTAR            | ORACLE_WSTAR_LO            | ORACLE_WSTAR_PPCI            |                  0.0221 |                    0.0112 |                        0.4938 |                     0.6382 |
| b=2       | NW_CV                   | NW_LO_CV                   | NW_PPCI_CV                   |                  0.0339 |                    0.0329 |                        0.0302 |                     0.9748 |
| b=2       | NW_US                   | NW_LO_US                   | NW_PPCI_US                   |                  0.0424 |                    0.0417 |                        0.0180 |                     0.9749 |
| b=2       | RKHS_LOCALIZATION_GH_P1 | RKHS_LOCALIZATION_LO_GH_P1 | RKHS_LOCALIZATION_PPCI_GH_P1 |                  0.0302 |                    0.0281 |                        0.0701 |                     0.9666 |
| b=2       | ORACLE_WSTAR            | ORACLE_WSTAR_LO            | ORACLE_WSTAR_PPCI            |                  0.0288 |                    0.0115 |                        0.6017 |                     0.5801 |

## Checks and Interpretation

- Core estimates and standard errors are finite: `True`.
- All intervals target beta: `True`.
- Actual RKHS localization weights exhibit signed weights: `True`.
- Oracle empirical variance is compared with its analytic formula in `sanity_checks.json`.
- Fixed-bandwidth uniform NW is recorded separately in `fixed_h_sanity.csv` and checked against the analytic bias `A h^2 / 3`.

The results are descriptive Monte Carlo evidence. No tuning constant was selected after inspecting coverage, and the two non-main `c_bias` values are sensitivity analyses only.
