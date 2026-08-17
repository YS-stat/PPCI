# Same-Bandwidth NW versus RKHS localization Follow-up

This follow-up fixes the same Matérn-5/2 base bandwidth `h in [0.08, 0.2, 0.4, 0.6]` for NW and RKHS localization weights. At every fixed h, RKHS localization selects only lambda using the existing stability screens, P1 labelled-scale budget `c_bias=0.18`, and least-normalized-violation fallback. Thus bandwidth selection cannot explain the comparison.

All methods use paired samples with `n=200`, `N=5000`, and `500` replicates. The main curvature remains `A=4`; `A in [2.0, 4.0, 8.0]` is a predeclared mechanism stress test. The finite-dimensional `w*` appears only in the representative weight plot.

## Main A=4 Same-h Results

|      b |      h | family            |   coverage |    bias |   rmse |   mean_width |   normalized_quadratic_imbalance |
|-------:|-------:|:------------------|-----------:|--------:|-------:|-------------:|---------------------------------:|
| 0.0000 | 0.0800 | NW                |     0.9560 |  0.0260 | 0.1766 |       0.7088 |                           0.0077 |
| 0.0000 | 0.0800 | RKHS_LOCALIZATION |     0.9800 |  0.0023 | 0.3806 |       1.4608 |                           0.0000 |
| 0.0000 | 0.2000 | NW                |     0.6520 |  0.1851 | 0.2175 |       0.4556 |                           0.0476 |
| 0.0000 | 0.2000 | RKHS_LOCALIZATION |     0.9680 |  0.0019 | 0.2601 |       1.0413 |                           0.0001 |
| 0.0000 | 0.4000 | NW                |     0.0000 |  0.5878 | 0.5943 |       0.3366 |                           0.1477 |
| 0.0000 | 0.4000 | RKHS_LOCALIZATION |     0.9520 | -0.0049 | 0.1960 |       0.7948 |                           0.0003 |
| 0.0000 | 0.6000 | NW                |     0.0000 |  0.8638 | 0.8675 |       0.3028 |                           0.2166 |
| 0.0000 | 0.6000 | RKHS_LOCALIZATION |     0.9420 | -0.0070 | 0.1676 |       0.6776 |                           0.0005 |
| 2.0000 | 0.0800 | NW                |     0.9560 |  0.0259 | 0.1765 |       0.7089 |                           0.0077 |
| 2.0000 | 0.0800 | RKHS_LOCALIZATION |     0.9800 |  0.0023 | 0.3806 |       1.4609 |                           0.0000 |
| 2.0000 | 0.2000 | NW                |     0.6460 |  0.1852 | 0.2175 |       0.4563 |                           0.0476 |
| 2.0000 | 0.2000 | RKHS_LOCALIZATION |     0.9680 |  0.0020 | 0.2602 |       1.0414 |                           0.0001 |
| 2.0000 | 0.4000 | NW                |     0.0000 |  0.5882 | 0.5947 |       0.3386 |                           0.1477 |
| 2.0000 | 0.4000 | RKHS_LOCALIZATION |     0.9520 | -0.0048 | 0.1961 |       0.7951 |                           0.0003 |
| 2.0000 | 0.6000 | NW                |     0.0000 |  0.8643 | 0.8680 |       0.3061 |                           0.2166 |
| 2.0000 | 0.6000 | RKHS_LOCALIZATION |     0.9420 | -0.0069 | 0.1677 |       0.6781 |                           0.0005 |

## Curvature Stress at the Widest Common h

|      A |      b | family            |   coverage |    bias |   rmse |   mean_width |
|-------:|-------:|:------------------|-----------:|--------:|-------:|-------------:|
| 2.0000 | 0.0000 | NW                |     0.0000 |  0.4306 | 0.4377 |       0.2998 |
| 2.0000 | 0.0000 | RKHS_LOCALIZATION |     0.9440 | -0.0079 | 0.1677 |       0.6775 |
| 2.0000 | 2.0000 | NW                |     0.0000 |  0.4310 | 0.4382 |       0.3032 |
| 2.0000 | 2.0000 | RKHS_LOCALIZATION |     0.9460 | -0.0078 | 0.1677 |       0.6780 |
| 4.0000 | 0.0000 | NW                |     0.0000 |  0.8638 | 0.8675 |       0.3028 |
| 4.0000 | 0.0000 | RKHS_LOCALIZATION |     0.9420 | -0.0070 | 0.1676 |       0.6776 |
| 4.0000 | 2.0000 | NW                |     0.0000 |  0.8643 | 0.8680 |       0.3061 |
| 4.0000 | 2.0000 | RKHS_LOCALIZATION |     0.9420 | -0.0069 | 0.1677 |       0.6781 |
| 8.0000 | 0.0000 | NW                |     0.0000 |  1.7303 | 1.7323 |       0.3143 |
| 8.0000 | 0.0000 | RKHS_LOCALIZATION |     0.9420 | -0.0051 | 0.1676 |       0.6779 |
| 8.0000 | 2.0000 | NW                |     0.0000 |  1.7308 | 1.7328 |       0.3175 |
| 8.0000 | 2.0000 | RKHS_LOCALIZATION |     0.9420 | -0.0051 | 0.1676 |       0.6784 |

## NW Lower-grid Robustness

|      b | grid     | rule   |   coverage |   bias |   rmse |   width |   selected_h_mean |   selected_h_median |   lower_bound_hit_rate |   us_clipping_rate |
|-------:|:---------|:-------|-----------:|-------:|-------:|--------:|------------------:|--------------------:|-----------------------:|-------------------:|
| 0.0000 | current  | CV     |     0.9560 | 0.0297 | 0.1783 |  0.7089 |            0.0830 |              0.0816 |                 0.0780 |             0.0000 |
| 0.0000 | current  | US     |     0.9640 | 0.0175 | 0.1970 |  0.7923 |            0.0650 |              0.0626 |                 0.0780 |             0.1720 |
| 0.0000 | extended | CV     |     0.9580 | 0.0299 | 0.1802 |  0.7189 |            0.0821 |              0.0832 |                 0.0040 |             0.0000 |
| 0.0000 | extended | US     |     0.9680 | 0.0176 | 0.2018 |  0.8135 |            0.0630 |              0.0638 |                 0.0040 |             0.0100 |
| 2.0000 | current  | CV     |     0.9560 | 0.0288 | 0.1793 |  0.7157 |            0.0813 |              0.0816 |                 0.0860 |             0.0000 |
| 2.0000 | current  | US     |     0.9680 | 0.0170 | 0.1982 |  0.7988 |            0.0638 |              0.0626 |                 0.0860 |             0.1800 |
| 2.0000 | extended | CV     |     0.9580 | 0.0285 | 0.1817 |  0.7253 |            0.0804 |              0.0832 |                 0.0040 |             0.0000 |
| 2.0000 | extended | US     |     0.9720 | 0.0166 | 0.2037 |  0.8206 |            0.0617 |              0.0638 |                 0.0040 |             0.0100 |

The key diagnostic is `normalized_quadratic_imbalance = |P_N(w X^2) / P_N(w)|`. It directly measures the residual curvature bias channel. The frontier is explicitly a P1-constrained fixed-h path, not an exhaustive unconstrained lambda search.
