# Reference-smoother sensitivity

This appendix experiment changes only the full-data finite-population reference
target used to score the fixed PPCI and LO estimates and intervals. It compares
Matérn-5/2 Nadaraya--Watson references at `0.8 h0`, `h0`, and `1.2 h0` with a
Gaussian reference whose bandwidth matches the local kernel-weight effective
sample size at `h0`.

Reference shifts, RMSE, and width use all 62 Income and 50 BlogFeedback target
points. Coverage uses eight prespecified targets per data set and 15 independent
unlabeled samples with 20 labeled draws each, hence 300 intervals per target
and method. Replicate tables are omitted; regenerate them on four CUDA devices
with

```bash
PYTHON_BIN=python bash experiments/launch_reference_smoother_pilot.sh
```

After the four jobs finish, rescore the representative-target intervals with

```bash
python experiments/run_reference_smoother_sensitivity.py \
  --dataset income --data data/census_income/census_income.npz \
  --income-sexes 1,2 --income-ages 70,80,90,100 \
  --replicate-files \
    runs/reference_smoother_sensitivity/income_sex1_gpu0/replicates.csv \
    runs/reference_smoother_sensitivity/income_sex2_gpu1/replicates.csv \
  --output-dir runs/reference_smoother_sensitivity/income_rescored

python experiments/run_reference_smoother_sensitivity.py \
  --dataset blogfeedback --data data/blogfeedback/blogfeedback.zip \
  --blog-x0-indices 0,7,14,21,28,35,42,49 \
  --replicate-files \
    runs/reference_smoother_sensitivity/blog_gpu2/replicates.csv \
    runs/reference_smoother_sensitivity/blog_gpu3/replicates.csv \
  --output-dir runs/reference_smoother_sensitivity/blog_rescored
```

The `income_all_targets/` and `blog_all_targets/` configurations additionally
record the all-target rescoring settings and formal summary inputs. Their
aggregate results, together with representative-target coverage, are:

| Data | Reference | Shift | PPCI coverage | LO coverage | PPCI RMSE | LO RMSE | PPCI width | LO width |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Income | Matérn `0.8 h0` | 0.1976 | 0.9329 | 0.9329 | 0.5181 | 0.5782 | 1.9079 | 2.1528 |
| Income | Matérn `h0` | 0 | 0.9429 | 0.9375 | 0.5571 | 0.6134 | 1.9079 | 2.1528 |
| Income | Matérn `1.2 h0` | 0.1681 | 0.9233 | 0.9242 | 0.6404 | 0.6898 | 1.9079 | 2.1528 |
| Income | Gaussian, ESS matched | 0.0630 | 0.9400 | 0.9408 | 0.5364 | 0.5946 | 1.9079 | 2.1528 |
| BlogFeedback | Matérn `0.8 h0` | 0.0200 | 0.9175 | 0.9404 | 0.0425 | 0.0603 | 0.1442 | 0.2227 |
| BlogFeedback | Matérn `h0` | 0 | 0.9533 | 0.9404 | 0.0366 | 0.0563 | 0.1442 | 0.2227 |
| BlogFeedback | Matérn `1.2 h0` | 0.0159 | 0.9213 | 0.9288 | 0.0397 | 0.0583 | 0.1442 | 0.2227 |
| BlogFeedback | Gaussian, ESS matched | 0.0016 | 0.9554 | 0.9433 | 0.0367 | 0.0564 | 0.1442 | 0.2227 |

The ESS-matched kernel change leaves the results nearly unchanged. The wider
bandwidth perturbations define meaningfully different evaluation targets, but
PPCI retains lower RMSE and shorter intervals than LO throughout.
