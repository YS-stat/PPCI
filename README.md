# PPCI: Prediction-Powered Conditional Inference

This repository contains the implementation of **Prediction-Powered Conditional Inference (PPCI)**.
PPCI enables valid uncertainty quantification for **conditional functionals** (e.g., conditional means)
in settings with **scarce labeled data**, **rich unlabeled data**, and an available **black-box predictor**.

---

## Requirements

- **CUDA-enabled GPU** environment
- Core computation is based on **CuPy**

---

## Repository Structure

The project has been reorganized into two main independent directories based on the type of inference:

1. `Conditional_mean_inference/`
2. `Conditional_quantile_inference/`

Each directory contains its own core functions, simulation scripts, and real-data applications.

---

## Part 1: Conditional Mean Inference

All files for mean inference are located in the `Conditional_mean_inference/` directory.

**Core File:**
- `conditional_mean_functions.py` — base functions for conditional mean inference

### 1) Simulation Experiments
- **Notebook:** `simu_mean.ipynb`
- **Outputs:** CSV results are saved to `./results/` and PDFs are generated in the root of this module.

### 2) Census Income Data Analysis
- **Location:** `./census_income_data/`
- **Notebooks to run:** - `Income_data_mean.ipynb`
  - `Income_data_different_unlabeled_sample_size.ipynb`
- **Outputs:** CSV results and PDFs are saved to `./census_income_data/results/`

### 3) BlogFeedback Data Analysis
- **Location:** `./blogfeedback_data/`
- **Important: run in the following order:**
  1. **Pre-processing:** Run `data_process.ipynb` first. This step generates processed predictions, training data, and test data used by the main analysis.
  2. **Main analysis:** Run `blogfeedback_data_mean.ipynb`
- **Outputs:** CSV results and PDFs are saved to `./blogfeedback_data/results/`

---

## Part 2: Conditional Quantile Inference

All files for quantile inference are located in the `Conditional_quantile_inference/` directory.

**Core File:**
- `conditional_quantile_functions.py` — base functions for conditional quantile inference

### 1) Simulation Experiments
- **Notebook to run:** `Simu_conditional_quantile_new.ipynb`
- **Plotting (optional):** `plot_simu_qunatile.ipynb`
- **Outputs:** CSV results are saved to `./results/` and PDFs (e.g., `simu_quantile_rmse_coverage_width_3x3.pdf`) are generated in the root of this module.

### 2) Census Income Data Analysis
- **Location:** `./census_income_data/`
- **Notebook to run:** `Income_data_conditional_quantile_new.ipynb`
- **Plotting (optional):** `income_data_plot.ipynb`
- **Outputs:** CSV results and PDFs are saved to `./census_income_data/results/`

---

## Output Locations Summary

Outputs are handled independently within their respective inference folders:
- **Simulation:** `[Inference_Type]/results/`
- **Census Income:** `[Inference_Type]/census_income_data/results/`
- **BlogFeedback (Mean only):** `Conditional_mean_inference/blogfeedback_data/results/`
