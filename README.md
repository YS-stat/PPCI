# PPCI: Prediction-Powered Conditional Inference

This repository contains the implementation of **Prediction-Powered Conditional Inference (PPCI)**.
PPCI enables valid uncertainty quantification for **conditional functionals** (e.g., conditional means and quantiles)
in settings with **scarce labeled data**, **rich ublabeled data**,  and an available **black-box predictor**.

---

## Requirements

- **CUDA-enabled GPU** environment
- Core computation is based on **CuPy**

---

## Core Files (Root Directory)

The core logic for PPCI is implemented in:

- `conditional_mean_functions.py` — base functions for conditional mean inference
- `conditional_quantile_functions.py` — base functions for conditional quantile inference

---

## How to Run

The project is organized into **three experimental modules**.

### 1) Simulation Experiments (Root)

**Notebooks to run**
- `Simu_conditional_mean_new.ipynb`
- `Simu_conditional_quantile_new.ipynb`

**Plotting (optional)**
- `plot_simu_mean_qunatile.ipynb`

**Outputs**
- CSV results are saved to: `./results/`
- Figures/PDFs may be generated/updated in the root directory (e.g., `simu_*_3x3.pdf`)

---

### 2) Census Income Data Analysis (`census_income_data/`)

**Location**
- `./census_income_data/`

**Notebooks to run**
- `Income_data_conditional_mean_new.ipynb`
- `Income_data_conditional_quantile_new.ipynb`

**Plotting (optional)**
- `income_data_plot.ipynb`

**Outputs**
- CSV results are saved to: `./census_income_data/results/`

---

### 3) BlogFeedback Data Analysis (`blogfeedback_data/`)

**Location**
- `./blogfeedback_data/`

**Important: run in the following order**

1. **Pre-processing (run first)**
   - `data_process.ipynb`
   - This step generates processed predictions / training data / test data used by the main analysis.

2. **Main analysis**
   - `Conditional_mean_blogfeedback_data_new.ipynb`

**Plotting (optional)**
- `blogfeedback_plot.ipynb`

**Outputs**
- CSV results are saved to: `./blogfeedback_data/results/`

---

## Output Locations Summary

- Simulation: `./results/`
- Census Income: `./census_income_data/results/`
- BlogFeedback: `./blogfeedback_data/results/`
