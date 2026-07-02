# Compressor Fouling Modeling

Bayesian and machine learning approaches for detecting and quantifying **compressor fouling** — the buildup of deposits inside a compressor that degrades performance over time.

This project builds a **digital twin** of a healthy compressor from historical sensor data, then detects anomalies (fouling) by tracking deviations between predicted and actual outlet pressure. It combines classical ML (ElasticNet, splines with Optuna hyperparameter tuning), Bayesian regression (PyMC + Nutpie MCMC), and probabilistic uncertainty quantification for fouling detection via CUSUM control charts.

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)
- [License](#license)

---

## Problem Overview

Compressor fouling reduces efficiency and can lead to unplanned downtime if undetected. The key insight: a model trained on data from a known healthy (baseline) period can predict expected outlet pressure during normal operation. When fouling develops, the residuals (actual − predicted) drift systematically, and a **CUSUM** (cumulative sum) control chart detects the onset.

A practical challenge: the compressor undergoes periodic maintenance shut-ins (outlet pressure setpoint = 0) where the unit is offline. Standard CUSUM would accumulate residuals during these off periods, inflating the statistic and triggering false alarms on restart. This project uses a **memoryless CUSUM** that resets during shut-ins, ensuring detection only reflects genuine fouling-related drift during active operation.

The project explores two complementary approaches:

| Approach | Description | Key Strength |
|----------|-------------|--------------|
| **Frequentist ML** | ElasticNet / Spline + ElasticNet pipeline with Optuna hyperparameter tuning | Fast, reproducible, well-understood |
| **Bayesian** | PyMC models with tunable priors, hierarchical structures, and Student-T likelihoods | Full posterior uncertainty, calibration diagnostics, probabilistic CUSUM |

---

## Features

### Data Preparation
- Automatic baseline period detection via rolling statistics
- Shut-in period flagging (setpoint = 0)
- Feature engineering: temperature rise, pressure × flow interactions

### Machine Learning (Frequentist)
- `ElasticNet` and `SplineTransformer + ElasticNet` pipelines
- **Optuna** hyperparameter optimization with time-series-aware cross-validation
- Learning curves and partial dependence plots

### Bayesian Modeling (PyMC + Nutpie)
- Configurable priors: Normal or Laplace (sparse)
- Hierarchical or non-hierarchical noise, intercept, and coefficient structures
- Normal or Student-T likelihoods (robust to outliers)
- JAX-backed NUTS sampling via **Nutpie** for fast MCMC
- Model comparison via ELPD (Pareto-smoothed importance sampling LOO)

### Anomaly Detection
- **Memoryless CUSUM** — a cumulative sum control chart that resets to zero during maintenance shut-in periods (outlet pressure setpoint = 0). This prevents residual accumulation when the compressor is offline, avoiding false positives when normal operation resumes.
- Configurable drift and threshold parameters
- Fouling onset date detection with signal-to-noise ratio (SNR)
- **Probabilistic CUSUM** — Monte Carlo CUSUM across full posterior draws, with exceedance probabilities and uncertainty bands

### Model Diagnostics & Calibration
- **LOO-PIT calibration curves** with dual uncertainty bands (expected sampling variation + Bayesian bootstrap)
- Bayesian R², MAE, RMSE, LOO-adjusted metrics
- Posterior predictive checks (ECDF, KDE, Q-Q plots)
- Variance decomposition and per-setpoint noise evaluation

### Pressure Regime Modeling
- PyMC Gaussian mixture model for identifying distinct operating regimes
- Component assignment and regime-specific diagnostics

---

## Architecture

```
data/raw/ds_compressor_data.csv
         │
         ▼
  [Feature Engineering]
  Temperature_Rise, Inlet_Pressure_x_Flow, Temp_x_Flow, ...
         │
         ▼
  [Baseline / Shut-in Masking]
         │
         ├──► X_baseline, y_baseline   (healthy period only)
         └──► X_full, y_full           (full time series)
         │
         ├──► APPROACH 1: Frequentist ML
         │       train_model() → ElasticNet / Spline + Optuna
         │
         └──► APPROACH 2: Bayesian
                 build_bayesian_model() → PyMC + Nutpie MCMC
         │
         ▼
  [Predict over full time series]
         │
         ▼
  [Residuals = y_actual − y_pred]
         │
         ├──► CUSUM Anomaly Detection (frequentist)
         │       predict_fouling_onset() → onset date + SNR
         │
         └──► Probabilistic CUSUM (Bayesian)
                 CUSUM per posterior draw → exceedance probability
         │
         ▼
  [Diagnostics & Calibration]
  LOO-PIT calibration curves
  Posterior predictive checks
  Model comparison (ELPD)
```

---

## Installation

### Prerequisites
- Python ≥ 3.12, < 3.14
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd compressor_fouling_modeling

# Without CUDA (CPU only)
uv sync

# With CUDA support (GPU-accelerated JAX)
uv sync --group cuda
```

### Activate the virtual environment

```bash
source .venv/bin/activate
```

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── ds_compressor_data.csv     # Raw daily sensor data
│   └── processed/                      # Preprocessed feature matrices
│       ├── X_full.csv, y_full.csv
│       ├── X_baseline.csv, y_baseline.csv
│       ├── baseline_mask.csv
│       └── shutin_mask.csv
├── docs/                               # Rendered notebooks & instructions
├── notebooks/
│   ├── compressor_fouling_eda.ipynb                # Exploratory data analysis
│   ├── compressor_fouling_no_uncertanity.ipynb     # Frequentist ML pipeline
│   ├── compressor_fouling_with_uncertanity.ipynb   # Bayesian pipeline
│   └── compressor_pressure_regimes_mixture_model.ipynb # Gaussian mixture model
├── results/                           # Generated plots & figures
├── src/
│   └── compressor_fouling_modeling/
│       ├── __init__.py
│       └── utility.py                 # All model code (6051 lines)
├── tests/
│   └── test_loo_calibration.py        # LOO-PIT calibration & curve tests
├── pyproject.toml                     # Project config & dependencies
├── requirements.txt                   # Pinned dependencies
├── uv.lock                            # uv lockfile
└── README.md
```

---

## Usage

### Run Notebooks (primary workflow)

```bash
jupyter lab
```

Recommended order:

1. **`notebooks/compressor_fouling_eda.ipynb`** — Explore raw data, understand the compressor physics, visualize pressure setpoints and trends.
2. **`notebooks/compressor_fouling_no_uncertanity.ipynb`** — Train ElasticNet / Spline models, apply CUSUM detection, generate the frequentist fouling summary.
3. **`notebooks/compressor_fouling_with_uncertanity.ipynb`** — Build Bayesian models, perform model comparison, compute probabilistic CUSUM with full uncertainty quantification.
4. **`notebooks/compressor_pressure_regimes_mixture_model.ipynb`** — Gaussian mixture model for pressure regime identification.

> The notebook HTML files in `docs/` are large and may not render properly on GitHub. You can view rendered versions at **[https://mghanava.github.io/compressor_fouling_modeling/](https://mghanava.github.io/compressor_fouling_modeling/)** without cloning the repository.

### Run Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (skip MCMC-heavy tests)
pytest tests/ -m "not slow" -v

# Slow tests only (includes MCMC sampling)
pytest tests/ -m slow -v
```

### Lint & Format

```bash
# Lint source code
ruff check src/

# Lint notebooks
ruff check notebooks/*.ipynb

# Auto-fix
ruff check --fix src/
```

### Use as a Library

```python
from compressor_fouling_modeling.utility import (
    train_model,
    build_bayesian_model,
    fit_bayesian_model,
    predict_fouling_onset,
    plot_loo_calibration_curve_with_reference,
)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymc` | Bayesian statistical modeling and MCMC sampling |
| `nutpie` | JAX-backed NUTS sampler (faster than default PyMC) |
| `jax` / `jaxtyping` | GPU-accelerated linear algebra |
| `arviz-base` / `arviz-plots` / `arviz-stats` | Bayesian workflow tools (ELPD, LOO, diagnostics) |
| `optuna` | Hyperparameter optimization for ML models |
| `scikit-learn` | ElasticNet, SplineTransformer, cross-validation |
| `pandas` / `numpy` / `scipy` | Data manipulation and numerical computing |
| `seaborn` / `matplotlib` / `plotly` | Static and interactive visualizations |
| `pytensor` | Computational backend for PyMC |
| `xarray` / `xarray-einstats` | Multi-dimensional MCMC chain arrays |

**Dev dependencies:** `jupyterlab`, `pytest`, `ruff`, `pre-commit`, `nbstripout`, `pydoclint`, `debugpy`, `ipywidgets`

---

## Results

The models produce two key diagnostic plots:

- **`results/fouling_summary.png`** — Frequentist 3-panel: actual vs predicted outlet pressure, residuals, and CUSUM with detected fouling onset.
- **`results/fouling_summary_probabilistic.png`** — Bayesian 4-panel: CUSUM paths across posterior draws, exceedance probabilities, residuals with uncertainty bands.

---

## License

This project is provided for research and educational purposes.
