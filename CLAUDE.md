# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**deep-fert** is a deep learning-based fertility rate prediction model. It uses TensorFlow/Keras neural networks to forecast age-specific fertility rates (ASFR) across different geographical regions over time.

## Setup and Dependencies

- **Python Version**: 3.13 (see `.python-version`)
- **Package Manager**: `uv` (uses `pyproject.toml` and `uv.lock`)
- **Virtual Environment**: `.venv/`

### Key Dependencies

- **TensorFlow 2.20.0** - Deep learning framework (primary)
- **NumPy 2.4.2** - Numerical computing
- **Pandas 3.0.0** - Data manipulation
- **Matplotlib 3.10.8** - Visualization
- **pyreadr 0.5.4** - Reading R data files
- **rpy2 3.6.4** - R integration
- **ipykernel 7.1.0** - Jupyter support

### Install Dependencies

```bash
uv sync
```

To add a new dependency:
```bash
uv add package_name
```

## Code Architecture

### Directory Structure

```
code/
├── training_functions.py      # Core ML functions (data prep, model creation, training)
├── evaluation_functions.py    # Evaluation and error calculation functions
├── data_preparation/          # Scripts to process and split raw data
│   ├── split_period_data.py   # Splits period-based ASFR data
│   ├── split_cohort_data.py   # Splits cohort-based ASFR data
│   └── split_tri_data.py      # Splits triangle-based ASFR data
├── benchmark_models/          # Stores trained model files
│   ├── period/                # Period-based models
│   └── pnas_cohort/           # Cohort-based models
├── train_dl_models.ipynb      # Main training notebook (period)
├── train_dl_models_tri.ipynb  # Triangle model training
├── eval_figures.ipynb         # Period model evaluation and visualization
├── eval_figures_cohort.ipynb  # Cohort model evaluation and visualization
└── read_asfr_data.ipynb       # Data loading and exploration

data/                          # Data files (gitignored)
models/                        # Pre-trained models
```

### Core Components

#### `training_functions.py`
- **`get_data()`** - Fetches individual data points with optional log transformation
- **`prep_data()`** - Converts raw arrays into TensorFlow datasets with batching and prefetching
- **`create_model()`** - Builds the main neural network with embeddings for age/geography and dense hidden layers
- **`create_log_model()`** - Similar to `create_model()` but for log-transformed rates (no sigmoid output)
- **`run_deep_model()`** - Trains model with early stopping and learning rate reduction callbacks

Model architecture uses:
- Embedding layers for categorical features (age: 55 dims→5 dims, geography: variable→5 dims)
- 3 hidden dense layers (64 units, ReLU) with LayerNormalization and 10% Dropout
- Residual connection from input to output layers
- Final output layer with sigmoid activation (for raw rates) or linear (for log rates)

#### `evaluation_functions.py`
- **`calculate_error()`** - Computes MSE between forecasted and actual rates
- **`calculate_error_by_category()`** - Breaks down MSE by feature (year, age, geography)
- Handles log-transformed rates by converting zeros to 1e-5 to avoid log(0)

### Data Format

ASFR data arrays have 4 columns:
```
[geography_index, year, age, rate]
```

Data is split chronologically:
- **Training**: 1950-2005
- **Validation**: 2005-2015
- **Final Test**: 2015-2019 (or 2022-2025 for newer splits)

### Model Training

Key hyperparameters (in notebooks):
- **Batch size**: 256
- **Embedding dimensions**: 5 for both age and geography
- **Hidden layers**: 3 × 64 units
- **Early stopping**: patience=10 epochs
- **Learning rate reduction**: factor=0.25, patience=3 epochs
- **Loss function**: MSE

## Common Development Tasks

### Running Data Preparation

```bash
cd code/data_preparation
python split_period_data.py    # Process period data
python split_cohort_data.py    # Process cohort data
python split_tri_data.py       # Process triangle data
```

### Training Models

Open and run the relevant Jupyter notebook:
- `code/train_dl_models.ipynb` - Period-based model
- `code/train_dl_models_tri.ipynb` - Triangle model

Notebooks import functions from `training_functions.py` and handle:
1. Loading data splits
2. Creating datasets via `prep_data()`
3. Training via `run_deep_model()`
4. Saving trained models to `benchmark_models/`

### Evaluating Models

Open and run:
- `code/eval_figures.ipynb` - Evaluate period models (uses `eval_figures_cohort.ipynb` for cohort)
- Notebooks use `evaluation_functions.py` to calculate and visualize errors

### Testing / Iteration

Currently, testing is done through Jupyter notebooks. To test individual functions interactively:

```python
# In a Python shell or notebook
from code.training_functions import create_model, run_deep_model
from code.evaluation_functions import calculate_error

# Test model creation
model = create_model(geo_dim=195)  # 195 countries in HMD data
```

## Important Notes

- **Data directory**: `/data` is gitignored; raw ASFR files are stored there (e.g., `asfr/asfrRR.txt`)
- **Trained models**: Stored in `code/benchmark_models/` and `/models` directories
- **Geo index mapping**: Saved as `/data/geos_key.npy` by data preparation scripts (country name → numeric index)
- **Rate constraints**: Rates are capped at 1.0 during data loading (`if rate > 1: rate = 1`)
- **Log transformation**: When `changeratetolog=True`, rates are converted via `log(max(rate, 1e-5))`
- **R integration**: Some benchmark models use R (via rpy2); check `/code/benchmark_models/pnas_cohort/` for R scripts

## Environment and Tools

- **Jupyter kernels**: Managed via ipykernel
- **Notebooks**: Save outputs but notebooks themselves are tracked in git
- **Model files**: TensorFlow saved models (directory format with assets, variables, saved_model.pb)

## Recent Development Context

Recent commits indicate work on:
- Triangle model development (`train_dl_models_tri.ipynb`)
- First period model training runs
- Lee-Carter benchmark implementation rework
- Data loading pipelines

Key branches and milestones are tracked in git history.
