# Plan: DL Cohort Fertility Training Notebook

## Context
The Lee (1993) benchmark uses rotating jump-off years (1985-2005) and produces cohort fertility rates. The current DL model uses a single fixed train/test split and produces only period rates. To enable a direct comparison, we need a new notebook that trains the non-log DL model with the same rotating jump-off year approach and converts predictions to cohort rates.

## New File: `code/train_dl_cohort.ipynb`

### Cell 1: Markdown header
Title and brief description.

### Cell 2: Imports
`tensorflow`, `numpy`, `pandas`, `os`, `random`

### Cell 3: Import training_functions
Reuse existing `training_functions.py` (no modifications needed).

### Cell 4: Load and concatenate data
- `np.vstack([asfr_training.txt, asfr_test.txt])` → `asfr_all` (92,190 rows)
- Columns: `[Country, Year, Age, Rate]`, years 1950-2015, ages 13-54, 39 countries

### Cell 5: Define constants
- `JUMP_OFF_YEARS = [1985, 1990, 1995, 2000, 2005]`
- Fixed year normalization: `(year - 1950) / (2015 - 1950)` (matches hardcoded value in `training_functions.get_data()` line 23)
- `STEPS_RATIO = 4.74` (derived from original: 1405 * 256 / 75936)
- `METHOD_NAME = "DL_NonLog"`
- `geo_dim = 39`

### Cell 6: Training loop over jump-off years
For each JOY:
1. **Filter training data**: all rows where `Year <= joy`
2. **Scale steps_per_epoch**: `int(train_size * 4.74 / 256)` — ranges from ~1019 (JOY=1985) to ~1405 (JOY=2005)
3. **Build forecast grid**: all `(Country, Age, Year)` combos where `Year > joy` and `Year <= 2015`, ages 13-54
4. **Validation data**: actual data for years > joy (used for early stopping)
5. **Train non-log model** via `training_functions.run_deep_model(..., lograte=False)`
6. **Predict** on forecast grid using same normalization as training
7. **Store** predicted period rates and observed period rates per JOY
8. **Save model** to `../models/dl_cohort_nonlog_joy{joy}.keras`

### Cell 7: Period-to-cohort conversion function
```python
def period_to_cohort(period_df):
    df = period_df.copy()
    df['Year'] = df['Year'] - df['Age']  # cohort_birth_year = period_year - age
    return df
```
This is a 1:1 mapping — each `(age, period_year)` maps to exactly one `(age, cohort_year)`. Matches the R function `asfr_period_to_cohort` from the Lee code.

### Cell 8: Build predCASFR (predicted cohort ASFR)
For each JOY, combine:
- **Observed period rates** (years <= JOY) — actual data, not model fitted values
- **DL-predicted period rates** (years > JOY) — model forecasts

Then convert combined period data to cohort and add metadata columns: `JumpOffYear, Method, Key`.

Output columns match Lee CSV: `[Age, Year, Rate, Country, JumpOffYear, Method, Key]`

### Cell 9: Build obsCASFR (observed cohort ASFR)
For each JOY, convert ALL observed period data (full 1950-2015 dataset) to cohort. The obs data is identical across JOYs but repeated with each JOY label (matches Lee CSV structure).

### Cell 10: Save CSVs
- `../data/dl_forecasts_cohort.csv` (pred cohort)
- `../data/dl_obs_cohort.csv` (obs cohort)

### Cell 11: Verification
- Compare column formats with Lee CSVs
- Trial merge on `['Age', 'Year', 'Country', 'JumpOffYear']` to confirm compatibility with `eval_figures_cohort.ipynb`
- Quick MAE/RMSE sanity check overall and by JOY

## Key Design Decisions
- **Maximizing training data**: all data up to JOY is used (no fixed window)
- **Year normalization**: fixed (1950, 2015) range — already hardcoded in `training_functions.py`, no changes needed
- **predCASFR composition**: observed + forecasted period rates combined before cohort conversion (matches Lee approach)
- **Ages 13-54**: full range from data; evaluation merge handles any filtering needed

## Files Involved
- **Reused (read-only)**: `code/training_functions.py` — `prep_data()`, `create_model()`, `run_deep_model()`
- **Pattern reference**: `code/train_dl_models.ipynb` — data loading, training loop, prediction extraction
- **New file**: `code/train_dl_cohort.ipynb`
- **Output files**: `data/dl_forecasts_cohort.csv`, `data/dl_obs_cohort.csv`
- **Downstream consumer**: `code/eval_figures_cohort.ipynb` — merges on `[Age, Year, Country, JumpOffYear]`

## Verification
1. Run the notebook end-to-end
2. Confirm output CSVs have correct columns matching Lee format
3. Load in `eval_figures_cohort.ipynb` and verify merge produces matched observations
