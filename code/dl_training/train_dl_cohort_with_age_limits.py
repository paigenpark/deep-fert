"""Deep Learning Cohort Fertility Prediction.

Trains the non-log DL model with rotating jump-off years (1985-2010) and converts
predictions to cohort rates for comparison with the Lee (1993) benchmark.

Aligned with R Lee-Carter approach:
- Ages 15-44 (matching `age1=15, age2=44`)
- Country filtering: skips countries missing any age in 15-44
- Year gap checks: skips country/JOY combos with gaps in observed years
- Forecast horizon: JOY + 30 years (matching `len=30`)

Run in the background from the repo root, e.g.:
    nohup python code/dl_training/train_dl_cohort_with_age_limits.py > train_cohort.log 2>&1 &
"""

import json
import os
import random
import signal

# Run cleanly when launched in the background. Under the shell's job control, a
# backgrounded process that touches the controlling terminal is stopped with
# SIGTTOU and shows up as "suspended (tty output)" -- TensorFlow/Keras can do
# this during import or fit even with stdout/stderr redirected to a log file.
# Ignoring the terminal-stop signals lets those terminal operations proceed
# instead of suspending the job; it's safe because all real output already goes
# to the redirected log. Set before importing TF so an import-time touch is
# covered too. (SIG_IGN is a no-op if there's no controlling terminal at all.)
signal.signal(signal.SIGTTOU, signal.SIG_IGN)
signal.signal(signal.SIGTTIN, signal.SIG_IGN)

import numpy as np
import pandas as pd
import tensorflow as tf

import training_functions

# Resolve paths relative to this script so it runs from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../models"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Which tuned variant to forecast. If best_config_<VARIANT_NAME>.json exists
# (written by tune_dl_cohort.py), its hyperparameters and train_window are used;
# otherwise this falls back to model defaults and unlimited data. Overridable via
# the VARIANT_NAME env var so the same script can forecast each tuned variant.
VARIANT_NAME = os.environ.get("VARIANT_NAME", "square_unlimited")

AGE_MIN = 15
AGE_MAX = 44
JUMP_OFF_YEARS = [1985, 1990, 1995, 2000, 2005, 2010]
# Latest permitted jump-off year (JOYs beyond this are skipped in
# validate_countries). Shares training_functions.YEAR_MAX -- the top of the
# year-normalization anchor and the last "known-data" year are the same 2015.
YEAR_MAX = training_functions.YEAR_MAX
FORECAST_LEN = 30  # Match R's len=30
GAP = 5  # temporal-holdout width (years) for leak-free model selection
STEPS_RATIO = 4.74
BATCH_SIZE = 256
METHOD_NAME = "DL_NonLog"
SEED = 42

PRED_OUT = os.path.join(DATA_DIR, f"dl_forecasts_{VARIANT_NAME}.csv")
OBS_OUT = os.path.join(DATA_DIR, f"dl_obs_cohort_{VARIANT_NAME}.csv")


# Data source per Lexis shape (mirrors DATA_FILES in tune_dl_cohort.py).
DATA_FILES = {
    "square": "asfr_1950_to_2023.txt",           # Lexis squares, completed age (ACY)
    "parallelogram": "asfrVV_1950_to_2023.txt",  # vertical parallelograms (F2), ARDY
}


def load_tuned_config(variant_name):
    """Load a variant's tuned config, returning (hparams, train_window, lexis_shape).

    hparams is None when no config file exists (falls back to model defaults).
    train_window is the amount-of-data window in years (None = unlimited), and
    lexis_shape selects the input file -- both read from the variant block written
    by tune_dl_cohort.py.
    """
    path = os.path.join(SCRIPT_DIR, f"best_config_{variant_name}.json")
    if not os.path.exists(path):
        print(f"No tuned config at {path}; using model defaults, unlimited square data.")
        return None, None, "square"
    with open(path) as f:
        cfg = json.load(f)
    hparams = cfg.get("hparams")
    variant = cfg.get("variant", {})
    train_window = variant.get("train_window")
    lexis_shape = variant.get("lexis_shape", "square")
    print(f"Loaded tuned config for '{variant_name}' (score={cfg.get('score')}): "
          f"lexis_shape={lexis_shape}, train_window={train_window}, hparams={hparams}")
    return hparams, train_window, lexis_shape


def load_data(lexis_shape="square"):
    """Load the Lexis-shape ASFR data file and filter to ages 15-44."""
    asfr_all_raw = np.loadtxt(os.path.join(DATA_DIR, DATA_FILES[lexis_shape]))

    age_mask = (asfr_all_raw[:, 2] >= AGE_MIN) & (asfr_all_raw[:, 2] <= AGE_MAX)
    asfr_all = asfr_all_raw[age_mask]

    print(f"Data ({lexis_shape}): {DATA_FILES[lexis_shape]}")
    print(f"Raw: {asfr_all_raw.shape}")
    print(f"Filtered (ages {AGE_MIN}-{AGE_MAX}): {asfr_all.shape}")
    print(f"Years: {int(asfr_all[:, 1].min())}-{int(asfr_all[:, 1].max())}")
    print(f"Ages: {int(asfr_all[:, 2].min())}-{int(asfr_all[:, 2].max())}")
    print(f"Countries: {int(asfr_all[:, 0].max()) + 1}")

    return asfr_all


def validate_countries(asfr_all):
    """Determine valid countries and valid country/JOY combos (matching R)."""
    geo_dim = int(asfr_all[:, 0].max()) + 1
    countries = np.arange(geo_dim)

    # Country validation: skip countries missing any age in 15-44 (matching R)
    required_ages = set(range(AGE_MIN, AGE_MAX + 1))
    valid_countries = []
    country_min_year = {}

    for c in countries:
        c_data = asfr_all[asfr_all[:, 0] == c]
        if len(c_data) == 0:
            continue
        c_ages = set(c_data[:, 2].astype(int))
        missing = required_ages - c_ages
        if missing:
            print(f"Skipping Country {int(c)} - Missing ages: {sorted(missing)}")
            continue
        valid_countries.append(int(c))
        country_min_year[int(c)] = int(c_data[:, 1].min())

    # For each JOY, check for year gaps per country (matching R)
    valid_combos = {}  # joy -> list of valid country indices
    for joy in JUMP_OFF_YEARS:
        valid_for_joy = []
        for c in valid_countries:
            min_yr = country_min_year[c]
            if joy > YEAR_MAX or joy < min_yr:
                continue
            # Check all years from min_data_year to joy are present
            c_data = asfr_all[asfr_all[:, 0] == c]
            available_years = set(c_data[:, 1].astype(int))
            required_years = set(range(min_yr, joy + 1))
            missing_years = required_years - available_years
            if missing_years:
                print(
                    f"Skipping Country {c}, JOY {joy} - "
                    f"Missing years: {sorted(missing_years)[:5]}"
                )
                continue
            valid_for_joy.append(c)
        valid_combos[joy] = valid_for_joy
        print(f"JOY {joy}: {len(valid_for_joy)} valid countries")

    print(f"\ngeo_dim: {geo_dim}")
    print(f"Valid countries: {len(valid_countries)}")
    print(f"Jump-off years: {JUMP_OFF_YEARS}")

    return geo_dim, valid_countries, valid_combos


def train_and_forecast(asfr_all, geo_dim, valid_countries, valid_combos,
                       hparams=None, train_window=None):
    """Train a model per jump-off year and produce period forecasts.

    train_window limits training to the most recent `train_window` years up to
    each JOY (None = unlimited), matching the amount-of-data variant being run.
    """
    ages = np.arange(AGE_MIN, AGE_MAX + 1)

    predicted_rates = {}
    observed_rates = {}
    best_epochs = {}

    for joy in JUMP_OFF_YEARS:
        print(f"\n{'=' * 50}")
        print(f"Jump-off year: {joy}")
        print(f"{'=' * 50}")

        in_valid = np.isin(asfr_all[:, 0].astype(int), valid_countries)

        # Lower bound of the training window: train on years (after, joy].
        # None -> unlimited history. Must match the tuning-time window so the
        # forecast model is trained on the same amount of data it was tuned for.
        after = -np.inf if train_window is None else joy - train_window

        # 1. Full training data: valid countries, ages 15-44, years (after, joy]
        train_full = asfr_all[(asfr_all[:, 1] > after) & (asfr_all[:, 1] <= joy) & in_valid]
        print(f"Full training rows (window={train_window}, <= {joy}): {train_full.shape[0]}")

        # 2. Temporal holdout for leak-free model selection:
        #    phase-1 trains on years (after, joy-GAP], validates on (joy-GAP, joy].
        #    Post-joy (graded) data is never touched during training.
        train_sub = asfr_all[(asfr_all[:, 1] > after) & (asfr_all[:, 1] <= joy - GAP) & in_valid]
        val_temporal = asfr_all[
            (asfr_all[:, 1] > joy - GAP) & (asfr_all[:, 1] <= joy) & in_valid
        ]
        print(f"  Phase-1 train rows (<= {joy - GAP}): {train_sub.shape[0]}")
        print(f"  Phase-1 val rows ({joy - GAP + 1}-{joy}): {val_temporal.shape[0]}")

        if train_sub.shape[0] == 0 or val_temporal.shape[0] == 0:
            print("Insufficient data for temporal holdout, skipping")
            continue

        # 3. Scale steps_per_epoch to each dataset's size
        steps_sub = int(train_sub.shape[0] * STEPS_RATIO / BATCH_SIZE)
        steps_full = int(train_full.shape[0] * STEPS_RATIO / BATCH_SIZE)
        print(f"Steps per epoch: phase-1={steps_sub}, phase-2={steps_full}")

        # 4. Prep datasets
        train_sub_prepped = training_functions.prep_data(
            train_sub, mode="train", changeratetolog=False
        )
        val_temporal_prepped = training_functions.prep_data(
            val_temporal, mode="test", changeratetolog=False
        )
        train_full_prepped = training_functions.prep_data(
            train_full, mode="train", changeratetolog=False
        )

        # 5. Set seeds and run two-phase leak-free refit:
        #    phase 1 picks the best epoch on the temporal holdout, phase 2 refits on
        #    all data <= joy for that many epochs with no validation peek.
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

        model, best_epoch, best_val = training_functions.run_deep_model_refit(
            train_sub_prepped,
            val_temporal_prepped,
            train_full_prepped,
            geo_dim,
            epochs=50,
            steps_per_epoch_sub=steps_sub,
            steps_per_epoch_full=steps_full,
            lograte=False,
            hparams=hparams,
        )
        best_epochs[joy] = best_epoch
        print(
            f"Best epoch (temporal holdout): {best_epoch}, "
            f"phase-1 val loss: {best_val:.6f}"
        )

        # 6. Forecast grid: valid countries for this JOY, ages 15-44,
        #    years JOY+1 to JOY+FORECAST_LEN
        forecast_year_max = joy + FORECAST_LEN
        forecast_years = np.arange(joy + 1, forecast_year_max + 1)
        valid_c_list = sorted(valid_combos[joy])
        grid = np.array(
            [(c, y, a) for c in valid_c_list for y in forecast_years for a in ages]
        )
        print(
            f"Forecast grid: {grid.shape[0]} ({len(valid_c_list)} countries, "
            f"{len(forecast_years)} years, {len(ages)} ages)"
        )

        # 7. Predict using the shared year normalization (same mapping as training)
        forecast_features = (
            tf.convert_to_tensor(
                training_functions.normalize_year(grid[:, 1]), dtype=tf.float32
            ),
            tf.convert_to_tensor(grid[:, 2], dtype=tf.float32),
            tf.convert_to_tensor(grid[:, 0], dtype=tf.float32),
        )
        preds = model.predict(forecast_features).flatten()

        # 8. Store predicted period rates
        pred_df = pd.DataFrame(
            {
                "Country": grid[:, 0].astype(int),
                "Year": grid[:, 1].astype(int),
                "Age": grid[:, 2].astype(int),
                "Rate": preds,
            }
        )
        predicted_rates[joy] = pred_df

        # 9. Store observed period rates for valid countries (year <= joy)
        valid_c_set = set(valid_c_list)
        obs_mask = np.isin(train_full[:, 0].astype(int), list(valid_c_set))
        obs_data = train_full[obs_mask]
        obs_df = pd.DataFrame(
            {
                "Country": obs_data[:, 0].astype(int),
                "Year": obs_data[:, 1].astype(int),
                "Age": obs_data[:, 2].astype(int),
                "Rate": obs_data[:, 3],
            }
        )
        observed_rates[joy] = obs_df

        # 10. Save model
        model_path = os.path.join(
            MODELS_DIR, f"dl_cohort_{VARIANT_NAME}_refit_joy{joy}.keras"
        )
        model.save(model_path)
        print(f"Model saved: {model_path}")

    print("\nAll jump-off years complete!")
    print(f"Best epochs per JOY: {best_epochs}")

    return predicted_rates, observed_rates


def period_to_cohort(period_df):
    df = period_df.copy()
    df["Year"] = df["Year"] - df["Age"]  # cohort_birth_year = period_year - age
    return df


def build_predicted_cohort(predicted_rates, valid_combos):
    """Convert DL-predicted period rates to cohort rates."""
    pred_cohort_dfs = []

    for joy in JUMP_OFF_YEARS:
        if joy not in predicted_rates:
            continue

        # DL-predicted period rates (JOY+1 to JOY+30)
        pred_period = predicted_rates[joy]

        # Convert to cohort
        cohort_df = period_to_cohort(pred_period)

        # Add metadata columns
        cohort_df["JumpOffYear"] = joy
        cohort_df["Method"] = METHOD_NAME
        cohort_df["Key"] = cohort_df["Method"] + "_" + cohort_df["Country"].astype(str)

        pred_cohort_dfs.append(cohort_df)

    predCASFR = pd.concat(pred_cohort_dfs, ignore_index=True)
    print(f"predCASFR shape: {predCASFR.shape}")
    print(f"Columns: {list(predCASFR.columns)}")

    return predCASFR


def build_observed_cohort(asfr_all, valid_combos):
    """Convert observed period rates to cohort rates for each JOY."""
    obs_cohort_dfs = []

    # Full observed period data (ages 15-44, already filtered)
    all_obs_df = pd.DataFrame(
        {
            "Country": asfr_all[:, 0].astype(int),
            "Year": asfr_all[:, 1].astype(int),
            "Age": asfr_all[:, 2].astype(int),
            "Rate": asfr_all[:, 3],
        }
    )

    for joy in JUMP_OFF_YEARS:
        valid_c_set = set(valid_combos[joy])

        # Filter to valid countries for this JOY (matching R's per-country obsCASFR)
        obs_filtered = all_obs_df[all_obs_df["Country"].isin(valid_c_set)]

        # Convert all observed period data to cohort
        cohort_df = period_to_cohort(obs_filtered)

        # Add metadata columns
        cohort_df["JumpOffYear"] = joy
        cohort_df["Method"] = METHOD_NAME
        cohort_df["Key"] = cohort_df["Method"] + "_" + cohort_df["Country"].astype(str)

        obs_cohort_dfs.append(cohort_df)

    obsCASFR = pd.concat(obs_cohort_dfs, ignore_index=True)
    print(f"obsCASFR shape: {obsCASFR.shape}")
    print(f"Columns: {list(obsCASFR.columns)}")

    return obsCASFR


def main():
    hparams, train_window, lexis_shape = load_tuned_config(VARIANT_NAME)
    asfr_all = load_data(lexis_shape)
    geo_dim, valid_countries, valid_combos = validate_countries(asfr_all)
    predicted_rates, observed_rates = train_and_forecast(
        asfr_all, geo_dim, valid_countries, valid_combos,
        hparams=hparams, train_window=train_window,
    )

    predCASFR = build_predicted_cohort(predicted_rates, valid_combos)
    obsCASFR = build_observed_cohort(asfr_all, valid_combos)

    predCASFR.to_csv(PRED_OUT, index=False)
    obsCASFR.to_csv(OBS_OUT, index=False)
    print(f"Saved: {PRED_OUT} ({predCASFR.shape[0]} rows)")
    print(f"Saved: {OBS_OUT} ({obsCASFR.shape[0]} rows)")


if __name__ == "__main__":
    main()
