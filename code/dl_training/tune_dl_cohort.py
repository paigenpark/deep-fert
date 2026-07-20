"""Hyperparameter tuning for the DL cohort fertility model, across data variants.

Two axes are searched here:

  1. Hyperparameters  (units, layers, dropout, learning rate, embedding dims) --
     the model knobs. Sampled by random search.

  2. Data variants    (Lexis shape: square vs parallelogram; amount of training
     data) -- the thing under study. Each variant is tuned SEPARATELY so the final
     comparison reflects each data setting at its own best-effort config, rather
     than confounding "worse data" with "config tuned for a different setting".

For every trial we score a config by its mean PHASE-1 temporal-holdout validation
loss across a subset of jump-off years (TUNING_JOYS). That holdout is leak-free:
the post-JOY forecast horizon (the true test set) is never touched here. Tune on
validation only; keep the forecast horizon for final evaluation.

Outputs:
  - tuning_results.csv           : one row per (variant, trial) with config + score
  - best_config_<variant>.json   : the winning config for each variant

The forecast script (train_dl_cohort_with_age_limits.py) consumes
best_config_<variant>.json to generate and save that variant's forecasts.

Run in the background from the repo root, e.g.:
    nohup python code/dl_training/tune_dl_cohort.py > tune_cohort.log 2>&1 &
"""

import json
import os
import random
import signal
from dataclasses import dataclass, field, asdict

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
import tensorflow as tf

import training_functions

# Resolve paths relative to this script so it runs from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../data"))
RESULTS_CSV = os.path.join(SCRIPT_DIR, "tuning_results.csv")


def best_config_path(variant_name):
    return os.path.join(SCRIPT_DIR, f"best_config_{variant_name}.json")


# Data source per Lexis shape. Both files are already in model format
# [geo, year, age, rate] and period-indexed by calendar year; they differ only in
# how the rate is measured. The parallelogram file is produced by
# code/data_preparation/split_vv_data.py.
DATA_FILES = {
    "square": "asfr_1950_to_2023.txt",           # Lexis squares, completed age (ACY)
    "parallelogram": "asfrVV_1950_to_2023.txt",  # vertical parallelograms (F2), ARDY
}


# ---------------------------------------------------------------------------
# Fixed pipeline settings (shared with the forecast script)
# ---------------------------------------------------------------------------
# Year normalization lives in training_functions (YEAR_MIN/YEAR_MAX,
# normalize_year) and is applied inside get_data, so it isn't re-declared here.
GAP = 5  # temporal-holdout width (years) for leak-free model selection
STEPS_RATIO = 4.74
BATCH_SIZE = 256

# Tune on a SUBSET of jump-off years to keep each trial cheap. Lock the winning
# config, then let the forecast script run all six JOYs.
TUNING_JOYS = [2005]  # shakedown: single JOY. Full run: [2000, 2005]

# How many random configs to try per variant.
N_TRIALS = 5  # shakedown value. Full run: 20 (or 30-100 for a thorough search)
SEED = 42


# ---------------------------------------------------------------------------
# Hyperparameter search space  (random search samples from these)
# ---------------------------------------------------------------------------
def sample_hparams(rng):
    """Draw one hyperparameter config. Ranges centered on the current defaults;
    widen/narrow these as you learn which knobs matter."""
    return {
        "units": int(rng.choice([32, 64, 128])),
        "n_layers": int(rng.choice([2, 3, 4, 5])),
        "dropout": float(rng.choice([0.0, 0.1, 0.2, 0.3])),
        "age_embed_dim": int(rng.choice([3, 5, 8])),
        "geo_embed_dim": int(rng.choice([3, 5, 8])),
        # log-uniform learning rate in [1e-4, 3e-3]
        "learning_rate": float(10 ** rng.uniform(-4, np.log10(3e-3))),
    }


# ---------------------------------------------------------------------------
# Data variants  (the axis under study)
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    lexis_shape: str = "square"     # "square" (RR/ACY) or "parallelogram" (VV/ARDY)
    lograte: bool = False           # raw-rate (sigmoid) vs log-rate (linear) output
    age_min: int = 15
    age_max: int = 44
    # "Amount of training data": length in years of the training window ending at
    # each jump-off year (JOY). e.g. 30 -> train on years (JOY-30, JOY]. None =
    # unlimited (go back as far as the data goes). Because the window is anchored
    # to the JOY, it is applied per-JOY in score_one_joy, not at load time.
    train_window: int | None = None
    # Temporal split points used for tuning (calendar years -- both Lexis shapes
    # are period-indexed, so these are the same for square and parallelogram).
    tuning_joys: list = field(default_factory=lambda: list(TUNING_JOYS))


# Variants to compare. Each is tuned independently and gets its own
# best_config_<name>.json. Two axes are crossed here:
#   - Lexis shape:   square (RR) vs parallelogram (VV)
#   - Amount of data: 30 / 35 / 40 / unlimited years of history up to each JOY
# That's 2 x 4 = 8 variants. Trim this list (or DATA_WINDOWS) if that's more
# tuning than you want -- each variant runs N_TRIALS x len(tuning_joys) fits.
DATA_WINDOWS = [None]  # shakedown: unlimited only (2 variants). Full run: [30, 35, 40, None]


def _window_tag(w):
    return "unlimited" if w is None else f"{w}yr"


VARIANTS = [
    Variant(name=f"{shape}_{_window_tag(w)}", lexis_shape=shape, train_window=w)
    for shape in ("square", "parallelogram")
    for w in DATA_WINDOWS
]


def load_variant_data(variant):
    """Load the variant's Lexis-shape data file (square RR or parallelogram VV),
    already in model format [geo, year, age, rate], and apply the age filter.

    Both shapes are period-indexed by calendar year, so no feature transform is
    needed -- they differ only in how the rate is measured (ACY vs ARDY). The
    train_window (amount-of-data) limit is applied per-JOY in score_one_joy, since
    it depends on the jump-off year.
    """
    if variant.lexis_shape not in DATA_FILES:
        raise ValueError(f"Unknown lexis_shape: {variant.lexis_shape}")
    raw = np.loadtxt(os.path.join(DATA_DIR, DATA_FILES[variant.lexis_shape]))

    age_mask = (raw[:, 2] >= variant.age_min) & (raw[:, 2] <= variant.age_max)
    return raw[age_mask]


def valid_country_mask(data, age_min, age_max):
    """Countries that have every age in [age_min, age_max] (matching R)."""
    geo_dim = int(data[:, 0].max()) + 1
    required_ages = set(range(age_min, age_max + 1))
    valid = []
    for c in range(geo_dim):
        c_data = data[data[:, 0] == c]
        if len(c_data) == 0:
            continue
        if not (required_ages - set(c_data[:, 2].astype(int))):
            valid.append(c)
    return geo_dim, valid


def score_one_joy(data, geo_dim, valid_countries, joy, hparams, lograte, train_window):
    """Train one leak-free refit model for a single jump-off year and return the
    phase-1 temporal-holdout validation loss (the tuning signal)."""
    in_valid = np.isin(data[:, 0].astype(int), valid_countries)

    # Lower bound of the training window: train on years (after, joy]. None (or
    # unlimited) -> use all available history. The window limits how far back
    # training goes but leaves the temporal-holdout split (joy-GAP, joy] intact.
    after = -np.inf if train_window is None else joy - train_window

    train_full = data[(data[:, 1] > after) & (data[:, 1] <= joy) & in_valid]
    train_sub = data[(data[:, 1] > after) & (data[:, 1] <= joy - GAP) & in_valid]
    val_temporal = data[(data[:, 1] > joy - GAP) & (data[:, 1] <= joy) & in_valid]

    if train_sub.shape[0] == 0 or val_temporal.shape[0] == 0:
        return None

    steps_sub = max(1, int(train_sub.shape[0] * STEPS_RATIO / BATCH_SIZE))
    steps_full = max(1, int(train_full.shape[0] * STEPS_RATIO / BATCH_SIZE))

    train_sub_prepped = training_functions.prep_data(
        train_sub, mode="train", changeratetolog=lograte)
    val_temporal_prepped = training_functions.prep_data(
        val_temporal, mode="test", changeratetolog=lograte)
    train_full_prepped = training_functions.prep_data(
        train_full, mode="train", changeratetolog=lograte)

    # Same seeds each trial so scores differ only by config, not RNG.
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    _, _, best_val = training_functions.run_deep_model_refit(
        train_sub_prepped, val_temporal_prepped, train_full_prepped, geo_dim,
        epochs=50,
        steps_per_epoch_sub=steps_sub,
        steps_per_epoch_full=steps_full,
        lograte=lograte,
        hparams=hparams,
    )
    return best_val


def score_config(variant, data, geo_dim, valid_countries, hparams):
    """Mean phase-1 validation loss across the variant's tuning JOYs."""
    losses = []
    for joy in variant.tuning_joys:
        loss = score_one_joy(data, geo_dim, valid_countries, joy,
                             hparams, variant.lograte, variant.train_window)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def append_result(row, write_header):
    """Append one trial row to the results CSV (kept dependency-light)."""
    import csv
    mode = "w" if write_header else "a"
    with open(RESULTS_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def tune_variant(variant, write_header):
    """Run random search for one variant; write its best_config JSON."""
    print(f"\n{'#' * 60}")
    print(f"# Tuning variant: {variant.name}  ({variant.lexis_shape}, "
          f"window={variant.train_window}, ages {variant.age_min}-{variant.age_max}, "
          f"lograte={variant.lograte})")
    print(f"{'#' * 60}")

    rng = np.random.default_rng(SEED)
    data = load_variant_data(variant)
    geo_dim, valid_countries = valid_country_mask(data, variant.age_min, variant.age_max)
    print(f"Rows: {data.shape[0]}, geo_dim: {geo_dim}, "
          f"valid countries: {len(valid_countries)}, tuning JOYs: {variant.tuning_joys}")

    best = {"score": float("inf"), "hparams": None, "trial": None}

    for trial in range(N_TRIALS):
        hparams = sample_hparams(rng)
        score = score_config(variant, data, geo_dim, valid_countries, hparams)
        print(f"[{variant.name}] trial {trial:>2}: score={score:.6f}  {hparams}")

        append_result(
            {"variant": variant.name, "trial": trial, "score": score, **hparams},
            write_header=write_header,
        )
        write_header = False  # only the very first row writes the header

        if score < best["score"]:
            best = {"score": score, "hparams": hparams, "trial": trial}

    # Persist the winning config (variant fields + best hparams + its score).
    best_config = {
        "variant": asdict(variant),
        "score": best["score"],
        "trial": best["trial"],
        "hparams": best["hparams"],
    }
    with open(best_config_path(variant.name), "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"[{variant.name}] BEST trial {best['trial']} "
          f"(score={best['score']:.6f}) -> {best_config_path(variant.name)}")

    return write_header


def main():
    write_header = not os.path.exists(RESULTS_CSV)
    for variant in VARIANTS:
        write_header = tune_variant(variant, write_header)
    print("\nAll variants tuned.")
    print(f"Trial log: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
