"""Multi-seed ENSEMBLE training for the DL *square* cohort model.

Why this exists
---------------
A single training run of the tuned `square_unlimited` model has large seed
variance: per-jump-off-year RMSE swings by ~2x across seeds and the mean
per-cell forecast SD is ~0.007. That variance made the jump-off-1995 RMSE
"spike" in the committed single run (spike ratio 1.65) look larger than it
really is -- across 5 fresh seeds the typical ratio is ~1.3, and one seed did
not spike at all. See the seed-variance investigation for details.

This script trains the SAME tuned model under several RNG seeds (identical data,
split, window, and hyperparameters -- only weight init and batch shuffling
differ) and averages the per-cell forecasts into an ensemble. The ensemble is
both more accurate overall (common-grid RMSE 0.01199 vs 0.01247 for the single
run) and a stable series to report, and it yields a per-cell spread band.

It reuses the exact pipeline in ``train_dl_cohort_with_age_limits.py`` (imported,
not duplicated), so every ensemble member is faithful to the committed
single-seed run -- the only change is the seed set via the module's SEED global.

Outputs (../../data)
--------------------
  dl_forecasts_square_ensemble.csv         per-cell MEAN forecast (cohort rates)
  dl_forecasts_square_ensemble_seedsd.csv  per-cell SD across seeds (spread band)
  dl_obs_cohort_square_ensemble.csv        observed cohort rates (seed-independent)
  dl_forecasts_square_seed<K>.csv          each member's forecast, for K in SEEDS

The ensemble file is a drop-in for the eval notebook: register it as
    'DL square': dict(pred='dl_forecasts_square_ensemble',
                      obs='dl_obs_cohort_square_ensemble', ...)

Per-member .keras models are written under a gitignored dir inside ../../data
(they are disposable -- the seed + this script fully reproduce them).

Run from the repo root (long-running; ~15-20 min/seed on CPU, so ~1.5h for 5):
    nohup python code/dl_training/train_dl_cohort_ensemble.py > train_ensemble.log 2>&1 &

Configure the seed set with the SEEDS env var (default "0,1,2,3,4").
"""
import os
import sys

import numpy as np
import pandas as pd

# Import the single-seed pipeline as a module (sibling file). Running a script
# puts its own directory on sys.path, so this resolves without extra setup.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import train_dl_cohort_with_age_limits as base  # noqa: E402

DATA_DIR = base.DATA_DIR
KEY = ["Country", "Year", "Age", "JumpOffYear"]
VARIANT = "square_unlimited"  # the tuned config the ensemble is built over
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]

# Keep per-member models out of the (tracked) /models dir: /data is gitignored.
MEMBER_MODELS_ROOT = os.path.join(DATA_DIR, "ensemble_member_models")


def train_member(seed, asfr_all, geo_dim, valid_countries, valid_combos,
                 hparams, train_window):
    """Train all jump-off-year models under one seed; return its cohort forecast.

    Sets the pipeline's SEED global (used to reseed every jump-off year) and
    redirects model saves to a per-seed, gitignored directory so members don't
    overwrite one another.
    """
    base.SEED = seed
    base.MODELS_DIR = os.path.join(MEMBER_MODELS_ROOT, f"seed{seed}")
    os.makedirs(base.MODELS_DIR, exist_ok=True)

    predicted_rates, _ = base.train_and_forecast(
        asfr_all, geo_dim, valid_countries, valid_combos,
        hparams=hparams, train_window=train_window,
    )
    pred = base.build_predicted_cohort(predicted_rates, valid_combos)
    pred.to_csv(os.path.join(DATA_DIR, f"dl_forecasts_square_seed{seed}.csv"),
                index=False)
    print(f"[seed {seed}] member forecast: {len(pred):,} rows")
    return pred


def main():
    hparams, train_window, lexis_shape = base.load_tuned_config(VARIANT)
    asfr_all = base.load_data(lexis_shape)
    geo_dim, valid_countries, valid_combos = base.validate_countries(asfr_all)

    # Observed cohort rates are seed-independent -- build once and save under the
    # ensemble name so the notebook registry can pair pred+obs consistently.
    obs = base.build_observed_cohort(asfr_all, valid_combos)
    obs.to_csv(os.path.join(DATA_DIR, "dl_obs_cohort_square_ensemble.csv"),
               index=False)

    members = []
    for seed in SEEDS:
        print(f"\n{'#' * 60}\n# ENSEMBLE MEMBER  seed={seed}\n{'#' * 60}")
        pred = train_member(seed, asfr_all, geo_dim, valid_countries,
                            valid_combos, hparams, train_window)
        members.append(pred.assign(_seed=seed))

    allm = pd.concat(members, ignore_index=True)

    # Ensemble = per-cell mean forecast across seeds (grids are identical).
    ens = allm.groupby(KEY, as_index=False)["Rate"].mean()
    ens["Method"] = "DL_NonLog"
    ens["Key"] = ens["Method"] + "_" + ens["Country"].astype(str)
    ens = ens[["Country", "Year", "Age", "Rate", "JumpOffYear", "Method", "Key"]]
    ens.to_csv(os.path.join(DATA_DIR, "dl_forecasts_square_ensemble.csv"),
               index=False)

    # Per-cell disagreement across seeds -> spread band for plots.
    sd = (allm.groupby(KEY, as_index=False)["Rate"].std()
          .rename(columns={"Rate": "RateSD"}))
    sd.to_csv(os.path.join(DATA_DIR, "dl_forecasts_square_ensemble_seedsd.csv"),
              index=False)

    print(f"\n{'=' * 60}\nEnsemble over seeds {SEEDS}: {len(ens):,} cells")
    print(f"Mean per-cell seed SD: {sd['RateSD'].mean():.5f}")
    print("Saved: dl_forecasts_square_ensemble.csv, "
          "dl_forecasts_square_ensemble_seedsd.csv, "
          "dl_obs_cohort_square_ensemble.csv, and per-seed member forecasts.")


if __name__ == "__main__":
    main()
