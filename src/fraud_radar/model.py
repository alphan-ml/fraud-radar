"""Model for Fraud Radar: time-aware split, LightGBM binary classifier,
isotonic calibration.

Split (strictly time-ordered, no shuffling):
  1. Sort all labeled transactions by TransactionDT ascending.
  2. First 80% -> train+validation pool. Last 20% -> holdout (touched only
     by eval.py, never by training, early stopping, or calibration).
  3. Within the 80% pool, the last 10% (also time-ordered) is held out as
     the early-stopping / calibration validation slice; the first 90% of
     the pool is what the booster actually trains on.
  4. Isotonic regression is fit on the validation slice's predicted
     probabilities vs. true labels, to correct LightGBM's raw scores into
     calibrated probabilities.
  5. Count-encoding frequency maps (card1/addr1/P_emaildomain -> count) are
     fit on the train slice only (step 3's first 90%), before the booster
     ever sees a `_count` feature, and reused as-is to recompute those
     columns for the validation slice and holdout -- see
     `fraud_radar.features.fit_freq_maps`/`apply_freq_maps`. Saved to
     `outputs/checkpoints/freq_maps.json`.
  6. The review policy -- the score cutoff for the review queue -- is fit
     separately from calibration: `REVIEW_TOP_FRACTION` (0.02) of the
     validation slice's calibrated scores, by rank, gives `fitted_cutoff`.
     This is a probability score threshold picked to realize a target
     review rate, not a fixed operating point -- it will differ run to run
     as the model changes. Saved to
     `outputs/checkpoints/review_policy.json`.

Seed 26 everywhere a seed is accepted (LightGBM's own `seed` param; the
split itself is deterministic by TransactionDT order, not randomized).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fraud_radar.features import (
    CATEGORICAL_BASE,
    COUNT_ENCODE_COLS,
    GROUP_STATS_DEFS,
    apply_freq_maps,
    apply_group_stats,
    fit_freq_maps,
    fit_group_stats,
)

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
SEED = 26

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "seed": SEED,
    "verbose": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
REVIEW_TOP_FRACTION = 0.02


def time_split(feat: pd.DataFrame, labels: pd.Series, holdout_frac: float = 0.20, val_frac: float = 0.10):
    """Returns (train_idx, val_idx, holdout_idx, cutoff_dt, val_cutoff_dt) --
    all positional indices into feat, sorted by TransactionDT ascending."""
    order = feat["TransactionDT"].to_numpy().argsort(kind="stable")
    n = len(order)
    n_holdout = round(n * holdout_frac)
    n_pool = n - n_holdout
    n_val = round(n_pool * val_frac)
    n_train = n_pool - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train:n_pool]
    holdout_idx = order[n_pool:]

    cutoff_dt = int(feat["TransactionDT"].to_numpy()[order[n_pool - 1]])
    val_cutoff_dt = int(feat["TransactionDT"].to_numpy()[order[n_train - 1]])
    return train_idx, val_idx, holdout_idx, cutoff_dt, val_cutoff_dt


def _categorical_columns(feat: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_BASE if c in feat.columns]


def _feature_columns(feat: pd.DataFrame) -> list[str]:
    exclude = {"TransactionID", "TransactionDT"}
    return [c for c in feat.columns if c not in exclude]


def train(feat: pd.DataFrame, labels: pd.Series) -> dict:
    train_idx, val_idx, holdout_idx, cutoff_dt, val_cutoff_dt = time_split(feat, labels)

    # Count-encoding maps must be fit on the train slice only -- fitting on
    # the full frame (as build_features does by default) leaks
    # validation/holdout population frequency into features those rows are
    # later scored on. Recompute the *_count columns for the whole frame
    # with the train-only maps before anything is scored.
    freq_maps = fit_freq_maps(feat.iloc[train_idx], COUNT_ENCODE_COLS)
    feat = apply_freq_maps(feat, freq_maps, COUNT_ENCODE_COLS)

    # Same leakage concern for the F3 group aggregates (card1/card1+addr1/
    # card1+P_emaildomain count and mean(amt_log), and the amt_to_card1_mean
    # ratio that depends on card1_mean_amt) -- fit on the train slice only,
    # then recompute for the whole frame.
    group_stats_maps = fit_group_stats(feat.iloc[train_idx], GROUP_STATS_DEFS)
    feat = apply_group_stats(feat, group_stats_maps, GROUP_STATS_DEFS)
    feat["amt_to_card1_mean_ratio"] = (
        feat["amt_log"] / feat["card1_mean_amt"].replace(0.0, np.nan)
    ).astype("float32")

    feature_cols = _feature_columns(feat)
    cat_cols = _categorical_columns(feat)

    X = feat[feature_cols]
    y = labels.to_numpy()

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = dict(LGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)],
    )

    val_raw_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw_pred, y_val)

    # Review policy: the cutoff is the calibrated score at the
    # REVIEW_TOP_FRACTION percentile of the validation slice, fit
    # separately from the probability calibration above. This is a
    # different quantity from "score >= a fixed probability" -- it is
    # whatever probability realizes the target review rate on this model's
    # actual score distribution, and moves run to run.
    val_calibrated = calibrator.predict(val_raw_pred)
    fitted_cutoff = float(np.percentile(val_calibrated, 100 * (1 - REVIEW_TOP_FRACTION)))
    review_policy = {
        "top_fraction": REVIEW_TOP_FRACTION,
        "fitted_cutoff": round(fitted_cutoff, 6),
        "fitted_on": "validation",
        "n_validation": len(val_idx),
    }

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(CKPT_DIR / "lgb_model.txt"))
    joblib.dump(calibrator, CKPT_DIR / "calibrator.joblib")
    (CKPT_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    (CKPT_DIR / "categorical_columns.json").write_text(json.dumps(cat_cols, indent=2))
    (CKPT_DIR / "freq_maps.json").write_text(json.dumps(freq_maps, indent=2))
    (CKPT_DIR / "group_stats.json").write_text(json.dumps(group_stats_maps, indent=2))
    (CKPT_DIR / "review_policy.json").write_text(json.dumps(review_policy, indent=2))

    split_info = {
        "n_total": len(feat),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_holdout": len(holdout_idx),
        "cutoff_dt_train_pool_vs_holdout": cutoff_dt,
        "val_cutoff_dt_train_vs_val": val_cutoff_dt,
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": int(booster.best_iteration),
        "seed": SEED,
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "holdout_idx": [int(i) for i in holdout_idx],
    }
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "split_info.json").write_text(
        json.dumps({k: v for k, v in split_info.items() if not k.endswith("_idx")}, indent=2)
    )
    np.savez(
        CKPT_DIR / "split_idx.npz",
        train_idx=train_idx, val_idx=val_idx, holdout_idx=holdout_idx,
    )

    return split_info


def load() -> tuple[lgb.Booster, IsotonicRegression, list[str], list[str]]:
    booster = lgb.Booster(model_file=str(CKPT_DIR / "lgb_model.txt"))
    calibrator = joblib.load(CKPT_DIR / "calibrator.joblib")
    feature_cols = json.loads((CKPT_DIR / "feature_columns.json").read_text())
    cat_cols = json.loads((CKPT_DIR / "categorical_columns.json").read_text())
    return booster, calibrator, feature_cols, cat_cols


def load_freq_maps() -> dict[str, dict[str, int]]:
    return json.loads((CKPT_DIR / "freq_maps.json").read_text())


def load_group_stats() -> dict[str, dict[str, dict[str, float]]]:
    return json.loads((CKPT_DIR / "group_stats.json").read_text())


def load_review_policy() -> dict:
    return json.loads((CKPT_DIR / "review_policy.json").read_text())


def predict_calibrated(booster: lgb.Booster, calibrator: IsotonicRegression, X: pd.DataFrame) -> np.ndarray:
    raw = booster.predict(X, num_iteration=booster.best_iteration)
    return calibrator.predict(raw)


def load_split_idx() -> dict:
    npz = np.load(CKPT_DIR / "split_idx.npz")
    return {k: npz[k] for k in npz.files}
