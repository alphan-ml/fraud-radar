"""Feature engineering for Fraud Radar.

Feature groups:
  - Numeric V / C / D columns: used as-is, cast to float32.
  - M columns: mostly T/F flags (M1, M2, M3, M5, M6, M7, M8, M9) -> mapped to
    1.0 / 0.0 / NaN and cast to float32. M4 takes values M0/M1/M2 (not T/F)
    and is kept as a LightGBM categorical instead.
  - card1-6, addr1, addr2, P_emaildomain, R_emaildomain, ProductCD,
    DeviceType, DeviceInfo, M4: LightGBM categoricals (pandas "category"
    dtype -- LightGBM's native categorical handling, no one-hot).
  - TransactionAmt -> log1p(TransactionAmt) (`amt_log`); the raw amount is
    dropped in favor of the log version (heavy right tail).
  - hour-of-day and day-index from TransactionDT (TransactionDT is seconds
    elapsed from a fixed but undocumented reference point -- Kaggle's own
    note on the dataset -- so only hour-of-day and elapsed-day-index are
    meaningful, not a calendar date).
  - Count encodings (frequency of each value, computed independently of
    isFraud) for card1, addr1, P_emaildomain -- fit with `fit_freq_maps` on
    a training slice only and passed in as `freq_maps=`, so serving and
    evaluation see the same population frequencies the model trained on
    instead of recomputing counts on whatever frame happens to be at hand
    (one row at serving time -> every count would otherwise be 1.0). An
    unseen key (a card1/addr1/P_emaildomain value never seen in the
    training slice) maps to 0. If `freq_maps` is omitted, `build_features`
    falls back to fitting on its own input, which is only appropriate for
    ad hoc/exploratory calls -- `model.train()` always supplies maps fit on
    the training slice.

`build_features` never reads the `isFraud` column -- it is dropped from the
input at the top of the function before any feature is computed, so a
mislabeled or shuffled `isFraud` column cannot change a single feature
value. tests/test_no_leak.py checks this directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60

M_BINARY_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
M_CATEGORICAL_COLS = ["M4"]

CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]
CATEGORICAL_BASE = CARD_COLS + [
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "ProductCD",
    "DeviceType", "DeviceInfo",
] + M_CATEGORICAL_COLS

COUNT_ENCODE_COLS = ["card1", "addr1", "P_emaildomain"]

ID_COLS = ["TransactionID"]
NON_FEATURE_COLS = {"TransactionID", "isFraud", "TransactionDT"}


def _numeric_prefixed(columns: list[str], prefix: str) -> list[str]:
    out = []
    for c in columns:
        if c.startswith(prefix) and c[len(prefix):].isdigit():
            out.append(c)
    return out


def _stringify(s: pd.Series) -> pd.Series:
    """Stringifies a (possibly categorical) column with a stable "nan"
    placeholder for missing values. Plain `.astype(str)` on a pandas
    category dtype leaves missing entries as float NaN instead of the
    string "nan", so they would never match a fitted missing-value bucket
    in a dict keyed by strings -- every missing value would count as
    "unseen" (0) regardless of how common missingness was in training."""
    return s.astype(object).where(s.notna(), "nan").astype(str)


def fit_freq_maps(
    df: pd.DataFrame, cols: list[str] = COUNT_ENCODE_COLS
) -> dict[str, dict[str, int]]:
    """Frequency maps (value, stringified -> count) for `cols`, fit on
    whatever rows are in `df`. Call this on a training slice only -- fitting
    on validation/holdout/serving rows is exactly the leak `apply_freq_maps`
    exists to avoid."""
    maps: dict[str, dict[str, int]] = {}
    for c in cols:
        counts = _stringify(df[c]).value_counts(dropna=False)
        maps[c] = {str(k): int(v) for k, v in counts.items()}
    return maps


def apply_freq_maps(
    df: pd.DataFrame, freq_maps: dict[str, dict[str, int]], cols: list[str] = COUNT_ENCODE_COLS
) -> pd.DataFrame:
    """Maps each column in `cols` to its fitted count via `freq_maps`,
    written to `f"{col}_count"`. A value not present in the fitted map
    (unseen at fit time) maps to 0."""
    df = df.copy()
    for c in cols:
        fmap = freq_maps.get(c, {})
        df[f"{c}_count"] = _stringify(df[c]).map(fmap).fillna(0.0).astype("float32")
    return df


def build_features(
    df: pd.DataFrame, freq_maps: dict[str, dict[str, int]] | None = None
) -> pd.DataFrame:
    """Build the model-ready feature frame. Input must have TransactionID and
    TransactionDT; isFraud, if present, is dropped immediately and never
    read again in this function.

    `freq_maps`, if given, must be the output of `fit_freq_maps` on a
    training slice -- it is used to compute card1_count/addr1_count/
    P_emaildomain_count so serving and evaluation match training. If
    omitted, the maps are fit on `df` itself (only correct when `df` is the
    same population the model should score, e.g. ad hoc exploration)."""
    df = df.drop(columns=["isFraud"], errors="ignore").copy()

    out = pd.DataFrame(index=df.index)
    out["TransactionID"] = df["TransactionID"].astype("int64")
    out["TransactionDT"] = df["TransactionDT"].astype("int64")

    # -- time features --
    out["hour_of_day"] = ((df["TransactionDT"] // 3600) % 24).astype("float32")
    out["day_index"] = (df["TransactionDT"] // SECONDS_PER_DAY).astype("int32")

    # -- amount: log1p(TransactionAmt), clipped at 0 to guard a stray
    # negative value before the log; NaN amounts stay NaN (LightGBM handles
    # missing values natively, no imputation needed). --
    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").clip(lower=0)
    out["amt_log"] = np.log1p(amt).astype("float32")

    # -- numeric V / C / D columns, as-is, float32 --
    v_cols = _numeric_prefixed(list(df.columns), "V")
    c_cols = _numeric_prefixed(list(df.columns), "C")
    d_cols = _numeric_prefixed(list(df.columns), "D")
    for c in v_cols + c_cols + d_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # -- M columns: binary T/F -> 1/0, M4 categorical --
    for c in M_BINARY_COLS:
        if c in df.columns:
            out[c] = df[c].map({"T": 1.0, "F": 0.0}).astype("float32")
    for c in M_CATEGORICAL_COLS:
        if c in df.columns:
            out[c] = df[c].astype("category")

    # -- card / addr / email / device categoricals --
    for c in CATEGORICAL_BASE:
        if c in df.columns and c not in out.columns:
            out[c] = df[c].astype("string").astype("category")

    # -- id_ numeric columns (identity), float32, as-is --
    id_numeric_cols = [c for c in df.columns if c.startswith("id_") and pd.api.types.is_numeric_dtype(df[c])]
    for c in id_numeric_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # -- count encodings (independent of isFraud) --
    if freq_maps is None:
        freq_maps = fit_freq_maps(out, COUNT_ENCODE_COLS)
    out = apply_freq_maps(out, freq_maps, COUNT_ENCODE_COLS)

    return out


CATEGORICAL_FEATURES = [c for c in CATEGORICAL_BASE]
