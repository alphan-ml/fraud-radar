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
    isFraud) for card1, addr1, P_emaildomain.

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature frame. Input must have TransactionID and
    TransactionDT; isFraud, if present, is dropped immediately and never
    read again in this function."""
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
    for c in COUNT_ENCODE_COLS:
        counts = df[c].value_counts(dropna=False)
        out[f"{c}_count"] = df[c].map(counts).astype("float32")

    return out


CATEGORICAL_FEATURES = [c for c in CATEGORICAL_BASE]
