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
    isFraud) for card1, addr1, P_emaildomain, uid -- fit with `fit_freq_maps`
    on a training slice only and passed in as `freq_maps=`, so serving and
    evaluation see the same population frequencies the model trained on
    instead of recomputing counts on whatever frame happens to be at hand
    (one row at serving time -> every count would otherwise be 1.0). An
    unseen key (a value never seen in the training slice) maps to 0. If
    `freq_maps` is omitted, `build_features` falls back to fitting on its
    own input, which is only appropriate for ad hoc/exploratory calls --
    `model.train()` always supplies maps fit on the training slice.
  - Group aggregates (feature pass F3) -- count and mean(amt_log) per
    card1+addr1 and card1+P_emaildomain, plus mean(amt_log) per card1 alone
    (card1's own count already comes from the count-encoding above) -- fit
    with `fit_group_stats` on a training slice only and passed in as
    `group_stats_maps=`, same leakage-safe pattern as `freq_maps`. The
    composite group key is never itself persisted as a column (it would be
    an unbounded-cardinality string LightGBM can't use); only its derived
    `_count`/`_mean_amt` numeric columns are.
  - `amt_to_card1_mean_ratio` -- amt_log divided by the card1 group's
    mean(amt_log) (from `group_stats_maps`); NaN when card1 is unseen in
    the training slice or its mean is 0.
  - `time_since_prev_card1` / `time_since_prev_card1_addr1` -- seconds
    since the previous transaction (by TransactionDT) sharing the same
    card1 / same (card1, addr1), computed by sorting the whole input by
    TransactionDT and taking a within-group diff. This never needs a
    fitted map and is never refit per split: a row's value only ever comes
    from rows strictly earlier in TransactionDT than that row, so it holds
    for train, validation and holdout rows alike without leaking anything
    forward. The first transaction seen for a key gets NaN (no prior
    transaction). A single-row call (serving) always yields NaN here --
    there is no history in a one-row frame -- a known limitation, not a bug
    (see README).
  - `uid` -- a stable per-account id, `card1|addr1|account_start_day`,
    where `account_start_day = day_index - D1` (D1 is IEEE-CIS's "days
    since this card's first transaction" field, undocumented beyond that;
    subtracting it from the current day index recovers an approximately
    constant "day zero" for the account, which is the standard "uid" trick
    for this dataset). NaN D1 collapses to a shared "nan" bucket per
    card1+addr1 (so account_start_day is unknown, not a distinct account).
    Kept as a LightGBM categorical (like card1); `uid_count` (via the
    count-encoding pipeline above) is its training-slice population count.

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
] + M_CATEGORICAL_COLS + ["uid"]

COUNT_ENCODE_COLS = ["card1", "addr1", "P_emaildomain", "uid"]

# Group aggregates (F3): each entry is (group_name, group_cols, stats).
# "card1" alone only needs mean_amt here -- its count is already produced by
# COUNT_ENCODE_COLS above, and re-deriving it under a second name would just
# be the same numbers under a different column.
GROUP_STATS_DEFS = [
    ("card1", ["card1"], ["mean_amt"]),
    ("card1_addr1", ["card1", "addr1"], ["count", "mean_amt"]),
    ("card1_P_emaildomain", ["card1", "P_emaildomain"], ["count", "mean_amt"]),
]

TIME_SINCE_PREV_GROUPS = [
    ("time_since_prev_card1", ["card1"]),
    ("time_since_prev_card1_addr1", ["card1", "addr1"]),
]

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


def _composite_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Stringified composite key for a group of columns, e.g. card1+addr1 ->
    "1001||200". Never persisted as a column -- an unbounded-cardinality
    string isn't something LightGBM can use -- only used as a groupby/map
    key inside `fit_group_stats`/`apply_group_stats`."""
    key = _stringify(df[cols[0]])
    for c in cols[1:]:
        key = key + "||" + _stringify(df[c])
    return key


def fit_group_stats(
    df: pd.DataFrame, group_defs: list[tuple[str, list[str], list[str]]] = GROUP_STATS_DEFS
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-group count / mean(amt_log) maps, fit on whatever rows are in
    `df`. Call this on a training slice only -- see `fit_freq_maps`, same
    leakage concern. `df` must already have `amt_log` computed."""
    maps: dict[str, dict[str, dict[str, float]]] = {}
    for name, cols, stats in group_defs:
        key = _composite_key(df, cols)
        entry: dict[str, dict[str, float]] = {}
        if "count" in stats:
            counts = key.value_counts(dropna=False)
            entry["count"] = {str(k): int(v) for k, v in counts.items()}
        if "mean_amt" in stats:
            means = df["amt_log"].groupby(key).mean()
            entry["mean_amt"] = {str(k): float(v) for k, v in means.items()}
        maps[name] = entry
    return maps


def apply_group_stats(
    df: pd.DataFrame,
    group_stats_maps: dict[str, dict[str, dict[str, float]]],
    group_defs: list[tuple[str, list[str], list[str]]] = GROUP_STATS_DEFS,
) -> pd.DataFrame:
    """Maps each group in `group_defs` to its fitted count/mean(amt_log) via
    `group_stats_maps`, written to `f"{name}_count"`/`f"{name}_mean_amt"`.
    An unseen key maps to 0 for count, NaN for mean_amt (LightGBM handles
    missing values natively; a fabricated 0.0 mean would look like a real,
    very-low-amount account instead of "unknown")."""
    df = df.copy()
    for name, cols, stats in group_defs:
        key = _composite_key(df, cols)
        entry = group_stats_maps.get(name, {})
        if "count" in stats:
            df[f"{name}_count"] = key.map(entry.get("count", {})).fillna(0.0).astype("float32")
        if "mean_amt" in stats:
            df[f"{name}_mean_amt"] = key.map(entry.get("mean_amt", {})).astype("float32")
    return df


def _compute_uid(card1: pd.Series, addr1: pd.Series, day_index: pd.Series, d1: pd.Series) -> pd.Series:
    """`card1|addr1|account_start_day` -- see the module docstring's "uid"
    entry. `account_start_day = day_index - D1`; NaN D1 collapses to a
    shared "nan" bucket (via `_stringify`) rather than raising or silently
    dropping the row."""
    account_start_day = day_index.astype("float64") - pd.to_numeric(d1, errors="coerce")
    start_day_str = account_start_day.round().astype(object).where(account_start_day.notna(), "nan").astype(str)
    return _stringify(card1) + "||" + _stringify(addr1) + "||" + start_day_str


def _time_since_prev(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Seconds since the previous transaction (by TransactionDT) sharing the
    same key in `cols`. Sorts by TransactionDT, takes a within-group diff,
    and scatters the result back to `df`'s original row order -- every value
    only ever comes from a row strictly earlier in TransactionDT, so this is
    leakage-safe for train/validation/holdout alike without being refit per
    split (tests/test_no_leak.py proves the strictly-earlier-only property
    directly)."""
    key = _composite_key(df, cols)
    order = df["TransactionDT"].to_numpy().argsort(kind="stable")
    sorted_dt = pd.Series(df["TransactionDT"].to_numpy()[order])
    sorted_key = key.to_numpy()[order]
    diff = sorted_dt.groupby(sorted_key).diff()
    result = np.empty(len(df), dtype="float64")
    result[order] = diff.to_numpy()
    return pd.Series(result, index=df.index)


def build_features(
    df: pd.DataFrame,
    freq_maps: dict[str, dict[str, int]] | None = None,
    group_stats_maps: dict[str, dict[str, dict[str, float]]] | None = None,
) -> pd.DataFrame:
    """Build the model-ready feature frame. Input must have TransactionID and
    TransactionDT; isFraud, if present, is dropped immediately and never
    read again in this function.

    `freq_maps`, if given, must be the output of `fit_freq_maps` on a
    training slice -- it is used to compute card1_count/addr1_count/
    P_emaildomain_count/uid_count so serving and evaluation match training.
    `group_stats_maps`, if given, must be the output of `fit_group_stats` on
    the same training slice -- it is used for the card1/card1_addr1/
    card1_P_emaildomain count and mean(amt_log) aggregates. If either is
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

    # -- uid: card1 + addr1 + D1-derived account start day (see module
    # docstring). Built from the raw input columns, not from `out`, so it
    # does not depend on where card1/addr1 land in the categorical loop
    # below. --
    card1_raw = df["card1"] if "card1" in df.columns else pd.Series(np.nan, index=df.index)
    addr1_raw = df["addr1"] if "addr1" in df.columns else pd.Series(np.nan, index=df.index)
    d1_raw = df["D1"] if "D1" in df.columns else pd.Series(np.nan, index=df.index)
    out["uid"] = _compute_uid(card1_raw, addr1_raw, out["day_index"], d1_raw).astype("category")

    # -- card / addr / email / device categoricals (includes uid, already
    # set above, so the "not in out.columns" guard skips it here) --
    for c in CATEGORICAL_BASE:
        if c in df.columns and c not in out.columns:
            out[c] = df[c].astype("string").astype("category")

    # -- id_ numeric columns (identity), float32, as-is --
    id_numeric_cols = [c for c in df.columns if c.startswith("id_") and pd.api.types.is_numeric_dtype(df[c])]
    for c in id_numeric_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # -- time since previous transaction, same card1 / same card1+addr1 --
    # purely causal (rows strictly earlier in TransactionDT only), so this
    # is never refit per split -- see module docstring and _time_since_prev.
    for feat_name, group_cols in TIME_SINCE_PREV_GROUPS:
        out[feat_name] = _time_since_prev(out, group_cols).astype("float32")

    # -- count encodings (independent of isFraud) --
    if freq_maps is None:
        freq_maps = fit_freq_maps(out, COUNT_ENCODE_COLS)
    out = apply_freq_maps(out, freq_maps, COUNT_ENCODE_COLS)

    # -- group aggregates: count / mean(amt_log) per card1, card1+addr1,
    # card1+P_emaildomain -- and amount relative to the card1 group mean. --
    if group_stats_maps is None:
        group_stats_maps = fit_group_stats(out, GROUP_STATS_DEFS)
    out = apply_group_stats(out, group_stats_maps, GROUP_STATS_DEFS)
    out["amt_to_card1_mean_ratio"] = (
        out["amt_log"] / out["card1_mean_amt"].replace(0.0, np.nan)
    ).astype("float32")

    return out


CATEGORICAL_FEATURES = [c for c in CATEGORICAL_BASE]
