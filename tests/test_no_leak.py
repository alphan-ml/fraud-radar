"""Proves build_features never uses isFraud information: features computed
from the same rows are bit-identical whether isFraud is present, absent, or
scrambled."""
from __future__ import annotations

import pandas as pd

from fraud_radar.features import build_features


def _rows(n: int, fraud: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "TransactionID": list(range(1, n + 1)),
        "isFraud": fraud,
        "TransactionDT": [86400 * i + 3600 * (i % 24) for i in range(n)],
        "TransactionAmt": [10.0 * (i + 1) for i in range(n)],
        "ProductCD": ["W"] * n,
        "card1": [1000 + i for i in range(n)],
        "addr1": [200] * n,
        "P_emaildomain": ["gmail.com"] * n,
        "C1": [float(i) for i in range(n)],
        "D1": [float(i) for i in range(n)],
        "V1": [float(i) for i in range(n)],
        "M1": ["T", "F"] * (n // 2) + (["T"] if n % 2 else []),
    })


def test_features_identical_when_isfraud_dropped():
    df_with = _rows(6, [0, 1, 0, 1, 0, 1])
    df_without = df_with.drop(columns=["isFraud"])
    feat_with = build_features(df_with)
    feat_without = build_features(df_without)
    pd.testing.assert_frame_equal(feat_with, feat_without)


def test_features_identical_when_isfraud_scrambled():
    df_a = _rows(6, [0, 0, 0, 0, 0, 0])
    df_b = df_a.copy()
    df_b["isFraud"] = [1, 1, 1, 1, 1, 1]
    feat_a = build_features(df_a)
    feat_b = build_features(df_b)
    pd.testing.assert_frame_equal(feat_a, feat_b)


def test_build_features_output_has_no_isfraud_column():
    df = _rows(4, [0, 1, 0, 1])
    feat = build_features(df)
    assert "isFraud" not in feat.columns


def test_time_since_prev_at_time_t_equals_value_from_rows_strictly_before_t():
    """F3 requirement: a time-delta feature for a transaction at time t must
    equal the value computed from rows strictly before t -- appending a row
    that happens later in time must not change any earlier row's value."""
    df = _rows(4, [0, 0, 0, 0])
    df["card1"] = [1001, 1001, 1001, 1001]  # same card1 throughout
    feat_before = build_features(df)

    later_row = df.iloc[[0]].copy()
    later_row["TransactionID"] = 999
    later_row["TransactionDT"] = df["TransactionDT"].max() + 100_000  # strictly later than every existing row
    df_with_future_row = pd.concat([df, later_row], ignore_index=True)
    feat_after = build_features(df_with_future_row)

    original_rows = feat_after[feat_after["TransactionID"] != 999].reset_index(drop=True)
    pd.testing.assert_series_equal(
        feat_before["time_since_prev_card1"], original_rows["time_since_prev_card1"]
    )


def test_time_since_prev_ignores_rows_that_come_after_it_even_when_present():
    """The same property stated the other way: an earlier row's value must
    match what you'd get computing it from only the rows before it, not from
    the full frame (which also contains later rows)."""
    df = _rows(3, [0, 0, 0])
    df["card1"] = [1001, 1001, 1001]
    full_feat = build_features(df)

    prefix_only = df.iloc[:2].copy()
    prefix_feat = build_features(prefix_only)

    assert full_feat.loc[1, "time_since_prev_card1"] == prefix_feat.loc[1, "time_since_prev_card1"]
