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
