"""Feature-shape and feature-value tests on small synthetic fixtures."""
from __future__ import annotations

import math

import pandas as pd

from fraud_radar.features import build_features


def _row(**overrides):
    base = {
        "TransactionID": 1,
        "isFraud": 0,
        "TransactionDT": 90061,  # 1 day, 1 hour, 1 minute, 1 second
        "TransactionAmt": 99.0,
        "ProductCD": "W",
        "card1": 1001,
        "addr1": 200,
        "P_emaildomain": "gmail.com",
        "C1": 1.0,
        "D1": 2.0,
        "V1": 3.0,
        "M1": "T",
    }
    base.update(overrides)
    return base


def test_hour_of_day_and_day_index():
    df = pd.DataFrame([_row(TransactionDT=90061)])  # day_index=1, hour=1
    feat = build_features(df)
    assert feat.loc[0, "day_index"] == 1
    assert feat.loc[0, "hour_of_day"] == 1.0


def test_amt_log_is_log1p():
    df = pd.DataFrame([_row(TransactionAmt=99.0)])
    feat = build_features(df)
    assert math.isclose(feat.loc[0, "amt_log"], math.log1p(99.0), rel_tol=1e-4)


def test_m_binary_mapped_to_1_0():
    df = pd.DataFrame([_row(M1="T"), _row(TransactionID=2, M1="F")])
    feat = build_features(df)
    assert feat.loc[0, "M1"] == 1.0
    assert feat.loc[1, "M1"] == 0.0


def test_categorical_dtype_for_card_and_email():
    df = pd.DataFrame([_row()])
    feat = build_features(df)
    assert str(feat["card1"].dtype) == "category"
    assert str(feat["P_emaildomain"].dtype) == "category"


def test_count_encoding_counts_repeats():
    df = pd.DataFrame([
        _row(TransactionID=1, card1=1001),
        _row(TransactionID=2, card1=1001),
        _row(TransactionID=3, card1=2002),
    ])
    feat = build_features(df)
    assert feat.loc[0, "card1_count"] == 2.0
    assert feat.loc[1, "card1_count"] == 2.0
    assert feat.loc[2, "card1_count"] == 1.0
