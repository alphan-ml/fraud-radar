"""Feature-shape and feature-value tests on small synthetic fixtures."""
from __future__ import annotations

import math

import pandas as pd

from fraud_radar.features import apply_freq_maps, build_features, fit_freq_maps


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


def test_fit_freq_maps_fits_on_given_frame_only():
    train_df = pd.DataFrame([
        _row(TransactionID=1, card1=1001),
        _row(TransactionID=2, card1=1001),
        _row(TransactionID=3, card1=2002),
    ])
    train_feat = build_features(train_df)  # self-fit, only to get card1 as a category column
    freq_maps = fit_freq_maps(train_feat, cols=["card1"])
    assert freq_maps["card1"] == {"1001": 2, "2002": 1}


def test_apply_freq_maps_uses_saved_counts_not_the_scored_frame():
    train_df = pd.DataFrame([
        _row(TransactionID=1, card1=1001),
        _row(TransactionID=2, card1=1001),
        _row(TransactionID=3, card1=2002),
    ])
    train_feat = build_features(train_df)
    freq_maps = fit_freq_maps(train_feat, cols=["card1"])

    # A single row scored alone must see the *training* count, not 1.0.
    one_row = pd.DataFrame([_row(TransactionID=99, card1=1001)])
    feat = build_features(one_row, freq_maps=freq_maps)
    assert feat.loc[0, "card1_count"] == 2.0


def test_unseen_key_maps_to_zero():
    freq_maps = {"card1": {"1001": 2}}
    df = pd.DataFrame([_row(TransactionID=1, card1=9999)])
    feat = build_features(df, freq_maps=freq_maps)
    assert feat.loc[0, "card1_count"] == 0.0
    # addr1/P_emaildomain weren't fit at all -- also unseen, also 0.
    assert feat.loc[0, "addr1_count"] == 0.0
    assert feat.loc[0, "P_emaildomain_count"] == 0.0


def test_apply_freq_maps_is_independent_of_frame_it_is_applied_to():
    freq_maps = {"card1": {"1001": 7}}
    small = pd.DataFrame({"card1": pd.Series([1001], dtype="category")})
    big = pd.DataFrame({"card1": pd.Series([1001] * 50, dtype="category")})
    small_out = apply_freq_maps(small, freq_maps, cols=["card1"])
    big_out = apply_freq_maps(big, freq_maps, cols=["card1"])
    assert small_out.loc[0, "card1_count"] == 7.0
    assert big_out.loc[0, "card1_count"] == 7.0
