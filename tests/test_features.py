"""Feature-shape and feature-value tests on small synthetic fixtures."""
from __future__ import annotations

import math

import pandas as pd

from fraud_radar.features import (
    apply_freq_maps,
    apply_group_stats,
    build_features,
    fit_freq_maps,
    fit_group_stats,
)


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


def _multi_row_df():
    return pd.DataFrame([
        _row(TransactionID=1, card1=1001, addr1=200, TransactionAmt=10.0, D1=0.0),
        _row(TransactionID=2, card1=1001, addr1=200, TransactionAmt=30.0, D1=1.0),
        _row(TransactionID=3, card1=1001, addr1=201, TransactionAmt=100.0, D1=0.0),
        _row(TransactionID=4, card1=2002, addr1=200, TransactionAmt=50.0, D1=0.0),
    ])


def test_uid_combines_card1_addr1_and_d1_derived_start_day():
    df = _multi_row_df()
    feat = build_features(df)
    # rows 0 and 1 share card1+addr1 but different D1 -> same day_index (all
    # TransactionDT equal in the fixture) means a different account_start_day
    # -> a different uid, since these look like two different accounts on
    # the same card+address.
    assert feat.loc[0, "uid"] != feat.loc[1, "uid"]
    # row 2 shares card1 but a different addr1 -> different uid from row 0.
    assert feat.loc[0, "uid"] != feat.loc[2, "uid"]
    # row 3 is a different card1 entirely -> different uid.
    assert feat.loc[0, "uid"] != feat.loc[3, "uid"]


def test_uid_missing_d1_uses_a_shared_nan_bucket():
    df = _multi_row_df()
    df.loc[0, "D1"] = None
    df.loc[3, "D1"] = None
    feat = build_features(df)
    # Different card1 -> still a different uid even with D1 missing on both.
    assert feat.loc[0, "uid"] != feat.loc[3, "uid"]


def test_group_stats_count_and_mean_amt_per_composite_key():
    df = _multi_row_df()  # rows 0,1 share card1=1001,addr1=200
    feat = build_features(df)
    assert feat.loc[0, "card1_addr1_count"] == 2.0
    assert feat.loc[1, "card1_addr1_count"] == 2.0
    assert feat.loc[2, "card1_addr1_count"] == 1.0
    expected_mean = (feat.loc[0, "amt_log"] + feat.loc[1, "amt_log"]) / 2
    assert math.isclose(feat.loc[0, "card1_addr1_mean_amt"], expected_mean, rel_tol=1e-4)


def test_fit_group_stats_fits_on_given_frame_only():
    df = _multi_row_df()
    feat = build_features(df)  # self-fit, only to get category dtypes right
    maps = fit_group_stats(feat, group_defs=[("card1_addr1", ["card1", "addr1"], ["count"])])
    assert maps["card1_addr1"]["count"] == {"1001||200": 2, "1001||201": 1, "2002||200": 1}


def test_apply_group_stats_uses_saved_stats_not_the_scored_frame():
    df = _multi_row_df()
    feat = build_features(df)
    maps = fit_group_stats(feat, group_defs=[("card1_addr1", ["card1", "addr1"], ["count"])])

    one_row = pd.DataFrame([_row(TransactionID=99, card1=1001, addr1=200)])
    one_feat = build_features(one_row)
    applied = apply_group_stats(one_feat, maps, group_defs=[("card1_addr1", ["card1", "addr1"], ["count"])])
    assert applied.loc[0, "card1_addr1_count"] == 2.0


def test_group_stats_unseen_key_is_zero_count_nan_mean():
    maps = {"card1_addr1": {"count": {"1001||200": 5}, "mean_amt": {"1001||200": 4.0}}}
    df = pd.DataFrame([_row(TransactionID=1, card1=9999, addr1=200)])
    feat = build_features(df)
    applied = apply_group_stats(feat, maps, group_defs=[("card1_addr1", ["card1", "addr1"], ["count", "mean_amt"])])
    assert applied.loc[0, "card1_addr1_count"] == 0.0
    assert math.isnan(applied.loc[0, "card1_addr1_mean_amt"])


def test_amt_to_card1_mean_ratio_is_amt_log_over_group_mean():
    df = _multi_row_df()
    feat = build_features(df)
    expected = feat.loc[2, "amt_log"] / feat.loc[2, "card1_mean_amt"]
    assert math.isclose(feat.loc[2, "amt_to_card1_mean_ratio"], expected, rel_tol=1e-4)


def test_time_since_prev_card1_is_seconds_since_last_same_card1():
    df = pd.DataFrame([
        _row(TransactionID=1, card1=1001, TransactionDT=1000),
        _row(TransactionID=2, card1=1001, TransactionDT=2500),
        _row(TransactionID=3, card1=2002, TransactionDT=3000),
        _row(TransactionID=4, card1=1001, TransactionDT=9000),
    ])
    feat = build_features(df)
    assert math.isnan(feat.loc[0, "time_since_prev_card1"])
    assert feat.loc[1, "time_since_prev_card1"] == 1500.0
    assert math.isnan(feat.loc[2, "time_since_prev_card1"])  # first card2002 row
    assert feat.loc[3, "time_since_prev_card1"] == 6500.0


def test_time_since_prev_card1_addr1_requires_both_to_match():
    df = pd.DataFrame([
        _row(TransactionID=1, card1=1001, addr1=200, TransactionDT=1000),
        _row(TransactionID=2, card1=1001, addr1=201, TransactionDT=1500),
        _row(TransactionID=3, card1=1001, addr1=200, TransactionDT=4000),
    ])
    feat = build_features(df)
    assert math.isnan(feat.loc[0, "time_since_prev_card1_addr1"])
    assert math.isnan(feat.loc[1, "time_since_prev_card1_addr1"])  # different addr1
    assert feat.loc[2, "time_since_prev_card1_addr1"] == 3000.0  # matches row 0, not row 1


def test_time_since_prev_is_unaffected_by_input_row_order():
    df = pd.DataFrame([
        _row(TransactionID=1, card1=1001, TransactionDT=1000),
        _row(TransactionID=2, card1=1001, TransactionDT=2500),
    ])
    shuffled = df.iloc[[1, 0]].reset_index(drop=True)
    feat = build_features(df)
    feat_shuffled = build_features(shuffled)
    row1_original = feat[feat["TransactionID"] == 2].iloc[0]
    row1_shuffled = feat_shuffled[feat_shuffled["TransactionID"] == 2].iloc[0]
    assert row1_original["time_since_prev_card1"] == row1_shuffled["time_since_prev_card1"] == 1500.0
