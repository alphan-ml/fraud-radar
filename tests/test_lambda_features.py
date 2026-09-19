"""Proves aws-lambda/lambda_function.py's pure-numpy build_features_row()
reproduces fraud_radar.features.build_features() (the real, pandas-based
feature builder score.py uses) for every feature pass F3 addition: the
card1/card1_addr1/card1_P_emaildomain count and mean(amt_log) group
aggregates, uid + uid_count, amt_to_card1_mean_ratio, and the always-NaN
time_since_prev_* pair.

This does not require a trained model -- group aggregates, count encodings,
and uid are pure functions of the raw input plus fitted maps, computed
identically whether or not a booster is attached. It uses fabricated maps
(built by the real fit_freq_maps/fit_group_stats on a small synthetic
"training slice"), not the real outputs/checkpoints/ artifacts, which is
why it runs unconditionally (no skipif), unlike tests/test_score.py."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

from fraud_radar.features import (
    COUNT_ENCODE_COLS,
    GROUP_STATS_DEFS,
    apply_freq_maps,
    build_features,
    fit_freq_maps,
    fit_group_stats,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_lambda_module():
    spec = importlib.util.spec_from_file_location(
        "lambda_function", ROOT / "aws-lambda" / "lambda_function.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def lf():
    return _load_lambda_module()


def _row(**overrides):
    base = {
        "TransactionID": 1,
        "isFraud": 0,
        "TransactionDT": 90061,
        "TransactionAmt": 99.0,
        "ProductCD": "W",
        "card1": 1001,
        "addr1": 200,
        "P_emaildomain": "gmail.com",
        "D1": 2.0,
    }
    base.update(overrides)
    return base


TRAIN_ROWS = [
    _row(TransactionID=1, card1=1001, addr1=200, P_emaildomain="gmail.com", D1=2.0, TransactionAmt=10.0, TransactionDT=90061),
    _row(TransactionID=2, card1=1001, addr1=200, P_emaildomain="gmail.com", D1=2.0, TransactionAmt=30.0, TransactionDT=90062),
    _row(TransactionID=3, card1=1001, addr1=201, P_emaildomain="yahoo.com", D1=5.0, TransactionAmt=100.0, TransactionDT=90063),
    _row(TransactionID=4, card1=2002, addr1=200, P_emaildomain="gmail.com", D1=0.0, TransactionAmt=50.0, TransactionDT=90064),
]


def _fit_maps():
    train_df = pd.DataFrame(TRAIN_ROWS)
    train_feat = build_features(train_df)  # self-fit, only to get category dtypes right
    freq_maps = fit_freq_maps(train_feat, COUNT_ENCODE_COLS)
    train_feat = apply_freq_maps(train_feat, freq_maps, COUNT_ENCODE_COLS)
    group_stats_maps = fit_group_stats(train_feat, GROUP_STATS_DEFS)
    return freq_maps, group_stats_maps


def _wire_lambda(lf, freq_maps, group_stats_maps, feature_cols, cat_code_maps=None):
    lf._feature_cols = feature_cols
    lf._count_maps = freq_maps
    lf._group_stats = group_stats_maps
    lf._cat_code_maps = cat_code_maps or {}


FEATURE_COLS_UNDER_TEST = [
    "hour_of_day", "day_index", "amt_log", "D1",
    "card1_count", "addr1_count", "P_emaildomain_count", "uid_count",
    "card1_mean_amt", "card1_addr1_count", "card1_addr1_mean_amt",
    "card1_P_emaildomain_count", "card1_P_emaildomain_mean_amt",
    "amt_to_card1_mean_ratio",
    "time_since_prev_card1", "time_since_prev_card1_addr1",
]


def _pandas_reference(row: dict, freq_maps, group_stats_maps) -> pd.Series:
    feat = build_features(pd.DataFrame([row]), freq_maps=freq_maps, group_stats_maps=group_stats_maps)
    return feat.iloc[0]


@pytest.mark.parametrize(
    "row",
    [
        _row(TransactionID=1, card1=1001, addr1=200, P_emaildomain="gmail.com", D1=2.0, TransactionAmt=10.0, TransactionDT=90061),
        _row(TransactionID=3, card1=1001, addr1=201, P_emaildomain="yahoo.com", D1=5.0, TransactionAmt=100.0, TransactionDT=90063),
        _row(TransactionID=99, card1=1001, addr1=200, P_emaildomain="gmail.com", D1=2.0, TransactionAmt=20.0, TransactionDT=90099),
        _row(TransactionID=100, card1=9999, addr1=9999, P_emaildomain="unseen.com", D1=None, TransactionAmt=15.0, TransactionDT=90100),
    ],
    ids=["seen-card1-addr1-a", "seen-card1-addr1-b", "seen-card1-new-row", "unseen-everything"],
)
def test_group_and_count_features_match_pandas(lf, row):
    freq_maps, group_stats_maps = _fit_maps()
    _wire_lambda(lf, freq_maps, group_stats_maps, FEATURE_COLS_UNDER_TEST)

    expected = _pandas_reference(row, freq_maps, group_stats_maps)
    actual = lf.build_features_row(row)

    for i, col in enumerate(FEATURE_COLS_UNDER_TEST):
        exp = float(expected[col])
        got = float(actual[i])
        if math.isnan(exp):
            assert math.isnan(got), f"{col}: expected NaN, got {got}"
        else:
            assert got == pytest.approx(exp, rel=1e-4), f"{col}: expected {exp}, got {got}"


@pytest.mark.parametrize(
    "row",
    [
        _row(TransactionID=1, card1=1001, addr1=200, D1=2.0, TransactionDT=90061),
        _row(TransactionID=3, card1=1001, addr1=201, D1=5.0, TransactionDT=90063),
        _row(TransactionID=100, card1=9999, addr1=9999, D1=None, TransactionDT=90100),
    ],
    ids=["a", "b", "missing-d1"],
)
def test_uid_key_matches_the_real_category_pandas_computes(lf, row):
    """The lambda's own uid string must exactly match the category value
    fraud_radar.features._compute_uid puts in the "uid" column -- that
    string is the cache key into cat_code_maps.json's "uid" map, so any
    mismatch here silently sends every uid lookup to NaN in production."""
    freq_maps, group_stats_maps = _fit_maps()
    expected_row = _pandas_reference(row, freq_maps, group_stats_maps)
    expected_uid = str(expected_row["uid"])

    dt = float(row["TransactionDT"])
    day_index = float(int(dt // 86400))
    got_uid_key = lf._uid_key(row, day_index)

    assert got_uid_key == expected_uid


def test_time_since_prev_is_always_nan_for_a_single_row(lf):
    """Matches fraud_radar.score.FraudScorer.score_one()'s single-row
    behaviour exactly -- see the module docstring in lambda_function.py."""
    freq_maps, group_stats_maps = _fit_maps()
    row = _row(TransactionID=1, card1=1001, addr1=200, TransactionDT=90061)
    _wire_lambda(lf, freq_maps, group_stats_maps, ["time_since_prev_card1", "time_since_prev_card1_addr1"])
    actual = lf.build_features_row(row)
    assert math.isnan(actual[0])
    assert math.isnan(actual[1])
