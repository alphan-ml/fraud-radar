"""Proves the train/val/holdout split is strictly time-ordered and the
80/20 (with a 10% early-stopping slice inside the 80%) split sizes hold."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_radar.model import time_split


def _feat(n: int) -> pd.DataFrame:
    dt = np.random.RandomState(0).permutation(n)  # out-of-order DT on purpose
    return pd.DataFrame({"TransactionDT": dt})


def test_split_is_time_ordered_across_groups():
    feat = _feat(1000)
    labels = pd.Series([0] * 1000)
    train_idx, val_idx, holdout_idx, cutoff_dt, val_cutoff_dt = time_split(feat, labels)

    dt = feat["TransactionDT"].to_numpy()
    assert dt[train_idx].max() <= dt[val_idx].min()
    assert dt[val_idx].max() <= dt[holdout_idx].min()
    assert dt[train_idx].max() == val_cutoff_dt
    assert dt[holdout_idx].min() > cutoff_dt or len(holdout_idx) == 0


def test_split_sizes_are_80_20_with_10pct_val_inside_pool():
    n = 1000
    feat = _feat(n)
    labels = pd.Series([0] * n)
    train_idx, val_idx, holdout_idx, _, _ = time_split(feat, labels)

    assert len(holdout_idx) == round(n * 0.20)
    pool = n - len(holdout_idx)
    assert len(val_idx) == round(pool * 0.10)
    assert len(train_idx) == pool - len(val_idx)


def test_split_partitions_every_row_exactly_once():
    n = 733  # not evenly divisible, exercises rounding
    feat = _feat(n)
    labels = pd.Series([0] * n)
    train_idx, val_idx, holdout_idx, _, _ = time_split(feat, labels)

    all_idx = np.concatenate([train_idx, val_idx, holdout_idx])
    assert len(all_idx) == n
    assert len(set(all_idx.tolist())) == n
