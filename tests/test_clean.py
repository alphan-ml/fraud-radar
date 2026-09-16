"""Merge and row-count tests for clean.py, on small synthetic fixtures (no
dependency on the real, gitignored Kaggle CSVs -- CI must run without them)."""
from __future__ import annotations

import pandas as pd

from fraud_radar.clean import merge_frames


def _tx(ids):
    return pd.DataFrame({
        "TransactionID": ids,
        "isFraud": [0] * len(ids),
        "TransactionDT": list(range(len(ids))),
        "TransactionAmt": [10.0] * len(ids),
    })


def _idn(ids):
    return pd.DataFrame({"TransactionID": ids, "id_01": [1.0] * len(ids)})


def test_left_join_keeps_all_transaction_rows():
    tx = _tx([1, 2, 3])
    idn = _idn([2])  # only transaction 2 has an identity row
    merged = merge_frames(tx, idn)
    assert len(merged) == 3
    assert set(merged["TransactionID"]) == {1, 2, 3}


def test_left_join_no_identity_rows_are_nan_not_dropped():
    tx = _tx([1, 2])
    idn = _idn([2])
    merged = merge_frames(tx, idn).set_index("TransactionID")
    assert pd.isna(merged.loc[1, "id_01"])
    assert merged.loc[2, "id_01"] == 1.0


def test_left_join_never_duplicates_transaction_rows():
    tx = _tx([1, 2, 3])
    idn = _idn([1, 2, 3])
    merged = merge_frames(tx, idn)
    assert len(merged) == len(tx)
