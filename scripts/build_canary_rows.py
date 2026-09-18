"""Builds canary/rows.json: a fixed set of 500 transactions from the
time-ordered HOLDOUT split, stratified so at least 40 are fraud.

Run once, by hand, with the real data pipeline already through `clean`
(`fraud-radar fetch && fraud-radar clean`, Kaggle credentials required for
`fetch`). Not run by the canary workflow itself -- the canary set is fixed
once committed, so the live endpoint is always compared against the same
500 rows.

Selection is deterministic: within the holdout (already time-ordered by
`TransactionDT`), fraud and non-fraud rows are separated, then `N_FRAUD`
fraud rows and `N_TOTAL - N_FRAUD` non-fraud rows are picked evenly spaced
(`np.linspace` over each stratum's positions) so the canary set spans the
whole holdout time range rather than clustering at one end. Same style as
`aws-lambda/build_verification.py`'s example-row picker: no random sampling,
nothing hand-picked.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_radar.model import time_split

ROOT = Path(__file__).resolve().parents[1]
N_TOTAL = 500
N_FRAUD = 50  # comfortably above the "at least 40" floor


def _evenly_spaced(positions: np.ndarray, k: int) -> np.ndarray:
    if k >= len(positions):
        return positions
    pick = np.linspace(0, len(positions) - 1, k).astype(int)
    return positions[pick]


def row_to_dict(row: pd.Series, is_float_col: dict) -> dict:
    """Same NaN -> null handling as aws-lambda/build_verification.py: the
    file is committed and read by strict JSON parsers, so no NaN tokens."""
    d = {}
    for k, v in row.items():
        if k == "isFraud":
            continue
        if pd.isna(v):
            d[k] = None
            continue
        if hasattr(v, "item"):
            v = v.item()
        d[k] = v
    return d


def main() -> None:
    train = pd.read_parquet(ROOT / "data" / "clean" / "train.parquet")
    labels = train["isFraud"]

    _train_idx, _val_idx, holdout_idx, cutoff_dt, _val_cutoff_dt = time_split(train, labels)

    recorded = json.loads((ROOT / "outputs" / "split_info.json").read_text())
    assert len(holdout_idx) == recorded["n_holdout"], "recomputed holdout size does not match outputs/split_info.json"
    assert cutoff_dt == recorded["cutoff_dt_train_pool_vs_holdout"], "recomputed cutoff_dt does not match outputs/split_info.json"

    holdout_labels = labels.to_numpy()[holdout_idx]
    fraud_positions = np.where(holdout_labels == 1)[0]
    nonfraud_positions = np.where(holdout_labels == 0)[0]

    picked_fraud = _evenly_spaced(fraud_positions, N_FRAUD)
    picked_nonfraud = _evenly_spaced(nonfraud_positions, N_TOTAL - N_FRAUD)
    picked = np.sort(np.concatenate([picked_fraud, picked_nonfraud]))
    picked_idx = holdout_idx[picked]

    is_float_col = {c: pd.api.types.is_float_dtype(train[c].dtype) for c in train.columns}

    rows = []
    for i in picked_idx:
        i = int(i)
        raw_row = train.iloc[i]
        d = row_to_dict(raw_row, is_float_col)
        rows.append({
            "transaction_id": d.get("TransactionID"),
            "input": d,
            "isFraud": int(raw_row["isFraud"]),
        })

    n_fraud = sum(r["isFraud"] for r in rows)
    assert len(rows) == N_TOTAL
    assert n_fraud >= 40, f"only {n_fraud} fraud rows, need at least 40"
    assert len({r["transaction_id"] for r in rows}) == N_TOTAL, "duplicate TransactionID in canary set"

    (ROOT / "canary").mkdir(exist_ok=True)
    (ROOT / "canary" / "rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(rows)} canary rows ({n_fraud} fraud) to canary/rows.json")


if __name__ == "__main__":
    main()
