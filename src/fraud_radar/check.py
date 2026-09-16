"""Day-1 data checks for IEEE-CIS Fraud Detection.

Prints, and writes to outputs/data_quality.json, the real facts about the
five extracted CSVs -- row counts, column counts, sha256 (from fetch.py),
and the raw fraud rate in train_transaction. Every number here comes from
the real CSVs on disk; nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from fraud_radar import fetch as fetch_mod

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "outputs" / "data_quality.json"

EXPECTED_ROW_COUNTS = {
    "train_transaction.csv": 590_540,
    "train_identity.csv": 144_233,
    "test_transaction.csv": 506_691,
    "test_identity.csv": 141_907,
}


def check() -> dict:
    fetch_info = fetch_mod.fetch()

    row_counts = {}
    col_counts = {}
    for name in fetch_mod.EXPECTED_FILES:
        path = RAW_DIR / name
        # count rows without loading the whole file into memory
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        row_counts[name] = n_lines - 1  # minus header
        with open(path, "r") as f:
            header = f.readline()
        col_counts[name] = len(header.strip().split(","))

    mismatches = {}
    for name, expected in EXPECTED_ROW_COUNTS.items():
        actual = row_counts[name]
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}

    train_tx = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=["TransactionID", "isFraud", "TransactionDT"])
    n_train = len(train_tx)
    n_fraud = int(train_tx["isFraud"].sum())
    fraud_rate = round(n_fraud / n_train, 6)
    dt_min = int(train_tx["TransactionDT"].min())
    dt_max = int(train_tx["TransactionDT"].max())

    train_id = pd.read_csv(RAW_DIR / "train_identity.csv", usecols=["TransactionID"])
    n_with_identity = train_tx["TransactionID"].isin(set(train_id["TransactionID"])).sum()

    result = {
        "zip_sha256": fetch_info["zip_sha256"],
        "zip_bytes": fetch_info["zip_bytes"],
        "csv_shas": fetch_info["csv_shas"],
        "row_counts": row_counts,
        "column_counts": col_counts,
        "row_count_mismatches_vs_expected": mismatches,
        "train_transaction_rows": n_train,
        "train_fraud_count": n_fraud,
        "train_fraud_rate": fraud_rate,
        "train_transaction_dt_min": dt_min,
        "train_transaction_dt_max": dt_max,
        "train_transactions_with_identity_row": int(n_with_identity),
        "train_transactions_with_identity_row_pct": round(100 * n_with_identity / n_train, 3),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = check()
    print(f"row_counts: {r['row_counts']}")
    print(f"mismatches vs expected: {r['row_count_mismatches_vs_expected']}")
    print(f"train_transaction_rows: {r['train_transaction_rows']:,}")
    print(f"train_fraud_count: {r['train_fraud_count']:,} ({100 * r['train_fraud_rate']:.3f}%)")
    print(f"TransactionDT range: {r['train_transaction_dt_min']} .. {r['train_transaction_dt_max']}")
    print(f"train rows with an identity row: {r['train_transactions_with_identity_row']:,} "
          f"({r['train_transactions_with_identity_row_pct']}%)")
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)
