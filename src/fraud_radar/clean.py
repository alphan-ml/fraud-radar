"""Cleaning for IEEE-CIS Fraud Detection.

Rules:
  - train_transaction + train_identity merged on TransactionID (left join --
    most transactions have no identity row; that absence is itself
    signal, encoded downstream as NaN, not dropped).
  - test_transaction + test_identity merged the same way and written
    separately. The test set has NO isFraud labels (it is Kaggle's
    private-leaderboard set) -- it is set aside here and is NEVER used for
    training, validation, calibration, or early stopping anywhere in this
    repo (see tests/test_no_leak.py).
  - Missing-value rates are recorded by column group (identity / card /
    addr / dist / email / C / D / M / V / other) so the feature step can
    make informed choices about what to keep.

Everything here operates on the real CSVs unzipped by fetch.py. No sampling,
no row filtering beyond the merge itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"


def _column_group(col: str) -> str:
    if col.startswith("id_"):
        return "identity"
    if col.startswith("card"):
        return "card"
    if col.startswith("addr"):
        return "addr"
    if col.startswith("dist"):
        return "dist"
    if "emaildomain" in col.lower():
        return "email"
    if col.startswith("C") and col[1:].isdigit():
        return "C"
    if col.startswith("D") and col[1:].isdigit():
        return "D"
    if col.startswith("M") and col[1:].isdigit():
        return "M"
    if col.startswith("V") and col[1:].isdigit():
        return "V"
    if col in ("DeviceType", "DeviceInfo"):
        return "device"
    return "other"


def _missing_rates_by_group(df: pd.DataFrame) -> dict:
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        groups.setdefault(_column_group(c), []).append(c)
    out = {}
    for group, cols in groups.items():
        rates = df[cols].isna().mean()
        out[group] = {
            "n_columns": len(cols),
            "mean_missing_rate": round(float(rates.mean()), 4),
            "min_missing_rate": round(float(rates.min()), 4),
            "max_missing_rate": round(float(rates.max()), 4),
        }
    return out


def merge_frames(tx: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    """Left join transaction onto identity on TransactionID. Every transaction
    row is kept even when it has no identity row (that absence is itself
    signal, encoded downstream as NaN, not dropped)."""
    return tx.merge(idn, on="TransactionID", how="left")


def _merge_one(split: str) -> pd.DataFrame:
    tx = pd.read_csv(RAW_DIR / f"{split}_transaction.csv")
    idn = pd.read_csv(RAW_DIR / f"{split}_identity.csv")
    return merge_frames(tx, idn)


def clean(force: bool = False) -> dict:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    train_path = CLEAN_DIR / "train.parquet"
    test_path = CLEAN_DIR / "test.parquet"

    stats = {}

    if train_path.exists() and test_path.exists() and not force:
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
    else:
        train = _merge_one("train")
        test = _merge_one("test")
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

    n_train = len(train)
    n_test = len(test)
    n_train_with_identity = int(train["id_01"].notna().sum()) if "id_01" in train.columns else None
    n_test_with_identity = int(test["id_01"].notna().sum()) if "id_01" in test.columns else None

    fraud_rate = round(float(train["isFraud"].mean()), 6)
    dt_min = int(train["TransactionDT"].min())
    dt_max = int(train["TransactionDT"].max())
    dt_min_test = int(test["TransactionDT"].min())
    dt_max_test = int(test["TransactionDT"].max())

    stats = {
        "train_rows": n_train,
        "test_rows": n_test,
        "train_columns": train.shape[1],
        "test_columns": test.shape[1],
        "train_rows_with_identity": n_train_with_identity,
        "test_rows_with_identity": n_test_with_identity,
        "train_fraud_rate": fraud_rate,
        "train_fraud_count": int(train["isFraud"].sum()),
        "train_transaction_dt_min": dt_min,
        "train_transaction_dt_max": dt_max,
        "test_transaction_dt_min": dt_min_test,
        "test_transaction_dt_max": dt_max_test,
        "test_has_isfraud_column": "isFraud" in test.columns,
        "missing_rates_by_group_train": _missing_rates_by_group(train),
    }

    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "clean_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats


if __name__ == "__main__":
    import sys

    r = clean(force="--force" in sys.argv)
    print(f"train: {r['train_rows']:,} rows x {r['train_columns']} cols "
          f"({r['train_rows_with_identity']:,} with an identity row)")
    print(f"test:  {r['test_rows']:,} rows x {r['test_columns']} cols "
          f"({r['test_rows_with_identity']:,} with an identity row), "
          f"has isFraud column: {r['test_has_isfraud_column']}")
    print(f"train fraud rate: {r['train_fraud_rate']:.4%} ({r['train_fraud_count']:,} fraud rows)")
    print(f"train TransactionDT range: {r['train_transaction_dt_min']} .. {r['train_transaction_dt_max']}")
