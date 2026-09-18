"""canary/rows.json is the fixed 500-row canary set used by the Live Eval
canary (src/fraud_radar/canary.py). These tests run without the real,
gitignored Kaggle CSVs -- they only check the committed artifact and the
already-recorded split boundary in outputs/split_info.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS_PATH = ROOT / "canary" / "rows.json"
SPLIT_INFO = json.loads((ROOT / "outputs" / "split_info.json").read_text())


def _reject_constant(name):
    raise ValueError(f"non-JSON constant in canary/rows.json: {name}")


def _load_rows():
    return json.loads(ROWS_PATH.read_text(), parse_constant=_reject_constant)


def test_canary_rows_is_strict_json_with_500_rows():
    rows = _load_rows()
    assert len(rows) == 500
    assert all("input" in r and "isFraud" in r and "transaction_id" in r for r in rows)


def test_canary_rows_has_no_nan_or_infinity_tokens():
    text = ROWS_PATH.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text


def test_canary_rows_at_least_40_fraud():
    rows = _load_rows()
    n_fraud = sum(r["isFraud"] for r in rows)
    assert n_fraud >= 40


def test_canary_rows_are_unique_transactions():
    rows = _load_rows()
    ids = [r["transaction_id"] for r in rows]
    assert len(set(ids)) == len(rows)


def test_canary_rows_are_all_in_the_holdout_split():
    """The holdout split is every row whose TransactionDT falls after the
    time-ordered cutoff recorded at training time
    (outputs/split_info.json's cutoff_dt_train_pool_vs_holdout) -- see
    fraud_radar.model.time_split. Every canary row's TransactionDT must be
    on the holdout side of that cutoff."""
    cutoff_dt = SPLIT_INFO["cutoff_dt_train_pool_vs_holdout"]
    rows = _load_rows()
    for r in rows:
        assert r["input"]["TransactionDT"] >= cutoff_dt, (
            f"transaction {r['transaction_id']} has TransactionDT "
            f"{r['input']['TransactionDT']} before the holdout cutoff {cutoff_dt}"
        )


def test_canary_rows_isfraud_matches_input_label_is_absent():
    """isFraud must not leak into the input dict sent to the live endpoint --
    it is only the label used to score the reported metric."""
    rows = _load_rows()
    assert all("isFraud" not in r["input"] for r in rows)
