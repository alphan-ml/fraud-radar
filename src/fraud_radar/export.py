"""Export the site-facing data artifact for Fraud Radar.

Writes outputs/site_data.json: dataset facts, the exact time-based split
cutoff, all holdout metrics, the calibration table, the review-queue table,
top feature importance, and 10 anonymized example holdout rows. Every
number is read back from files written by earlier pipeline steps -- nothing
is typed by hand here.

No site file on giggitai.com is read or written by this repo (standing
rule -- see CONTEXT.md).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def export() -> dict:
    data_quality = json.loads((OUT_DIR / "data_quality.json").read_text())
    clean_stats = json.loads((OUT_DIR / "clean_stats.json").read_text())
    split_info = json.loads((OUT_DIR / "split_info.json").read_text())
    metrics = json.loads((OUT_DIR / "metrics.json").read_text())
    examples = json.loads((OUT_DIR / "holdout_examples.json").read_text())

    site_data = {
        "model": "fraud-radar",
        "dataset_facts": {
            "source": "Kaggle competition ieee-fraud-detection (IEEE-CIS Fraud Detection)",
            "zip_sha256": data_quality["zip_sha256"],
            "zip_bytes": data_quality["zip_bytes"],
            "train_transaction_rows": data_quality["train_transaction_rows"],
            "train_fraud_count": data_quality["train_fraud_count"],
            "train_fraud_rate": data_quality["train_fraud_rate"],
            "train_transaction_dt_min": data_quality["train_transaction_dt_min"],
            "train_transaction_dt_max": data_quality["train_transaction_dt_max"],
            "merged_train_rows": clean_stats["train_rows"],
            "merged_train_columns": clean_stats["train_columns"],
            "test_rows_set_aside_no_labels": clean_stats["test_rows"],
        },
        "split": {
            "n_total": split_info["n_total"],
            "n_train": split_info["n_train"],
            "n_val": split_info["n_val"],
            "n_holdout": split_info["n_holdout"],
            "cutoff_dt_train_pool_vs_holdout": split_info["cutoff_dt_train_pool_vs_holdout"],
            "val_cutoff_dt_train_vs_val": split_info["val_cutoff_dt_train_vs_val"],
            "train_fraud_rate": split_info["train_fraud_rate"],
            "val_fraud_rate": split_info["val_fraud_rate"],
            "scale_pos_weight": split_info["scale_pos_weight"],
            "best_iteration": split_info["best_iteration"],
            "seed": split_info["seed"],
        },
        "metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
            "holdout_n": metrics["holdout_n"],
            "holdout_fraud_count": metrics["holdout_fraud_count"],
            "holdout_fraud_rate": metrics["holdout_fraud_rate"],
        },
        "review_queue": metrics["review_queue"],
        "review_policy": metrics["review_policy"],
        "calibration_table": metrics["calibration_table"],
        "top_feature_importance": metrics["top_feature_importance"],
        "example_rows": examples,
    }
    (OUT_DIR / "site_data.json").write_text(json.dumps(site_data, indent=2, default=str))
    return site_data


if __name__ == "__main__":
    export()
