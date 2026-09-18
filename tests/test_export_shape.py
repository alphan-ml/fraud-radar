"""Shape tests for export.py, against small synthetic outputs/*.json fixtures
(no dependency on a real trained model)."""
from __future__ import annotations

import json

import pytest

from fraud_radar import export as export_mod


@pytest.fixture
def fake_outputs(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "data_quality.json").write_text(json.dumps({
        "zip_sha256": "deadbeef", "zip_bytes": 123, "train_transaction_rows": 10,
        "train_fraud_count": 1, "train_fraud_rate": 0.1,
        "train_transaction_dt_min": 0, "train_transaction_dt_max": 100,
    }))
    (out_dir / "clean_stats.json").write_text(json.dumps({
        "train_rows": 10, "train_columns": 5, "test_rows": 3,
    }))
    (out_dir / "split_info.json").write_text(json.dumps({
        "n_total": 10, "n_train": 7, "n_val": 1, "n_holdout": 2,
        "cutoff_dt_train_pool_vs_holdout": 80, "val_cutoff_dt_train_vs_val": 70,
        "train_fraud_rate": 0.1, "val_fraud_rate": 0.1, "scale_pos_weight": 9.0,
        "best_iteration": 5, "seed": 26,
    }))
    (out_dir / "metrics.json").write_text(json.dumps({
        "roc_auc": 0.9, "pr_auc": 0.5, "brier_score": 0.01,
        "holdout_n": 2, "holdout_fraud_count": 1, "holdout_fraud_rate": 0.5,
        "review_queue": [{"top_fraction": 0.01, "n_reviewed": 1, "true_positives": 1,
                           "precision": 1.0, "recall": 1.0}],
        "review_policy": {"top_fraction": 0.02, "fitted_cutoff": 0.05, "fitted_on": "validation",
                           "n_validation": 1, "holdout_review_rate": 0.5, "holdout_n_reviewed": 1,
                           "holdout_true_positives": 1, "holdout_precision": 1.0, "holdout_recall": 1.0},
        "calibration_table": [{"bin": 1, "count": 2, "mean_predicted": 0.5,
                                "observed_fraud_rate": 0.5}],
        "top_feature_importance": [{"feature": "amt_log", "gain": 100.0, "gain_share": 1.0}],
    }))
    (out_dir / "holdout_examples.json").write_text(json.dumps([
        {"example_id": f"HOLDOUT-{i:02d}", "amount_log": 1.0, "product_code": "W",
         "calibrated_score": 0.1, "true_label": 0}
        for i in range(1, 11)
    ]))
    monkeypatch.setattr(export_mod, "OUT_DIR", out_dir)
    return out_dir


def test_export_shape(fake_outputs):
    site_data = export_mod.export()
    assert site_data["model"] == "fraud-radar"
    for key in ("dataset_facts", "split", "metrics", "review_queue", "review_policy",
                "calibration_table", "top_feature_importance", "example_rows"):
        assert key in site_data
    assert len(site_data["example_rows"]) == 10
    written = json.loads((fake_outputs / "site_data.json").read_text())
    assert written == site_data
