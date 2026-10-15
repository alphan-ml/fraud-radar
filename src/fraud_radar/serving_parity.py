"""Compare fraud batch evaluation to single-transaction serving-compatible features.

Does not claim Lambda equivalence, and does not tune/promote a model on holdout.
The AWS handler scores one transaction and always sets historical deltas to NaN.
Run: python -m fraud_radar.serving_parity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_radar import eval as metrics
from fraud_radar import features, model

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
SINGLE_ROW_MISSING = ("time_since_prev_card1", "time_since_prev_card1_addr1")


def single_row_compatible(frame: pd.DataFrame) -> pd.DataFrame:
    """Zero out no other information: only historical features unavailable to Lambda."""
    result = frame.copy()
    for col in SINGLE_ROW_MISSING:
        if col in result.columns:
            result[col] = np.nan
    return result


def compare_scores(labels: np.ndarray, batch_scores: np.ndarray, serving_like_scores: np.ndarray) -> dict:
    """Diagnostic comparison at identical scores, labels and fixed review budgets."""
    y = np.asarray(labels)
    raw = np.asarray(batch_scores)
    simulated = np.asarray(serving_like_scores)
    if len(y) != len(raw) or len(y) != len(simulated) or len(y) < 2:
        raise ValueError("invalid parity diagnostic arrays")
    k = max(1, round(len(y) * 0.02))
    top_batch = set(np.argsort(-raw, kind="stable")[:k])
    top_serving = set(np.argsort(-simulated, kind="stable")[:k])
    queues = {
        "batch": metrics.review_queue_table(y, raw),
        "single_row_compatible": metrics.review_queue_table(y, simulated),
    }
    return {
        "holdout_n": len(y),
        "feature_compatibility_check": list(SINGLE_ROW_MISSING),
        "mean_abs_probability_shift": float(np.mean(np.abs(raw - simulated))),
        "max_abs_probability_shift": float(np.max(np.abs(raw - simulated))),
        "pct_predictions_shifted_more_than_1pp": float((np.abs(raw - simulated) > 0.01).mean()),
        "review_top_2pct_overlap": float(len(top_batch & top_serving) / k),
        "batch": {
            "roc_auc": float(roc_auc_score(y, raw)),
            "pr_auc": float(average_precision_score(y, raw)),
            "review_queue": queues["batch"],
        },
        "single_row_compatible": {
            "roc_auc": float(roc_auc_score(y, simulated)),
            "pr_auc": float(average_precision_score(y, simulated)),
            "review_queue": queues["single_row_compatible"],
        },
        "limitation": (
            "Local checkpoint, simulated missing historical features only. "
            "This is not an independent end-to-end Lambda deployment check. "
            "Do not claim offline/serving parity or promote models based on this report."
        ),
    }


def main() -> None:
    feat_path = OUT_DIR / "features.parquet"
    label_path = OUT_DIR / "labels.parquet"
    if not feat_path.exists() or not label_path.exists():
        raise SystemExit("Run the existing fraud features/train pipeline with the original dataset first")
    feat = pd.read_parquet(feat_path)
    y = pd.read_parquet(label_path)["isFraud"].to_numpy()
    booster, calibrator, columns, _ = model.load()
    feat = features.apply_freq_maps(feat, model.load_freq_maps())
    feat = features.apply_group_stats(feat, model.load_group_stats())
    feat["amt_to_card1_mean_ratio"] = (
        feat["amt_log"] / feat["card1_mean_amt"].replace(0.0, np.nan)
    ).astype("float32")
    holdout = model.load_split_idx()["holdout_idx"]
    X = feat[columns].iloc[holdout]
    y = y[holdout]
    batch = model.predict_calibrated(booster, calibrator, X)
    serving_like = model.predict_calibrated(booster, calibrator, single_row_compatible(X))
    report = compare_scores(y, batch, serving_like)
    report["model_origin"] = "local saved checkpoint; deployment version not verified"
    result_path = OUT_DIR / "serving_parity_diagnostic.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(result_path), "review_top_2pct_overlap": report["review_top_2pct_overlap"]}))


if __name__ == "__main__":
    main()
