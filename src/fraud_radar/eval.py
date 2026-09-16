"""Evaluation for Fraud Radar. Everything here runs on the time-ordered 20%
holdout, which the model never sees during training, early stopping, or
calibration. Nothing on the site is typed by hand -- every number comes
from outputs/metrics.json, written by this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"

REVIEW_QUEUE_FRACTIONS = [0.005, 0.01, 0.02, 0.05]


def review_queue_table(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    n = len(y_true)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = int(y_true.sum())
    rows = []
    for frac in REVIEW_QUEUE_FRACTIONS:
        k = max(1, round(n * frac))
        top_k = y_sorted[:k]
        tp = int(top_k.sum())
        precision = tp / k
        recall = tp / total_pos if total_pos else float("nan")
        rows.append({
            "top_fraction": frac,
            "n_reviewed": k,
            "true_positives": tp,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        })
    return rows


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> list[dict]:
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    p_sorted = y_score[order]
    n = len(y_true)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        bin_p = p_sorted[lo:hi]
        bin_y = y_sorted[lo:hi]
        rows.append({
            "bin": i + 1,
            "count": int(hi - lo),
            "mean_predicted": round(float(bin_p.mean()), 6),
            "observed_fraud_rate": round(float(bin_y.mean()), 6),
        })
    return rows


def feature_importance_table(booster, feature_cols: list[str], top_n: int = 15) -> list[dict]:
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gains)[:top_n]
    total = float(gains.sum()) or 1.0
    return [
        {
            "feature": feature_cols[i],
            "gain": float(gains[i]),
            "gain_share": round(float(gains[i]) / total, 4),
        }
        for i in order
    ]


def evaluate(y_true: np.ndarray, y_score: np.ndarray, booster, feature_cols: list[str]) -> dict:
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))

    metrics = {
        "holdout_n": len(y_true),
        "holdout_fraud_count": int(y_true.sum()),
        "holdout_fraud_rate": round(float(y_true.mean()), 6),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "review_queue": review_queue_table(y_true, y_score),
        "calibration_table": calibration_table(y_true, y_score),
        "top_feature_importance": feature_importance_table(booster, feature_cols),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
