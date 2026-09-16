"""Serving module for Fraud Radar.

Loads the trained LightGBM booster and isotonic calibrator once, then
scores one transaction dict at a time (or a small batch), returning a
calibrated fraud probability and a review flag.

This module does not fetch, clean, or train anything -- it only loads
the artifacts written by `fraud_radar.model.train()` (outputs/checkpoints/
lgb_model.txt, calibrator.joblib, feature_columns.json,
categorical_columns.json) and reuses `fraud_radar.features.build_features`
so a raw transaction record is turned into the same feature frame the
model was trained on.

Usage:
    from fraud_radar.score import FraudScorer
    scorer = FraudScorer.load()
    result = scorer.score_one({"TransactionID": 1, "TransactionDT": 86400,
                                "TransactionAmt": 50.0, "ProductCD": "W",
                                "card1": 1234, ...})
    # result == {"calibrated_probability": 0.0123, "review_flag": False}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fraud_radar import model as model_mod
from fraud_radar.features import build_features

# Default review threshold: flag anything the calibrated model scores at or
# above the 2% review-queue operating point from outputs/metrics.json
# (precision 0.7041 / recall 0.4092 on the real holdout -- see README).
DEFAULT_REVIEW_THRESHOLD = 0.02


@dataclass
class FraudScorer:
    booster: Any
    calibrator: Any
    feature_cols: list[str]
    cat_cols: list[str]
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD

    @classmethod
    def load(cls, review_threshold: float = DEFAULT_REVIEW_THRESHOLD) -> FraudScorer:
        booster, calibrator, feature_cols, cat_cols = model_mod.load()
        return cls(booster, calibrator, feature_cols, cat_cols, review_threshold)

    def _to_feature_row(self, transaction: dict) -> pd.DataFrame:
        """Turns one raw transaction dict into a single-row feature frame
        aligned to the columns the model was trained on. Missing columns
        (any column the trained feature set has that this transaction
        doesn't supply) are filled with NaN -- LightGBM handles missing
        values natively, so no imputation is needed here."""
        row = pd.DataFrame([transaction])
        if "TransactionID" not in row.columns:
            row["TransactionID"] = 0
        if "TransactionDT" not in row.columns:
            raise ValueError("transaction dict must include TransactionDT")
        feat = build_features(row)
        for c in self.feature_cols:
            if c not in feat.columns:
                feat[c] = pd.NA
        for c in self.cat_cols:
            if c in feat.columns:
                feat[c] = feat[c].astype("category")
        return feat[self.feature_cols]

    def score_one(self, transaction: dict) -> dict:
        X = self._to_feature_row(transaction)
        proba = float(model_mod.predict_calibrated(self.booster, self.calibrator, X)[0])
        return {
            "calibrated_probability": round(proba, 6),
            "review_flag": proba >= self.review_threshold,
        }
