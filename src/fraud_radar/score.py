"""Serving module for Fraud Radar.

Loads the trained LightGBM booster and isotonic calibrator once, then
scores one transaction dict at a time (or a small batch), returning a
calibrated fraud probability and a review flag.

This module does not fetch, clean, or train anything -- it only loads
the artifacts written by `fraud_radar.model.train()` (outputs/checkpoints/
lgb_model.txt, calibrator.joblib, feature_columns.json,
categorical_columns.json, freq_maps.json, review_policy.json) and reuses
`fraud_radar.features.build_features` so a raw transaction record is
turned into the same feature frame the model was trained on -- including
the card1_count/addr1_count/P_emaildomain_count columns, computed from the
saved training-slice frequency maps rather than from the single row being
scored (which would otherwise put every count at 1.0).

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


@dataclass
class FraudScorer:
    booster: Any
    calibrator: Any
    feature_cols: list[str]
    cat_cols: list[str]
    freq_maps: dict[str, dict[str, int]]
    review_threshold: float

    @classmethod
    def load(cls, review_threshold: float | None = None) -> FraudScorer:
        """`review_threshold`, if omitted, defaults to `fitted_cutoff` from
        `outputs/checkpoints/review_policy.json` -- the score at the
        `top_fraction` percentile of the validation slice, not a fixed
        constant (see `fraud_radar.model.train`)."""
        booster, calibrator, feature_cols, cat_cols = model_mod.load()
        freq_maps = model_mod.load_freq_maps()
        if review_threshold is None:
            review_threshold = model_mod.load_review_policy()["fitted_cutoff"]
        return cls(booster, calibrator, feature_cols, cat_cols, freq_maps, review_threshold)

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
        feat = build_features(row, freq_maps=self.freq_maps)
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
