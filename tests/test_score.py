"""Proves the serving module (FraudScorer) loads the real trained artifacts
and produces a calibrated probability consistent with the batch prediction
path used at train/eval time, for a genuine raw transaction record."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_radar.score import FraudScorer

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "outputs" / "checkpoints"

pytestmark = pytest.mark.skipif(
    not (CKPT_DIR / "lgb_model.txt").exists(),
    reason="requires a trained model in outputs/checkpoints (run `fraud-radar train` first)",
)


def _raw_train_row() -> dict:
    """Grabs one real row from data/clean/train.parquet (if present) so the
    scorer is exercised on genuine data, not a hand-typed fixture."""
    train_path = ROOT / "data" / "clean" / "train.parquet"
    if not train_path.exists():
        pytest.skip("requires data/clean/train.parquet")
    df = pd.read_parquet(train_path)
    row = df.drop(columns=["isFraud"], errors="ignore").iloc[0]
    return row.to_dict()


def test_score_one_returns_probability_and_flag():
    scorer = FraudScorer.load()
    txn = _raw_train_row()
    result = scorer.score_one(txn)

    assert "calibrated_probability" in result
    assert "review_flag" in result
    assert 0.0 <= result["calibrated_probability"] <= 1.0
    assert isinstance(result["review_flag"], bool)


def test_score_one_matches_batch_prediction_path():
    from fraud_radar import model as model_mod
    from fraud_radar.features import build_features

    scorer = FraudScorer.load()
    txn = _raw_train_row()

    result = scorer.score_one(txn)

    raw_df = pd.DataFrame([txn])
    feat = build_features(raw_df, freq_maps=scorer.freq_maps)
    for c in scorer.feature_cols:
        if c not in feat.columns:
            feat[c] = pd.NA
    for c in scorer.cat_cols:
        if c in feat.columns:
            feat[c] = feat[c].astype("category")
    X = feat[scorer.feature_cols]
    expected = float(
        model_mod.predict_calibrated(scorer.booster, scorer.calibrator, X)[0]
    )

    assert result["calibrated_probability"] == round(expected, 6)


def test_review_threshold_is_configurable():
    scorer_low = FraudScorer.load(review_threshold=0.0)
    scorer_high = FraudScorer.load(review_threshold=1.0)
    txn = _raw_train_row()

    assert scorer_low.score_one(txn)["review_flag"] is True
    assert scorer_high.score_one(txn)["review_flag"] is False


def test_default_review_threshold_is_the_saved_fitted_cutoff():
    from fraud_radar import model as model_mod

    policy = model_mod.load_review_policy()
    scorer = FraudScorer.load()

    assert scorer.review_threshold == policy["fitted_cutoff"]


def test_score_one_matches_the_same_row_scored_inside_a_batch():
    """card1_count/addr1_count/P_emaildomain_count come from the saved
    frequency maps, not from value_counts() on whatever frame happens to be
    passed in -- so a row's calibrated score must be identical whether it is
    scored alone or as part of a larger batch."""
    from fraud_radar import model as model_mod
    from fraud_radar.features import build_features

    scorer = FraudScorer.load()
    train_path = ROOT / "data" / "clean" / "train.parquet"
    if not train_path.exists():
        pytest.skip("requires data/clean/train.parquet")
    df = pd.read_parquet(train_path).drop(columns=["isFraud"], errors="ignore")
    batch = df.iloc[:5].reset_index(drop=True)

    single_result = scorer.score_one(batch.iloc[0].to_dict())

    feat_batch = build_features(batch, freq_maps=scorer.freq_maps)
    for c in scorer.cat_cols:
        if c in feat_batch.columns:
            feat_batch[c] = feat_batch[c].astype("category")
    X_batch = feat_batch[scorer.feature_cols]
    batch_scores = model_mod.predict_calibrated(scorer.booster, scorer.calibrator, X_batch)

    assert round(float(batch_scores[0]), 6) == single_result["calibrated_probability"]
