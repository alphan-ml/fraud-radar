"""Tests for the train-time frequency-map and review-policy fitting in
model.train() -- both must be fit on the train slice / validation slice
only, never on the full population or on rows past their intended split."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fraud_radar import model as model_mod

N = 300


def _synthetic_feat_labels(n: int = N, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    dt = np.arange(n) * 100  # already time-ordered, so positional order == TransactionDT order

    # "card4" only ever appears in the last 10 rows, which land in the
    # holdout (last 20%) for n=300 -- never in the train or validation slice.
    card1_values = [f"card{i % 4}" for i in range(n - 10)] + ["card4"] * 10
    card1 = pd.array(card1_values, dtype="string").astype("category")
    addr1 = pd.array(["addr0"] * n, dtype="string").astype("category")
    email = pd.array(["gmail.com"] * n, dtype="string").astype("category")
    uid_values = [f"uid{i % 4}" for i in range(n - 10)] + ["uid4"] * 10
    uid = pd.array(uid_values, dtype="string").astype("category")

    feat = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": dt,
        "x1": rng.normal(size=n).astype("float32"),
        "amt_log": rng.uniform(0.0, 5.0, size=n).astype("float32"),
        "card1": card1,
        "addr1": addr1,
        "P_emaildomain": email,
        "uid": uid,
    })
    labels = pd.Series((rng.random(n) < 0.2).astype(int))
    return feat, labels


def _patch_for_fast_train(monkeypatch, tmp_path):
    monkeypatch.setattr(model_mod, "ROOT", tmp_path)
    monkeypatch.setattr(model_mod, "CKPT_DIR", tmp_path / "outputs" / "checkpoints")
    monkeypatch.setattr(model_mod, "NUM_BOOST_ROUND", 20)
    monkeypatch.setattr(model_mod, "EARLY_STOPPING_ROUNDS", 5)


def test_freq_maps_are_fit_on_train_slice_only(monkeypatch, tmp_path):
    _patch_for_fast_train(monkeypatch, tmp_path)
    feat, labels = _synthetic_feat_labels()

    train_idx = model_mod.time_split(feat, labels)[0]
    expected_card0_count = int((feat["card1"].astype(str).iloc[train_idx] == "card0").sum())

    model_mod.train(feat, labels)

    freq_maps = json.loads((tmp_path / "outputs" / "checkpoints" / "freq_maps.json").read_text())
    assert "card4" not in freq_maps["card1"]  # never seen in the train slice
    assert freq_maps["card1"]["card0"] == expected_card0_count


def test_group_stats_are_fit_on_train_slice_only(monkeypatch, tmp_path):
    _patch_for_fast_train(monkeypatch, tmp_path)
    feat, labels = _synthetic_feat_labels()

    train_idx = model_mod.time_split(feat, labels)[0]
    train_card1_addr1 = (
        feat["card1"].astype(str).iloc[train_idx] + "||" + feat["addr1"].astype(str).iloc[train_idx]
    )
    expected_card0_addr0_count = int((train_card1_addr1 == "card0||addr0").sum())

    model_mod.train(feat, labels)

    group_stats = json.loads((tmp_path / "outputs" / "checkpoints" / "group_stats.json").read_text())
    # card4 only appears in the last 10 rows (the holdout) -- never in train.
    assert "card4||addr0" not in group_stats["card1_addr1"]["count"]
    assert group_stats["card1_addr1"]["count"]["card0||addr0"] == expected_card0_addr0_count
    assert "mean_amt" in group_stats["card1_addr1"]
    assert "mean_amt" in group_stats["card1"]


def test_review_policy_is_saved_with_expected_shape(monkeypatch, tmp_path):
    _patch_for_fast_train(monkeypatch, tmp_path)
    feat, labels = _synthetic_feat_labels()

    info = model_mod.train(feat, labels)

    policy = json.loads((tmp_path / "outputs" / "checkpoints" / "review_policy.json").read_text())
    assert policy["top_fraction"] == model_mod.REVIEW_TOP_FRACTION
    assert policy["fitted_on"] == "validation"
    assert isinstance(policy["fitted_cutoff"], float)
    assert policy["n_validation"] == info["n_val"] > 0
