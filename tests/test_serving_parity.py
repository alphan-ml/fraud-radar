import numpy as np
import pandas as pd

from fraud_radar.serving_parity import (
    compare_scores,
    release_gate,
    single_row_compatible,
)


def test_single_row_mask_does_not_change_other_features_or_source():
    original = pd.DataFrame({
        "V1": [1.0, 2.0],
        "time_since_prev_card1": [15.0, 30.0],
        "time_since_prev_card1_addr1": [20.0, 40.0],
    })
    masked = single_row_compatible(original)
    assert masked["time_since_prev_card1"].isna().all()
    assert masked["time_since_prev_card1_addr1"].isna().all()
    assert masked["V1"].tolist() == [1.0, 2.0]
    assert original["time_since_prev_card1"].notna().all()


def test_identical_probabilities_pass_the_release_gate():
    y = np.array([0, 1] * 100, dtype=int)
    scores = np.arange(len(y), dtype=float) / len(y)
    report = compare_scores(y, scores, scores.copy())
    assert report["review_top_2pct_overlap"] == 1.0
    assert report["mean_abs_probability_shift"] == 0.0
    assert release_gate(report)["passes"]


def test_guard_rejects_degraded_precision_or_recall():
    report = {
        "batch": {"review_queue": [{"top_fraction": 0.02, "precision": 0.8, "recall": 0.44}]},
        "single_row_compatible": {"review_queue": [
            {"top_fraction": 0.02, "precision": 0.70, "recall": 0.43}
        ]},
    }
    assert not release_gate(report)["passes"]
