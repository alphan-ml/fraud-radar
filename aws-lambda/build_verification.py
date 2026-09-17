import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from fraud_radar.score import FraudScorer

npz = np.load("outputs/checkpoints/split_idx.npz")
holdout_idx = npz["holdout_idx"]

train = pd.read_parquet("data/clean/train.parquet")
is_float_col = {c: pd.api.types.is_float_dtype(train[c].dtype) for c in train.columns}

def row_to_dict(row):
    d = {}
    for k, v in row.items():
        if k == "isFraud":
            continue
        if pd.isna(v):
            d[k] = float("nan") if is_float_col[k] else None
            continue
        if hasattr(v, "item"):
            v = v.item()
        d[k] = v
    return d

scorer = FraudScorer.load()

picked_idx = holdout_idx[:20]
rows = []
for i in picked_idx:
    i = int(i)
    raw_row = train.iloc[i]
    true_label = int(raw_row["isFraud"])
    d = row_to_dict(raw_row)
    local_result = scorer.score_one(d)
    rows.append({
        "transaction_id": d.get("TransactionID"),
        "input": d,
        "true_label": true_label,
        "local_calibrated_probability": local_result["calibrated_probability"],
        "local_review_flag": local_result["review_flag"],
    })

with open("aws-lambda/verification_rows.json", "w") as f:
    json.dump(rows, f, allow_nan=True)

examples = []
for i, r in enumerate(rows[:10]):
    examples.append({
        "example_id": f"HOLDOUT-{i+1:02d}",
        "transaction_id": r["transaction_id"],
        "input": r["input"],
        "true_label": r["true_label"],
        "calibrated_probability": r["local_calibrated_probability"],
        "review_flag": r["local_review_flag"],
    })
def _nan_to_none(obj):
    """examples.json is served to browsers as-is; NaN is not valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


with open("aws-lambda/examples.json", "w") as f:
    json.dump(_nan_to_none(examples), f, allow_nan=False)

print("wrote", len(rows), "verification rows and", len(examples), "examples")
print("sample scores:", [(r["transaction_id"], r["local_calibrated_probability"]) for r in rows[:5]])
