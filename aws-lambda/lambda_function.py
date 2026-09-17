"""Fraud Radar -- Lambda scoring handler.

Loads the trained LightGBM booster (outputs/checkpoints/lgb_model.txt) and
the isotonic calibrator (reduced to its sorted-threshold interpolation
table -- see isotonic_thresholds.json, verified 0.0 max abs diff against
sklearn's IsotonicRegression.predict, out_of_bounds="clip") from S3, then
reproduces fraud_radar.features.build_features() and
fraud_radar.score.FraudScorer.score_one() in pure numpy (no pandas/sklearn
bundled) -- verified byte-identical against the real score_one() on 20
real IEEE-CIS holdout transactions, max abs diff 0.0 (see
aws-lambda/verification_rows.json / README.md).

Category codes for the 14 LightGBM categorical columns (card1-6, addr1,
addr2, P_emaildomain, R_emaildomain, ProductCD, DeviceType, DeviceInfo,
M4) come from the "pandas_categorical" section embedded in the trained
model file itself -- the exact codes LightGBM used at training time, not
recomputed -- so category encoding is byte-identical by construction.
Count-encoding maps (card1_count, addr1_count, P_emaildomain_count) are
the same `outputs/checkpoints/freq_maps.json` fit on the training slice by
`fraud_radar.model.train()` (count_maps.json here), downloaded from S3
like the other artifacts -- not recomputed and not hard-coded, so a
serving-time lookup for a value unseen in training slice comes out 0,
matching `fraud_radar.features.apply_freq_maps`. The review-queue cutoff
(review_policy.json, also from S3) is likewise the cutoff fit on the
validation slice, not a hard-coded probability.
"""
from __future__ import annotations

import base64
import json
import math
import os

import boto3
import lightgbm as lgb
import numpy as np

S3_BUCKET = os.environ.get("MODEL_BUCKET", "giggit-fraud-radar-models")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models")

ASSET_FILES = [
    "lgb_model.txt",
    "feature_columns.json",
    "cat_code_maps.json",
    "isotonic_thresholds.json",
    "count_maps.json",
    "review_policy.json",
]

M_BINARY_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
CATEGORICAL_COLS = [
    "M4", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "ProductCD",
    "DeviceType", "DeviceInfo",
]
ID_NUMERIC_COLS = [
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08",
    "id_09", "id_10", "id_11", "id_13", "id_14", "id_17", "id_18", "id_19",
    "id_20", "id_21", "id_22", "id_24", "id_25", "id_26", "id_32",
]
COUNT_ENCODE_COLS = ["card1", "addr1", "P_emaildomain"]
REQUIRED_FIELDS = ["TransactionID", "TransactionDT", "TransactionAmt"]

_booster = None
_feature_cols = None
_cat_code_maps = None
_iso_X_thresholds = None
_iso_y_thresholds = None
_iso_X_min = None
_iso_X_max = None
_count_maps = None
_review_threshold = None


def _load_models() -> None:
    global _booster, _feature_cols, _cat_code_maps
    global _iso_X_thresholds, _iso_y_thresholds, _iso_X_min, _iso_X_max
    global _count_maps, _review_threshold
    if _booster is not None:
        return
    s3 = boto3.client("s3")
    for fname in ASSET_FILES:
        local_path = f"/tmp/{fname}"
        if not os.path.exists(local_path):
            s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/{fname}", local_path)

    _booster = lgb.Booster(model_file="/tmp/lgb_model.txt")
    with open("/tmp/feature_columns.json") as f:
        _feature_cols = json.loads(f.read())
    with open("/tmp/cat_code_maps.json") as f:
        _cat_code_maps = json.loads(f.read())
    with open("/tmp/isotonic_thresholds.json") as f:
        iso = json.loads(f.read())
    _iso_X_thresholds = np.array(iso["X_thresholds"], dtype="float64")
    _iso_y_thresholds = np.array(iso["y_thresholds"], dtype="float64")
    _iso_X_min = float(iso["X_min"])
    _iso_X_max = float(iso["X_max"])
    with open("/tmp/count_maps.json") as f:
        _count_maps = json.loads(f.read())
    with open("/tmp/review_policy.json") as f:
        _review_threshold = float(json.loads(f.read())["fitted_cutoff"])


def _num(transaction: dict, key: str) -> float:
    v = transaction.get(key)
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(f):
        return float("nan")
    return f


def _cat_code(transaction: dict, col: str) -> float:
    v = transaction.get(col)
    if v is None:
        return float("nan")
    key = str(v)
    code = _cat_code_maps.get(col, {}).get(key)
    if code is None:
        return float("nan")
    return float(code)


def build_features_row(transaction: dict) -> np.ndarray:
    """Builds one feature row (float64, column order = feature_columns.json)
    from a raw transaction dict -- mirrors fraud_radar.features.build_features
    exactly (verified byte-identical, see module docstring)."""
    dt = transaction.get("TransactionDT")
    if dt is None:
        raise ValueError("transaction dict must include TransactionDT")
    dt = float(dt)

    values: dict[str, float] = {}
    values["hour_of_day"] = float(int(dt // 3600) % 24)
    values["day_index"] = float(int(dt // 86400))

    amt = _num(transaction, "TransactionAmt")
    amt = max(amt, 0.0) if not math.isnan(amt) else amt
    values["amt_log"] = math.log1p(amt) if not math.isnan(amt) else float("nan")

    for col in _feature_cols:
        if col in values:
            continue
        if (
            (col.startswith("V") and col[1:].isdigit())
            or (col.startswith("C") and col[1:].isdigit())
            or (col.startswith("D") and col[1:].isdigit())
        ):
            values[col] = _num(transaction, col)
        elif col in M_BINARY_COLS:
            v = transaction.get(col)
            if v == "T":
                values[col] = 1.0
            elif v == "F":
                values[col] = 0.0
            else:
                values[col] = float("nan")
        elif col in CATEGORICAL_COLS:
            values[col] = _cat_code(transaction, col)
        elif col in ID_NUMERIC_COLS:
            values[col] = _num(transaction, col)
        elif col.endswith("_count") and col[: -len("_count")] in COUNT_ENCODE_COLS:
            base_col = col[: -len("_count")]
            v = transaction.get(base_col)
            key = "nan" if v is None else str(v)
            values[col] = float(_count_maps.get(base_col, {}).get(key, 0))
        else:
            values[col] = float("nan")

    row = np.array([values[c] for c in _feature_cols], dtype="float64")
    # fraud_radar.features.build_features() casts every numeric feature to
    # float32 before it reaches the booster (df_dtypes promotion in
    # lightgbm's _data_from_pandas ends up at float32 for this feature set)
    # -- round-trip through float32 here so split-threshold comparisons see
    # bit-identical values.
    row = row.astype("float32").astype("float64")
    return row


def predict_calibrated(row: np.ndarray) -> float:
    raw = _booster.predict(row.reshape(1, -1), num_iteration=_booster.best_iteration)[0]
    clipped = min(max(raw, _iso_X_min), _iso_X_max)
    calibrated = float(np.interp(clipped, _iso_X_thresholds, _iso_y_thresholds))
    return calibrated


def score_one(transaction: dict, review_threshold: float | None = None) -> dict:
    _load_models()
    row = build_features_row(transaction)
    proba = predict_calibrated(row)
    if review_threshold is None:
        review_threshold = _review_threshold
    return {
        "calibrated_probability": round(proba, 6),
        "review_flag": proba >= review_threshold,
    }


_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

_ALLOWED_ORIGINS = {"https://www.giggitai.com", "https://giggitai.com"}


def _nan_to_none(obj):
    """Replaces float NaN/inf with None, recursively. Python's json.dumps
    writes NaN as the bare token `NaN`, which is not JSON: a browser's
    JSON.parse rejects the whole body. That is what broke the site's
    "Score a Transaction" example picker (/examples carried 1,294 NaN
    values from the raw holdout rows). The scoring path already treats
    None as missing, so null round-trips through POST /score unchanged."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _resp(status: int, body, origin: str | None = None) -> dict:
    headers = dict(_HEADERS)
    if origin in _ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
    return {
        "statusCode": status,
        "headers": headers,
        "body": body if isinstance(body, str) else json.dumps(_nan_to_none(body), allow_nan=False),
    }


def handler(event, context):
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    )
    path = event.get("rawPath") or event.get("path") or ""
    headers_in = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    origin = headers_in.get("origin")

    if method == "OPTIONS":
        return _resp(200, "", origin)

    if path.endswith("/health"):
        return _resp(200, {"status": "ok"}, origin)

    if path.endswith("/examples"):
        try:
            _load_models()
            with open("/tmp/examples.json") as f:
                examples = json.load(f)
        except FileNotFoundError:
            s3 = boto3.client("s3")
            s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/examples.json", "/tmp/examples.json")
            with open("/tmp/examples.json") as f:
                examples = json.load(f)
        return _resp(200, {"examples": examples}, origin)

    if method != "POST":
        return _resp(405, {"error": "use POST /score"}, origin)

    try:
        raw_body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode()
        payload = json.loads(raw_body)
        missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] is None]
        if missing:
            return _resp(400, {"error": f"missing fields: {missing}"}, origin)
        result = score_one(payload)
        return _resp(200, result, origin)
    except ValueError as e:
        return _resp(400, {"error": str(e)}, origin)
    except json.JSONDecodeError:
        return _resp(400, {"error": "invalid JSON body"}, origin)
    except Exception as e:  # noqa: BLE001
        return _resp(500, {"error": "internal error", "detail": str(e)}, origin)
