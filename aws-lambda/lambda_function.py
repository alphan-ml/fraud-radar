"""Fraud Radar -- Lambda scoring handler.

Loads the trained LightGBM booster (outputs/checkpoints/lgb_model.txt) and
the isotonic calibrator (reduced to its sorted-threshold interpolation
table -- see isotonic_thresholds.json, verified 0.0 max abs diff against
sklearn's IsotonicRegression.predict, out_of_bounds="clip") from S3, then
reproduces fraud_radar.features.build_features() and
fraud_radar.score.FraudScorer.score_one() in pure numpy (no pandas/sklearn
bundled) -- see README's "How the handler mirrors score.py" and
`aws-lambda/DEPLOY.md`.

Category codes for the 14 LightGBM categorical columns (card1-6, addr1,
addr2, P_emaildomain, R_emaildomain, ProductCD, DeviceType, DeviceInfo,
M4) plus `uid` (feature pass F3) come from the "pandas_categorical" section
embedded in the trained model file itself -- the exact codes LightGBM used
at training time, not recomputed -- so category encoding is byte-identical
by construction. Count-encoding maps (card1_count, addr1_count,
P_emaildomain_count, uid_count) are the same
`outputs/checkpoints/freq_maps.json` fit on the training slice by
`fraud_radar.model.train()` (count_maps.json here), downloaded from S3
like the other artifacts -- not recomputed and not hard-coded, so a
serving-time lookup for a value unseen in training slice comes out 0,
matching `fraud_radar.features.apply_freq_maps`. The review-queue cutoff
(review_policy.json, also from S3) is likewise the cutoff fit on the
validation slice, not a hard-coded probability.

Feature pass F3 additions -- the `card1`/`card1_addr1`/`card1_P_emaildomain`
count and mean(amt_log) group aggregates (from `group_stats.json`, also
downloaded from S3 rather than bundled in the deploy zip: it is several MB,
the same size class as `count_maps.json`), `amt_to_card1_mean_ratio`, and
the `uid` categorical plus its `uid_count` -- are computed here the same
way `fraud_radar.features.build_features()`/`apply_group_stats()` compute
them from `group_stats_maps`. This is a reimplementation, not a call into
`fraud_radar.score`: `score.py` exposes no bare, pandas-free function that
builds a feature row (its `_to_feature_row` is a `FraudScorer` method that
calls the pandas-based `build_features()`), and bundling pandas/
scikit-learn to call it verbatim was already rejected on Lambda package
size grounds (see README, "No pandas, no scikit-learn bundled").
`time_since_prev_card1`/`time_since_prev_card1_addr1` are always NaN here,
matching `score.py`'s own single-row `score_one()` -- a lone transaction
has no prior-transaction history to diff against no matter which code path
computes it.
"""
from __future__ import annotations

import base64
import json
import math
import os

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
    "group_stats.json",
    "review_policy.json",
]

M_BINARY_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
CATEGORICAL_COLS = [
    "M4", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "ProductCD",
    "DeviceType", "DeviceInfo", "uid",
]
ID_NUMERIC_COLS = [
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08",
    "id_09", "id_10", "id_11", "id_13", "id_14", "id_17", "id_18", "id_19",
    "id_20", "id_21", "id_22", "id_24", "id_25", "id_26", "id_32",
]
COUNT_ENCODE_COLS = ["card1", "addr1", "P_emaildomain", "uid"]
REQUIRED_FIELDS = ["TransactionID", "TransactionDT", "TransactionAmt"]

# Feature pass F3 group aggregates -- mirrors fraud_radar.features.GROUP_STATS_DEFS
# exactly (group_name, group_cols, stats). Duplicated here as a plain literal,
# not imported, so the handler never has to import fraud_radar.features (which
# pulls in pandas) -- see module docstring.
GROUP_STATS_DEFS = [
    ("card1", ["card1"], ["mean_amt"]),
    ("card1_addr1", ["card1", "addr1"], ["count", "mean_amt"]),
    ("card1_P_emaildomain", ["card1", "P_emaildomain"], ["count", "mean_amt"]),
]

_booster = None
_feature_cols = None
_cat_code_maps = None
_iso_X_thresholds = None
_iso_y_thresholds = None
_iso_X_min = None
_iso_X_max = None
_count_maps = None
_group_stats = None
_review_threshold = None


def _load_models() -> None:
    global _booster, _feature_cols, _cat_code_maps
    global _iso_X_thresholds, _iso_y_thresholds, _iso_X_min, _iso_X_max
    global _count_maps, _group_stats, _review_threshold
    if _booster is not None:
        return
    import boto3  # in the Lambda runtime; imported here so the tests load the module without it

    s3 = boto3.client("s3")
    for fname in ASSET_FILES:
        local_path = f"/tmp/{fname}"
        if os.path.exists(local_path):
            continue
        try:
            s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/{fname}", local_path)
        except Exception:
            # group_stats.json exists only for feature-pass F3 and later. A prefix
            # without it (the F2 model) still loads; the F3 columns are simply not in
            # that prefix's feature_columns.json, so their values are never selected.
            if fname == "group_stats.json":
                with open(local_path, "w") as f:
                    f.write("{}")
            else:
                raise

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
    with open("/tmp/group_stats.json") as f:
        _group_stats = json.loads(f.read())
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


def _cat_code_for_key(col: str, key: str | None) -> float:
    if key is None:
        return float("nan")
    code = _cat_code_maps.get(col, {}).get(key)
    if code is None:
        return float("nan")
    return float(code)


def _cat_code(transaction: dict, col: str) -> float:
    v = transaction.get(col)
    if v is None:
        return float("nan")
    return _cat_code_for_key(col, str(v))


def _key_part(transaction: dict, col: str) -> str:
    """Stringifies one raw field the way fraud_radar.features._stringify
    stringifies a single-row column: missing -> "nan", otherwise str() of
    the value -- matches str(v) on the raw JSON payload type (int/float/str)
    the same way a one-row pandas DataFrame built from that same dict would
    infer a dtype from it."""
    v = transaction.get(col)
    return "nan" if v is None else str(v)


def _composite_key(transaction: dict, cols: list[str]) -> str:
    """Mirrors fraud_radar.features._composite_key: cols joined by "||"."""
    return "||".join(_key_part(transaction, c) for c in cols)


def _uid_key(transaction: dict, day_index: float) -> str:
    """Mirrors fraud_radar.features._compute_uid for a single row:
    card1||addr1||account_start_day, account_start_day = day_index - D1,
    rounded to match pandas' Series.round(); a missing D1 collapses to a
    shared "nan" bucket rather than a distinct value per row."""
    d1 = _num(transaction, "D1")
    account_start_day = day_index - d1
    if math.isnan(account_start_day):
        start_day_str = "nan"
    else:
        start_day_str = str(float(np.round(account_start_day)))
    return f"{_key_part(transaction, 'card1')}||{_key_part(transaction, 'addr1')}||{start_day_str}"


def _group_stat(name: str, cols: list[str], stat: str, transaction: dict) -> float:
    """Mirrors fraud_radar.features.apply_group_stats: an unseen composite
    key maps to 0 for count, NaN for mean_amt (not a fabricated 0.0 mean)."""
    key = _composite_key(transaction, cols)
    val = _group_stats.get(name, {}).get(stat, {}).get(key)
    if val is None:
        return 0.0 if stat == "count" else float("nan")
    return float(val)


def build_features_row(transaction: dict) -> np.ndarray:
    """Builds one feature row (float64, column order = feature_columns.json)
    from a raw transaction dict -- mirrors fraud_radar.features.build_features
    column-for-column, including the feature-pass-F3 additions (the group
    count/mean(amt_log) aggregates, uid/uid_count, amt_to_card1_mean_ratio,
    and the always-NaN time_since_prev_* pair -- see module docstring)."""
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

    # `uid`/`uid_count` are not raw input fields, so they must be computed
    # (and already present in `values`) before the generic per-column loop
    # below runs -- its "if col in values: continue" guard is what keeps it
    # from treating them as ordinary raw-field lookups (which would fail:
    # transaction.get("uid") is always None).
    uid_key = _uid_key(transaction, values["day_index"])
    values["uid"] = _cat_code_for_key("uid", uid_key)
    values["uid_count"] = float(_count_maps.get("uid", {}).get(uid_key, 0))

    # Group aggregates (feature pass F3), same reasoning -- not raw fields.
    for name, cols, stats in GROUP_STATS_DEFS:
        if "count" in stats:
            values[f"{name}_count"] = _group_stat(name, cols, "count", transaction)
        if "mean_amt" in stats:
            values[f"{name}_mean_amt"] = _group_stat(name, cols, "mean_amt", transaction)
    card1_mean_amt = values.get("card1_mean_amt", float("nan"))
    if math.isnan(card1_mean_amt) or card1_mean_amt == 0.0:
        values["amt_to_card1_mean_ratio"] = float("nan")
    else:
        values["amt_to_card1_mean_ratio"] = values["amt_log"] / card1_mean_amt

    # A lone transaction scored by itself has no prior-transaction history
    # to diff against, in this handler or in fraud_radar.score.score_one()
    # -- see module docstring.
    values["time_since_prev_card1"] = float("nan")
    values["time_since_prev_card1_addr1"] = float("nan")

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
            import boto3

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
