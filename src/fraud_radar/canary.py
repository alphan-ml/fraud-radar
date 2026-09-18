"""Live Eval canary for Fraud Radar.

Calls the deployed `/score` endpoint once per row in `canary/rows.json`
(500 real transactions from the time-ordered HOLDOUT split -- see
`scripts/build_canary_rows.py`), computes ROC-AUC of the returned
`calibrated_probability` against the true `isFraud` label, and compares it
to the recorded holdout ROC-AUC in `outputs/metrics.json` within a fixed
tolerance. Prints one JSON record (the public ledger's record schema) to
stdout.

No fallback numbers: this module never substitutes the recorded metric, or
any other placeholder, for a value the live endpoint failed to produce. A
request that errors (timeout, non-200, malformed body, missing field) is
counted in `errors` and forces `match=false` for the whole run -- the
canary reports what the live endpoint does right now, including its
failures, not what the model would have said.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/score"
DEFAULT_ROWS = ROOT / "canary" / "rows.json"
DEFAULT_METRICS = ROOT / "outputs" / "metrics.json"
SYSTEM = "fraud-radar"
METRIC_NAME = "roc_auc"
TOLERANCE = 0.030
REQUEST_TIMEOUT_S = 15.0


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _score_one(endpoint: str, transaction: dict, timeout: float) -> tuple[float | None, float]:
    """POSTs one transaction dict to the live endpoint. Returns
    (calibrated_probability, elapsed_ms); calibrated_probability is None on
    any error (network, HTTP, or a malformed/missing field in the response)."""
    body = json.dumps(transaction).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.monotonic() - t0) * 1000
        proba = payload.get("calibrated_probability")
        if not isinstance(proba, (int, float)):
            return None, elapsed_ms
        return float(proba), elapsed_ms
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
        elapsed_ms = (time.monotonic() - t0) * 1000
        return None, elapsed_ms


def _percentile(sorted_values: list[float], p: float) -> int:
    if not sorted_values:
        return 0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return round(sorted_values[idx])


def run_canary(endpoint: str, rows_path: Path, metrics_path: Path, n: int | None, timeout: float) -> dict:
    rows = json.loads(rows_path.read_text())
    if n is not None:
        rows = rows[:n]
    recorded = json.loads(metrics_path.read_text())[METRIC_NAME]

    t_start = time.monotonic()
    y_true: list[int] = []
    y_score: list[float] = []
    latencies_ms: list[float] = []
    errors = 0

    for row in rows:
        proba, elapsed_ms = _score_one(endpoint, row["input"], timeout)
        latencies_ms.append(elapsed_ms)
        if proba is None:
            errors += 1
            continue
        y_true.append(row["isFraud"])
        y_score.append(proba)

    duration_s = time.monotonic() - t_start

    observed = None
    if len(y_true) >= 2 and len(set(y_true)) > 1:
        observed = round(float(roc_auc_score(y_true, y_score)), 6)

    match = errors == 0 and observed is not None and abs(observed - recorded) <= TOLERANCE

    latencies_sorted = sorted(latencies_ms)
    p50_ms = _percentile(latencies_sorted, 0.50)
    p95_ms = _percentile(latencies_sorted, 0.95)

    release = _git_sha()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    observed_str = f"{observed:.4f}" if observed is not None else "n/a"
    match_str = "MATCH" if match else "MISMATCH"

    lines = [
        f"$ canary {SYSTEM} --n {len(rows)} --endpoint /score",
        f"release {release} · {len(rows)} holdout rows · {errors} errors",
        (
            f"{METRIC_NAME} {observed_str} · recorded {recorded:.4f} · "
            f"tolerance {TOLERANCE:.3f} · {match_str}"
        ),
        f"p50 {p50_ms} ms · p95 {p95_ms} ms · {duration_s:.1f}s total",
    ]

    return {
        "ts": ts,
        "system": SYSTEM,
        "kind": "canary",
        "release": release,
        "endpoint": endpoint,
        "n": len(rows),
        "metric": METRIC_NAME,
        "recorded": recorded,
        "observed": observed,
        "tolerance": TOLERANCE,
        "match": match,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "errors": errors,
        "duration_s": round(duration_s, 1),
        "lines": lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="canary")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--n", type=int, default=None, help="limit rows scored (debug only)")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S)
    args = parser.parse_args()

    record = run_canary(args.endpoint, args.rows, args.metrics, args.n, args.timeout)
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
