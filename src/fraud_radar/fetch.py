"""Unzip the IEEE-CIS Fraud Detection dataset (Kaggle competition
`ieee-fraud-detection`).

The zip (data/raw/ieee-fraud-detection.zip) is downloaded separately -- the
Kaggle competition requires an authenticated, rules-accepted account, so this
module does not call the Kaggle API. It only extracts the five CSVs from the
zip already on disk, verifies their sha256, and records row counts.
Resumable: skips extraction when all five CSVs are already present, unless
`force=True`.
"""
from __future__ import annotations

import hashlib
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ieee-fraud-detection.zip"

EXPECTED_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool = False) -> dict:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"{ZIP_PATH} not found -- download the Kaggle competition zip "
            "'ieee-fraud-detection' with an authenticated, rules-accepted "
            "account first (not automated here)."
        )

    already_present = all((RAW_DIR / f).exists() for f in EXPECTED_FILES)
    if already_present and not force:
        print(f"[fetch] all {len(EXPECTED_FILES)} CSVs already present in {RAW_DIR}, skipping unzip")
    else:
        t0 = time.time()
        print(f"[fetch] unzipping {ZIP_PATH} -> {RAW_DIR}")
        with zipfile.ZipFile(ZIP_PATH) as z:
            names_in_zip = set(z.namelist())
            missing = [f for f in EXPECTED_FILES if f not in names_in_zip]
            if missing:
                raise RuntimeError(f"zip is missing expected files: {missing}; has: {sorted(names_in_zip)}")
            z.extractall(RAW_DIR)
        print(f"[fetch] unzip done in {time.time() - t0:.1f}s")

    zip_sha = _sha256(ZIP_PATH)
    csv_shas = {}
    for f in EXPECTED_FILES:
        p = RAW_DIR / f
        if not p.exists():
            raise RuntimeError(f"expected {p} after unzip, not found")
        csv_shas[f] = {"sha256": _sha256(p), "bytes": p.stat().st_size}

    return {
        "zip_path": str(ZIP_PATH),
        "zip_sha256": zip_sha,
        "zip_bytes": ZIP_PATH.stat().st_size,
        "csv_shas": csv_shas,
    }


if __name__ == "__main__":
    r = fetch(force="--force" in sys.argv)
    print(f"[fetch] zip sha256: {r['zip_sha256']}")
    for name, info in r["csv_shas"].items():
        print(f"[fetch]   {name}: {info['bytes']:,} bytes, sha256={info['sha256']}")
