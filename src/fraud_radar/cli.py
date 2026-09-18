"""fraud-radar fetch | check | clean | features | train | eval | export | all

Each step is resumable: it skips when outputs/checkpoints/<step>.done
exists, unless --force is passed. One line per step, with row counts and
wall time, to stdout and outputs/run.log.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_radar import check as check_mod
from fraud_radar import clean as clean_mod
from fraud_radar import eval as eval_mod
from fraud_radar import export as export_mod
from fraud_radar import features as features_mod
from fraud_radar import fetch as fetch_mod
from fraud_radar import model as model_mod

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
OUT_DIR = ROOT / "outputs"
LOG_PATH = OUT_DIR / "run.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _done(step: str) -> bool:
    return (CKPT_DIR / f"{step}.done").exists()


def _mark_done(step: str) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / f"{step}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def step_fetch(force: bool = False) -> None:
    if _done("fetch") and not force:
        _log("fetch: skipped (checkpoint present)")
        return
    t0 = time.time()
    r = fetch_mod.fetch(force=force)
    _log(f"fetch: {len(r['csv_shas'])} CSVs verified, done in {time.time() - t0:.1f}s")
    _mark_done("fetch")


def step_check(force: bool = False) -> None:
    if _done("check") and not force:
        _log("check: skipped (checkpoint present)")
        return
    t0 = time.time()
    r = check_mod.check()
    _log(f"check: {r['train_transaction_rows']:,} train rows, "
         f"fraud_rate={r['train_fraud_rate']:.4%}, done in {time.time() - t0:.1f}s")
    _mark_done("check")


def step_clean(force: bool = False) -> None:
    if _done("clean") and not force:
        _log("clean: skipped (checkpoint present)")
        return
    t0 = time.time()
    r = clean_mod.clean(force=force)
    _log(f"clean: train={r['train_rows']:,} rows, test={r['test_rows']:,} rows (set aside, no labels), "
         f"done in {time.time() - t0:.1f}s")
    _mark_done("clean")


def step_features(force: bool = False) -> None:
    if _done("features") and not force:
        _log("features: skipped (checkpoint present)")
        return
    t0 = time.time()
    train = pd.read_parquet(ROOT / "data" / "clean" / "train.parquet")
    labels = train["isFraud"].astype("int64")
    feat = features_mod.build_features(train)
    feat.to_parquet(OUT_DIR / "features.parquet", index=False)
    labels.to_frame("isFraud").to_parquet(OUT_DIR / "labels.parquet", index=False)
    _log(f"features: {len(feat):,} rows, {feat.shape[1]} columns, done in {time.time() - t0:.1f}s")
    _mark_done("features")


def step_train(force: bool = False) -> None:
    if _done("train") and not force:
        _log("train: skipped (checkpoint present)")
        return
    t0 = time.time()
    feat = pd.read_parquet(OUT_DIR / "features.parquet")
    labels = pd.read_parquet(OUT_DIR / "labels.parquet")["isFraud"]
    info = model_mod.train(feat, labels)
    _log(
        f"train: best_iter={info['best_iteration']} n_train={info['n_train']:,} "
        f"n_val={info['n_val']:,} n_holdout={info['n_holdout']:,} "
        f"cutoff_dt={info['cutoff_dt_train_pool_vs_holdout']} "
        f"scale_pos_weight={info['scale_pos_weight']:.2f}, done in {time.time() - t0:.1f}s"
    )
    _mark_done("train")


def step_eval(force: bool = False) -> None:
    if _done("eval") and not force:
        _log("eval: skipped (checkpoint present)")
        return
    t0 = time.time()
    feat = pd.read_parquet(OUT_DIR / "features.parquet")
    labels = pd.read_parquet(OUT_DIR / "labels.parquet")["isFraud"].to_numpy()

    booster, calibrator, feature_cols, _cat_cols = model_mod.load()
    freq_maps = model_mod.load_freq_maps()
    review_policy = model_mod.load_review_policy()
    split_idx = model_mod.load_split_idx()
    holdout_idx = split_idx["holdout_idx"]

    feat = features_mod.apply_freq_maps(feat, freq_maps)
    X_holdout = feat[feature_cols].iloc[holdout_idx]
    y_holdout = labels[holdout_idx]
    scores = model_mod.predict_calibrated(booster, calibrator, X_holdout)

    metrics = eval_mod.evaluate(y_holdout, scores, booster, feature_cols, review_policy)

    # 10 anonymized example holdout rows: masked TransactionID, amount,
    # product code, calibrated score, true label. Not typed by hand --
    # deterministic sample (every Nth row by calibrated score rank) from
    # the real holdout predictions computed above.
    example_order = np.argsort(-scores)
    n = len(example_order)
    pick_positions = np.linspace(0, n - 1, 10).astype(int)
    picks = example_order[pick_positions]
    examples = []
    for rank, pos in enumerate(picks, start=1):
        row = X_holdout.iloc[pos]
        examples.append({
            "example_id": f"HOLDOUT-{rank:02d}",
            "amount_log": round(float(row["amt_log"]), 3),
            "product_code": str(row.get("ProductCD", "NA")),
            "calibrated_score": round(float(scores[pos]), 6),
            "true_label": int(y_holdout[pos]),
        })
    (OUT_DIR / "holdout_examples.json").write_text(json.dumps(examples, indent=2))

    _log(
        f"eval: roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
        f"brier={metrics['brier_score']:.4f} holdout_n={metrics['holdout_n']:,}, "
        f"done in {time.time() - t0:.1f}s"
    )
    _mark_done("eval")


def step_export(force: bool = False) -> None:
    if _done("export") and not force:
        _log("export: skipped (checkpoint present)")
        return
    t0 = time.time()
    export_mod.export()
    _log(f"export: done in {time.time() - t0:.1f}s")
    _mark_done("export")


STEPS = {
    "fetch": step_fetch,
    "check": step_check,
    "clean": step_clean,
    "features": step_features,
    "train": step_train,
    "eval": step_eval,
    "export": step_export,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="fraud-radar")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in list(STEPS.keys()) + ["all"]:
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "all":
        for step_fn in STEPS.values():
            step_fn(force=args.force)
    else:
        STEPS[args.command](force=args.force)


if __name__ == "__main__":
    main()
