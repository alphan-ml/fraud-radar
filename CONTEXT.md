# Context: Fraud Radar build

Local-only build. No push, no site change, no deployment -- everything in
this repo runs against the local Kaggle download and writes local files.
Identity on every commit: `Alpha N <45754668+alphan-ml@users.noreply.github.com>`.

## What was built, in order

1. **fetch** -- unzip `data/raw/ieee-fraud-detection.zip` (sha256
   `4cc646d...ae0b829`, 123,856,947 bytes), verify sha256 of all five
   extracted CSVs, delete nothing that isn't already gitignored.
2. **check** -- row counts (train_transaction 590,540, train_identity
   144,233, test_transaction 506,691, test_identity 141,907 -- all match
   the expected counts, `row_count_mismatches_vs_expected: {}`), column
   counts, raw fraud rate (3.499%, 20,663 / 590,540), `TransactionDT`
   range (86,400 .. 15,811,131). Written to `outputs/data_quality.json`.
3. **clean** -- left join identity onto transaction on `TransactionID`
   (train: 590,540 rows x 434 columns after merge; test: 506,691 rows x
   433 columns, no `isFraud` column at all so it is structurally
   impossible to train on it). Missing rates recorded by column group
   (`identity` group averages 84.8% missing, `dist` 76.6%, `M` 49.9%, `V`
   43.0%, `email` 46.4%, `D` 58.2%, `addr` 11.1%, `card` 0.51%, `C` 0%) --
   full table in `outputs/clean_stats.json`.
4. **features** -- V/C/D numerics to float32; M1/M2/M3/M5/M6/M7/M8/M9
   T/F->1/0 float32, M4 kept categorical (it takes M0/M1/M2, not T/F);
   card1-6/addr1-2/P_emaildomain/R_emaildomain/ProductCD/DeviceType/
   DeviceInfo as LightGBM native categoricals; `amt_log = log1p(clip(
   TransactionAmt, 0))`; `hour_of_day`/`day_index` from `TransactionDT`;
   count encodings for card1/addr1/P_emaildomain computed independently of
   `isFraud`. `build_features()` drops `isFraud` from its input at the top
   of the function, before any feature is computed, so it cannot leak --
   `tests/test_no_leak.py` proves this by shuffling `isFraud` on a copy of
   the input and checking every output feature is bit-identical.
5. **train** -- sort by `TransactionDT`; last 20% (118,108 rows) held out
   untouched; of the remaining 80% pool (472,432 rows), last 10% (47,243
   rows) is the early-stopping/calibration validation slice, first 90%
   (425,189 rows) is what the booster trains on. `scale_pos_weight` =
   27.450250920040148 (train slice n_neg/n_pos). LightGBM binary, seed 26,
   early stopping on AUC over the validation slice, best_iteration = 169.
   Isotonic regression fit on the validation slice's raw scores vs. true
   labels. Model -> `outputs/checkpoints/lgb_model.txt`, calibrator ->
   `outputs/checkpoints/calibrator.joblib`.
6. **eval** -- holdout only (118,108 rows, 4,064 fraud, 3.4409% fraud
   rate). ROC-AUC 0.8935432901709163, PR-AUC 0.5070429412996057, Brier
   0.022254943076573455. Review-queue precision/recall at 0.5/1/2/5%
   (table in README). 10-bin calibration table (mean predicted vs.
   observed fraud rate per bin, monotone and reasonably tight -- bin 10's
   mean predicted 0.256 vs. observed 0.237). Top-15 feature importance by
   LightGBM total gain. 10 anonymized example holdout rows picked
   deterministically by score-rank (`np.linspace` over the score-sorted
   order, not hand-picked) -> `outputs/holdout_examples.json`.
7. **export** -- `outputs/site_data.json` assembles dataset facts, split,
   metrics, review_queue, calibration_table, top_feature_importance, and
   example_rows, all read back from the files the earlier steps wrote.
8. **score.py** -- serving module, added in this session. `FraudScorer.
   load()` loads the saved booster/calibrator/feature-column/categorical-
   column artifacts once; `score_one(transaction_dict)` reuses
   `features.build_features` on a single-row frame and returns
   `{"calibrated_probability": float, "review_flag": bool}`.
   `review_flag` defaults to the 2%-review-rate operating point (score >=
   0.02). `tests/test_score.py` (3 tests) checks the output shape, that
   `score_one` matches the batch `predict_calibrated` path exactly on a
   real training row, and that the threshold is configurable. All three
   are skipped automatically if no trained model is present.

## Decisions and why (see README for the full list)

- Left join for identity, not inner: only 24.4% of transactions have an
  identity row; an inner join would silently drop the other 75.6%.
- Time-ordered 80/20 split (not random, not stratified-random): a fraud
  model is deployed against future transactions; random splits leak
  future card/merchant patterns into training and inflate metrics.
  `tests/test_split.py` (3 tests) proves the ordering and exact sizes.
- LightGBM native categoricals, no one-hot: `card1` alone has thousands of
  distinct values.
- Calibration fit on the validation slice, never on the holdout: fitting
  on holdout would leak it into the deployed model.
- `scale_pos_weight` over undersampling/SMOTE: keeps every real negative
  example in view rather than discarding or synthesizing rows.

## What is NOT done

- Nothing is deployed and no site file is touched -- out of scope for
  this build by the standing rule (see `.env.example`).
- The Kaggle test set (`test_transaction.csv`/`test_identity.csv`,
  506,691 rows) is cleaned and available at `data/clean/test.parquet` but
  is never scored or submitted -- it has no `isFraud` labels, so there is
  nothing to evaluate against locally, and Kaggle submission is out of
  scope for a local build.
- No hyperparameter search was run; `LGB_PARAMS` in `model.py` is a single
  reasonable configuration (`num_leaves=63`, `learning_rate=0.05`,
  `min_data_in_leaf=50`, `feature_fraction`/`bagging_fraction=0.8`), not a
  tuned optimum. Early stopping (100 rounds, best at 169) is the only
  model-selection step performed.
- `score.py` scores one transaction (or a small in-memory batch via
  repeated calls) at a time; there is no batch-file or streaming
  interface, and no serving process (API, queue consumer) is stood up --
  that would be a deployment decision for the parent session.
- SHAP / per-prediction feature attribution was not built; only aggregate
  LightGBM gain-based importance (`top_feature_importance` in
  `outputs/metrics.json`) is available.

## Next command

```bash
cd ~/Claude/fraud-radar && source .venv/bin/activate
python3 -m fraud_radar.cli all      # re-runs any step whose checkpoint is missing (all present now: no-op)
python3 -m pytest -q                # 18 tests
python3 -m ruff check .             # clean
```

To retrain from scratch: `python3 -m fraud_radar.cli all --force`.
