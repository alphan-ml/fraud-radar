# Fraud Radar

A time-aware LightGBM binary classifier, isotonic-calibrated, that scores each
credit-card transaction with a probability of fraud. Built and evaluated on
the full real IEEE-CIS Fraud Detection dataset (Kaggle competition
`ieee-fraud-detection`), 590,540 labeled training transactions.

## Architecture

`clean.py` left-joins `train_identity.csv` onto `train_transaction.csv` on
`TransactionID` (only 24.4% of transactions have a matching identity row --
the join is left, not inner, so the other 75.6% are kept with identity
columns as missing, not dropped). The Kaggle test set has no `isFraud`
column at all, so it is cleaned and set aside but never touched by training
or evaluation. `features.py` turns the merged frame into a model-ready
feature set: V/C/D numeric columns cast to float32, M1-M9 T/F flags mapped
to 1/0, `card1-6`/`addr1-2`/`P_emaildomain`/`R_emaildomain`/`ProductCD`/
`DeviceType`/`DeviceInfo`/`M4` kept as LightGBM native categoricals (no
one-hot), `TransactionAmt` log1p-transformed, hour-of-day and day-index
derived from `TransactionDT`, and count encodings for `card1`, `addr1`,
`P_emaildomain` computed independently of `isFraud`. `model.py` sorts every
labeled row by `TransactionDT` ascending, takes the last 20% as a holdout
untouched by training, early stopping, or calibration, and inside the
remaining 80% pool takes the last 10% (also time-ordered) as an
early-stopping / calibration validation slice. LightGBM trains on the first
90% of the pool with `scale_pos_weight` set to the train slice's
negative/positive ratio; isotonic regression is then fit on the validation
slice's raw scores vs. true labels to produce calibrated probabilities.
`eval.py` scores the untouched holdout only. `score.py` is the serving path:
it loads the saved booster and calibrator once and turns one raw transaction
dict into a calibrated probability and a review flag.

## How to run

```bash
git clone <this repo> && cd fraud-radar
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
# data/raw/ieee-fraud-detection.zip must already be present -- the Kaggle
# competition requires an authenticated, rules-accepted account, so this
# repo does not call the Kaggle API itself.
python3 -m fraud_radar.cli all      # fetch -> check -> clean -> features -> train -> eval -> export
python3 -m pytest -q
python3 -m ruff check .
```

Each step is resumable: rerunning `fraud-radar all` (or `python3 -m
fraud_radar.cli all`) skips any step whose `outputs/checkpoints/<step>.done`
marker exists; pass `--force` to redo a step (or all of them).

## Example: scoring one transaction

```python
from fraud_radar.score import FraudScorer

scorer = FraudScorer.load()
result = scorer.score_one({
    "TransactionID": 1, "TransactionDT": 86400, "TransactionAmt": 50.0,
    "ProductCD": "W", "card1": 13926, "card4": "visa",
})
# {"calibrated_probability": 0.0123, "review_flag": False}
```

`review_flag` defaults to the 2%-review-queue operating point (score >=
0.02), which on the real holdout catches 40.9% of fraud at 70.4% precision
(see Results below) -- pass `review_threshold=` to `FraudScorer.load()` to
use a different operating point.

## Design decisions

1. **Left join, not inner join, for identity.** Only 144,233 of 590,540
   (24.4%) train transactions have a matching identity row. An inner join
   would silently drop the other 75.6% of the data the model is meant to
   score; missingness in the identity columns is itself informative (see
   `outputs/clean_stats.json`), so it is kept as `NaN`/missing category
   rather than imputed or dropped.
2. **Time-ordered split, not random.** Fraud detection is deployed against
   future transactions, never past ones. A random split leaks future
   information (fraud patterns, card reuse, merchant drift) into training
   and inflates the reported metrics relative to real deployment. The split
   is strictly by `TransactionDT`: first 80% train pool, last 20% holdout,
   with the last 10% of the 80% pool held out again for early stopping and
   calibration -- `tests/test_split.py` proves the ordering and sizes.
3. **LightGBM native categoricals over one-hot encoding.** `card1` alone has
   thousands of distinct values; one-hot would blow up the feature space
   and lose LightGBM's built-in categorical split handling. `card1-6`,
   `addr1-2`, `P_emaildomain`, `R_emaildomain`, `ProductCD`, `DeviceType`,
   `DeviceInfo`, and `M4` are passed as pandas `category` dtype directly.
4. **Isotonic calibration on a held-out validation slice, not on the
   training data or the final holdout.** Fitting the calibrator on training
   data would overfit; fitting it on the holdout would leak the holdout
   into the deployed model. The 10%-of-pool validation slice used for
   early stopping doubles as the calibration set, and is never touched
   again after `train()` returns -- only `eval.py` reads the holdout, and
   only after training is fully done.
5. **`scale_pos_weight`, not undersampling or SMOTE.** The training slice's
   real fraud rate (3.51%) is preserved; LightGBM's native
   `scale_pos_weight` (27.45 for this run, `n_neg/n_pos` on the train
   slice) reweights the loss instead of discarding or synthesizing rows,
   which keeps every real negative example in the model's view.

## Data provenance

Dataset: Kaggle competition `ieee-fraud-detection` (IEEE Computational
Intelligence Society, IEEE-CIS Fraud Detection). Downloaded zip sha256
`4cc646da09d0a9b265983ffed775b1f9ee15af5266586df610e04d6adae0b829`
(123,856,947 bytes). 590,540 labeled train transactions (`TransactionDT`
86,400 -- 15,811,131, seconds elapsed from an undocumented reference point
per Kaggle's own note, not a calendar date), 20,663 fraud (3.499%
fraud rate). 144,233 train transactions (24.4%) have a matching identity
row. 506,691 test transactions are cleaned and set aside with no labels --
never used for training or evaluation. Full column-group missing-rate
breakdown: `outputs/clean_stats.json`. Full row/column counts and file
hashes for all five extracted CSVs: `outputs/data_quality.json`.

## Results (time-ordered 20% holdout, seed 26, 118,108 transactions, 4,064 fraud)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8935 |
| PR-AUC | 0.5070 |
| Brier score | 0.02225 |

Review-queue precision/recall at fixed review rates:

| Review rate | Transactions reviewed | True positives caught | Precision | Recall |
|---|---|---|---|---|
| 0.5% | 591 | 543 | 0.9188 | 0.1336 |
| 1% | 1,181 | 1,047 | 0.8865 | 0.2576 |
| 2% | 2,362 | 1,663 | 0.7041 | 0.4092 |
| 5% | 5,905 | 2,306 | 0.3905 | 0.5674 |

Top-5 features by LightGBM total gain: `card1` (34.77% of gain), `V258`
(9.41%), `card2` (7.38%), `addr1` (5.64%), `V70` (5.42%). Full top-15 table,
the 10-bin calibration table, and the calibrated review-queue table above
are all in `outputs/metrics.json` and `outputs/site_data.json` -- written by
`fraud-radar eval` / `fraud-radar export`, never typed by hand.

## Live scoring API (AWS Lambda + API Gateway)

Deployed 2026-09-16. Base URL:

```
https://r3skxlusm4.execute-api.us-east-1.amazonaws.com
```

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | `{"status": "ok"}` |
| `/examples` | GET | 10 real holdout transactions (full input feature dict + true label + calibrated score each), for prefilling a web form |
| `/score` | POST | Score one transaction dict, returns `{"calibrated_probability": float, "review_flag": bool}` |

CORS is restricted to `https://www.giggitai.com` and `https://giggitai.com`
(both at the API Gateway CORS config and echoed by the Lambda for direct
invokes).

### Infrastructure

- **S3** `giggit-fraud-radar-models` (versioned) -- `models/lgb_model.txt`,
  `models/feature_columns.json`, `models/cat_code_maps.json`,
  `models/isotonic_thresholds.json`, `models/examples.json`;
  `lambda-code/deploy.zip`.
- **IAM role** `fraud-lambda-execution-role` -- `AWSLambdaBasicExecutionRole`
  plus an inline policy granting `s3:GetObject` on
  `giggit-fraud-radar-models/models/*` only.
- **Lambda** `fraud-score-transaction` -- Python 3.11, **arm64**, 1024 MB,
  30 s timeout. Downloads the booster, category-code maps, and isotonic
  thresholds from S3 into `/tmp` on cold start, then serves from memory.
  Bundled dependencies: numpy, a hand-trimmed scipy (scipy.sparse's own
  import chain only -- stats/optimize/spatial/io/signal/interpolate/
  integrate/ndimage/cluster/fftpack/odr/differentiate/datasets/misc
  removed, ~35 MB saved), lightgbm 4.7.0, narwhals (a lightgbm 4.x
  dependency), and a vendored `lib/libgomp.so.1` (arm64 OpenMP runtime,
  not present on the Lambda base image; picked up automatically because
  `$LAMBDA_TASK_ROOT/lib` is on the default `LD_LIBRARY_PATH`). No pandas,
  no scikit-learn in the Lambda -- see "How the handler mirrors score.py"
  below. Deployment zip: 37 MB compressed / 122 MB unzipped, both well
  under the 250 MB unzipped Lambda limit.
- **API Gateway** `fraud-score-api` (HTTP API) -- routes above, Lambda
  proxy integration, auto-deployed to `$default`.

### How the handler mirrors score.py

The Lambda handler (`aws-lambda/lambda_function.py`) does not bundle
pandas or scikit-learn. Instead it reproduces
`fraud_radar.features.build_features()` and `FraudScorer.score_one()` in
pure numpy:

- **Category codes** for the 14 LightGBM categorical columns (`card1-6`,
  `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `ProductCD`,
  `DeviceType`, `DeviceInfo`, `M4`) come directly from the
  `pandas_categorical` section embedded in the trained model file itself
  -- the exact codes LightGBM's own `_data_from_pandas` uses at predict
  time (`.cat.set_categories(category)` then `.cat.codes`, confirmed by
  reading `lightgbm/basic.py` in the installed package) -- not
  recomputed, so encoding is byte-identical by construction.
  `aws-lambda/cat_code_maps.json`.
- **Count-encoded columns** (`card1_count`, `addr1_count`,
  `P_emaildomain_count`): `build_features()` calls
  `df[c].value_counts(dropna=False)` on whatever DataFrame it is given.
  At serving time that DataFrame is always exactly one row, so every
  count-encoded column comes out as **1.0** for every single-transaction
  score in the real, unmodified `score.py` -- not the full-population
  frequency. Reproduced exactly (hard-coded `1.0`), not "fixed", to match
  `score.py`'s actual behavior. Worth knowing if this model is ever
  batch-scored differently.
- **Isotonic calibration**: `outputs/checkpoints/calibrator.joblib`'s
  106 sorted `(X_thresholds_, y_thresholds_)` pairs were extracted once
  (`aws-lambda/isotonic_thresholds.json`) and are applied with
  `np.interp` after clipping to `[X_min_, X_max_]` -- verified 0.0 max
  abs diff against `sklearn.isotonic.IsotonicRegression.predict` on 50
  random points, so no scikit-learn/scipy.interpolate dependency is
  needed for calibration.
- **Float32 rounding**: `build_features()` casts every numeric feature to
  float32 before LightGBM sees it (`_data_from_pandas`'s dtype promotion
  lands on float32 for this feature set). The handler round-trips its
  float64 feature row through float32 before calling `Booster.predict`
  so split-threshold comparisons see the same values.

### Verification (real, not synthetic)

20 real transactions from the time-ordered 20% holdout
(`outputs/checkpoints/split_idx.npz` -> `holdout_idx[:20]`,
`data/clean/train.parquet`) were scored two ways: locally through the
unmodified `FraudScorer.score_one()`, and live through the deployed
Lambda (`aws lambda invoke`, then again through the public API Gateway
URL). **Max abs diff on `calibrated_probability`: 0.0. All 20
`review_flag` values matched.** Full rows (input + both scores):
`aws-lambda/verification_rows.json`. Reproduce with
`aws-lambda/build_verification.py` (builds the 20 rows + 10 examples)
and `aws-lambda/test_invoke.py` (scores all 20 through the live Lambda
and diffs).

### Live curl proof

```
$ curl -s https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/health
{"status": "ok"}

$ curl -s https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/examples | head -c 200
{"examples": [{"example_id": "HOLDOUT-01", "transaction_id": 3459432, "input": {"TransactionID": 3459432, ...

$ curl -s -X POST https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/score \
    -H 'Content-Type: application/json' \
    -d '{"TransactionID": 3459432, "TransactionDT": 12192900, "TransactionAmt": 33.261, "ProductCD": "C", "card1": 9300, "card2": 103.0, "card3": 185.0, "card4": "visa", "card5": 138.0, "card6": "debit", ...}'
{"calibrated_probability": 0.01709, "review_flag": false}
# matches local score_one() on the same real holdout transaction exactly

$ curl -s -X POST https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/score -d '{}'
{"error": "missing fields: ['TransactionID', 'TransactionDT', 'TransactionAmt']"}

$ curl -s -D - -o /dev/null -X OPTIONS https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/score \
    -H 'Origin: https://www.giggitai.com' -H 'Access-Control-Request-Method: POST' | grep -i access-control
access-control-allow-origin: https://www.giggitai.com
access-control-allow-methods: GET,OPTIONS,POST
access-control-allow-headers: content-type
```

## What this repo does not do

No file on giggitai.com is written by this repo (the live API below is
called by the site, not the other way around). No push to a remote git
host is performed by anything in this repo unless `git remote origin`
already exists. See `CONTEXT.md` for open items.
