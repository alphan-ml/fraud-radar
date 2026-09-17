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
`P_emaildomain` computed independently of `isFraud` -- via frequency maps
(`build_features(df, freq_maps=...)`) fit on a training slice and reused
as-is at evaluation and serving time, not recomputed on whatever frame is
at hand (see "Design decisions" below). `model.py` sorts every labeled row
by `TransactionDT` ascending, takes the last 20% as a holdout untouched by
training, early stopping, or calibration, and inside the remaining 80%
pool takes the last 10% (also time-ordered) as an early-stopping /
calibration validation slice. LightGBM trains on the first 90% of the pool
with `scale_pos_weight` set to the train slice's negative/positive ratio;
isotonic regression is then fit on the validation slice's raw scores vs.
true labels to produce calibrated probabilities, and a review-queue score
cutoff is fit separately on that same validation slice (see "Design
decisions"). `eval.py` scores the untouched holdout only. `score.py` is the
serving path: it loads the saved booster, calibrator, frequency maps, and
review-policy cutoff once, and turns one raw transaction dict into a
calibrated probability and a review flag.

## How to run

```bash
git clone <this repo> && cd fraud-radar
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

# fetch: download the competition zip with an authenticated, rules-accepted
# Kaggle account (KAGGLE_USERNAME / KAGGLE_KEY env vars or secrets) --
# `fetch.py` itself only unzips and verifies data/raw/ieee-fraud-detection.zip,
# it does not call the Kaggle API.
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/raw

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

`review_flag` defaults to the *fitted* review-queue cutoff -- the calibrated
score at the 98th percentile of the validation slice
(`outputs/checkpoints/review_policy.json`, `fitted_cutoff`), a probability
threshold picked to realize a 2% review rate, not a fixed probability
constant (see "Design decisions" below). On the real holdout this cutoff
realizes a 1.65% review rate, catching 38.5% of fraud at 80.1% precision
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
6. **Count-encoding frequency maps fit on the train slice only, saved, and
   reused everywhere.** `build_features()`'s `card1_count`/`addr1_count`/
   `P_emaildomain_count` used to be `value_counts()` on whatever frame the
   function was given -- the full 590,540-row population before the time
   split during training, but a single row at serving time, so every
   served count came out `1.0` regardless of the actual value. The maps
   are now fit once, on the training slice only (`fraud_radar.features.
   fit_freq_maps`), saved to `outputs/checkpoints/freq_maps.json`, and
   applied identically (`apply_freq_maps`) to the validation slice, the
   holdout, and every served transaction -- a value never seen in the
   training slice maps to `0`. `tests/test_model.py` proves the maps are
   fit on the train slice only; `tests/test_score.py` proves a row scores
   identically alone or inside a batch.
7. **The review-queue cutoff is fit, not a fixed probability.** The
   previous default, `DEFAULT_REVIEW_THRESHOLD = 0.02`, flagged
   `calibrated_probability >= 0.02` and was documented as "the
   2%-review-queue operating point" -- but a fixed probability and a fixed
   review-queue *rate* are different quantities that happen to have
   coincided for one particular model. The cutoff is now fit on the
   validation slice as the calibrated score at the 98th percentile
   (`REVIEW_TOP_FRACTION = 0.02` in `model.py`), saved to
   `outputs/checkpoints/review_policy.json`
   (`top_fraction`/`fitted_cutoff`/`fitted_on`/`n_validation`), and used as
   `score.py`'s default -- it will differ from `0.02` and from run to run
   as the model changes. The realized review rate/precision/recall at this
   cutoff, measured on the untouched holdout, is in
   `outputs/metrics.json`'s `review_policy` block (see Results below).

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
| ROC-AUC | 0.8932 |
| PR-AUC | 0.5256 |
| Brier score | 0.02152 |

Review-queue precision/recall at fixed review rates:

| Review rate | Transactions reviewed | True positives caught | Precision | Recall |
|---|---|---|---|---|
| 0.5% | 591 | 544 | 0.9205 | 0.1339 |
| 1% | 1,181 | 1,064 | 0.9009 | 0.2618 |
| 2% | 2,362 | 1,725 | 0.7303 | 0.4245 |
| 5% | 5,905 | 2,365 | 0.4005 | 0.5819 |

Fitted review policy (`outputs/checkpoints/review_policy.json`, see "Design
decisions"): a `fitted_cutoff` of `0.571429`, fit as the calibrated score at
the 98th percentile of the 47,243-row validation slice. Realized on the
holdout, this cutoff reviews 1.65% of transactions (1,954), at 80.14%
precision and 38.53% recall -- close to, but not identical to, the fixed-2%
row above, because the fitted cutoff is a probability threshold, not a rank
threshold, and the holdout's score distribution is not perfectly identical
to validation's.

Top-5 features by LightGBM total gain: `card1` (35.02% of gain), `V258`
(8.12%), `card2` (7.11%), `addr1` (6.26%), `V70` (4.63%). Full top-15 table,
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
  `models/isotonic_thresholds.json`, `models/count_maps.json`,
  `models/review_policy.json`, `models/examples.json`; `lambda-code/deploy.zip`.
- **IAM role** `fraud-lambda-execution-role` -- `AWSLambdaBasicExecutionRole`
  plus an inline policy granting `s3:GetObject` on
  `giggit-fraud-radar-models/models/*` only.
- **Lambda** `fraud-score-transaction` -- Python 3.11, **arm64**, 1024 MB,
  30 s timeout. Downloads the booster, category-code maps, count-encoding
  frequency maps, the review-policy cutoff, and isotonic thresholds from S3
  into `/tmp` on cold start, then serves from memory.
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
  `P_emaildomain_count`): looked up from `aws-lambda/count_maps.json` (the
  same `outputs/checkpoints/freq_maps.json` fit on the training slice by
  `fraud_radar.model.train()`), downloaded from S3 like the other assets.
  A value not present in the map -- unseen in the training slice, or a
  missing field -- looks up 0, matching
  `fraud_radar.features.apply_freq_maps()` exactly. (Previously these were
  hard-coded to `1.0` to match a `score.py` bug where the map was
  recomputed on whatever frame was at hand -- a single row at serving
  time. Fixed in both places together; see README "Design decisions".)
- **Isotonic calibration**: `outputs/checkpoints/calibrator.joblib`'s
  106 sorted `(X_thresholds_, y_thresholds_)` pairs were extracted once
  (`aws-lambda/isotonic_thresholds.json`) and are applied with
  `np.interp` after clipping to `[X_min_, X_max_]` -- verified 0.0 max
  abs diff against `sklearn.isotonic.IsotonicRegression.predict` on 50
  random points, so no scikit-learn/scipy.interpolate dependency is
  needed for calibration.
- **Review-queue cutoff**: read from `aws-lambda/review_policy.json`'s
  `fitted_cutoff` (the same file `fraud_radar.model.train()` writes to
  `outputs/checkpoints/review_policy.json`) and used as the handler's
  default, instead of a hard-coded `0.02` probability.
- **Float32 rounding**: `build_features()` casts every numeric feature to
  float32 before LightGBM sees it (`_data_from_pandas`'s dtype promotion
  lands on float32 for this feature set). The handler round-trips its
  float64 feature row through float32 before calling `Booster.predict`
  so split-threshold comparisons see the same values.

### Verification (real, not synthetic)

20 real transactions from the time-ordered 20% holdout
(`outputs/checkpoints/split_idx.npz` -> `holdout_idx[:20]`,
`data/clean/train.parquet`) were scored two ways: locally through the
unmodified `FraudScorer.score_one()`, and through the pure-numpy Lambda
handler with all assets staged locally
(`aws-lambda/verify_lambda_local.py`, no S3/AWS calls). **Max abs diff on
`calibrated_probability`: 0.0. All 20 `review_flag` values matched.** Full
rows (input + local score): `aws-lambda/verification_rows.json`. Reproduce
with `aws-lambda/build_verification.py` (builds the 20 rows + 10 examples,
and regenerates `cat_code_maps.json`/`isotonic_thresholds.json`/
`count_maps.json`/`review_policy.json`/`feature_columns.json` from the
current `outputs/checkpoints/` model) then `aws-lambda/verify_lambda_local.py`.

The 2026-09-16 deploy below verified the same way but against the *live*
Lambda (`aws lambda invoke`, then the public API Gateway URL,
`aws-lambda/test_invoke.py`) -- that verification predates this session's
retrain (new `freq_maps.json`/`review_policy.json`, corrected count
encoding), and the live endpoint still serves the previously deployed
model/artifacts until the owner runs the deploy step with the new
`outputs/checkpoints/` and uploads the new S3 assets (see "Do not" in the
originating issue -- deploying is explicitly the owner's step, not
automated here).

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
