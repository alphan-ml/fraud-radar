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
`P_emaildomain`, `uid` computed independently of `isFraud` -- via frequency
maps (`build_features(df, freq_maps=...)`) fit on a training slice and
reused as-is at evaluation and serving time, not recomputed on whatever
frame is at hand (see "Design decisions" below). Feature pass F3 added
card/uid group aggregates, time-since-previous-transaction deltas, an
amount-relative-to-card-mean ratio, and the `uid` stable-account id itself
(`build_features(df, freq_maps=..., group_stats_maps=...)`) -- see "Feature
pass F3" below for the full spec. `model.py` sorts every labeled row
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
realizes a 1.73% review rate, catching 41.5% of fraud at 82.4% precision
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

Numbers below are current (after the F3 feature pass -- card/uid aggregates,
time deltas, amount-relative-to-mean; see "Feature pass F3" below). The
pre-F3 numbers are kept exactly as they were, unedited, in
`outputs/metrics_before_feature_pass.json`.

| Metric | Before (F2) | After (F3) |
|---|---|---|
| ROC-AUC | 0.8932 | **0.9002** |
| PR-AUC | 0.5256 | **0.5531** |
| Brier score | 0.02152 | **0.02064** |

Review-queue precision/recall at fixed review rates:

| Review rate | Reviewed | TP before | Precision before | Recall before | TP after | Precision after | Recall after |
|---|---|---|---|---|---|---|---|
| 0.5% | 591 | 543 | 0.9188 | 0.1336 | **556** | **0.9408** | **0.1368** |
| 1% | 1,181 | 1,064 | 0.9009 | 0.2618 | **1,088** | **0.9213** | **0.2677** |
| 2% | 2,362 | 1,736 | 0.7350 | 0.4272 | **1,806** | **0.7646** | **0.4444** |
| 5% | 5,905 | 2,361 | 0.3998 | 0.5810 | **2,432** | **0.4119** | **0.5984** |

Fitted review policy (`outputs/checkpoints/review_policy.json`, see "Design
decisions"): before F3, a `fitted_cutoff` of `0.571429` reviewed 1.65% of the
holdout (1,954 transactions) at 80.14% precision / 38.53% recall. After F3,
the refit `fitted_cutoff` is `0.666667` (a higher cutoff realizes the same
2% validation-slice target because F3's scores are better separated),
reviewing 1.73% of the holdout (2,045 transactions) at **82.40%** precision
and **41.46%** recall.

Top-15 feature importance (LightGBM total gain, after F3; full table in
`outputs/metrics.json`'s `top_feature_importance`):

| Rank | Feature | Gain share |
|---|---|---|
| 1 | `card1` | 28.11% |
| 2 | `uid` | 23.33% |
| 3 | `V258` | 7.69% |
| 4 | `V70` | 5.09% |
| 5 | `addr1` | 3.21% |
| 6 | `card2` | 3.07% |
| 7 | `C14` | 2.84% |
| 8 | `V294` | 2.62% |
| 9 | `uid_count` | 1.91% |
| 10 | `DeviceInfo` | 1.87% |
| 11 | `C4` | 1.31% |
| 12 | `V91` | 1.17% |
| 13 | `C1` | 1.04% |
| 14 | `amt_log` | 0.89% |
| 15 | `C13` | 0.88% |

`uid` and `uid_count` alone account for 25.2% of total gain, second only to
`card1` -- the F3 pass's biggest single change (the "uid" pattern -- card1 +
addr1 + D1-derived account start day -- is a well-known strong signal for
this specific dataset) shows up exactly where expected.

## Feature pass F3: card aggregates, time deltas, uid

Added in `src/fraud_radar/features.py` (`fit_group_stats`/`apply_group_stats`,
`_compute_uid`, `_time_since_prev`), all computed from the training window
only, never from the row being scored or a later row -- see the module
docstring for the full spec of each. Summary:

- **Group aggregates** -- count and mean(`amt_log`) per `card1`,
  `card1+addr1`, and `card1+P_emaildomain`. Fit on the train slice only
  (`fit_group_stats`, mirroring the existing `fit_freq_maps` count-encoding
  pattern), saved to `outputs/checkpoints/group_stats.json`, reapplied
  identically to validation, holdout, and every served transaction. An
  unseen composite key maps to `0` count / `NaN` mean.
- **`amt_to_card1_mean_ratio`** -- `amt_log` divided by the card1 group's
  mean `amt_log`.
- **Time since previous transaction** -- `time_since_prev_card1` and
  `time_since_prev_card1_addr1`, seconds since the previous transaction (by
  `TransactionDT`) sharing the same key. Purely causal (sort by
  `TransactionDT`, diff within group), so it is never refit per split -- a
  row's value only ever comes from rows strictly earlier in time, which
  `tests/test_no_leak.py` proves directly (appending a later-in-time row
  does not change any earlier row's value; a row's value is unchanged
  whether or not later rows are even present in the frame). A single-row
  call (serving) always returns `NaN` here -- there is no history in a
  one-row frame -- a known limitation, not a bug, same category as the
  count-encoding limitation described in decision 6 above.
- **`uid`** -- `card1|addr1|account_start_day`, `account_start_day =
  day_index - D1` (the standard "uid" trick for this dataset -- D1 is
  IEEE-CIS's undocumented "days since this card's first transaction" field;
  subtracting it from the current day index recovers an approximately
  constant "day zero" per account). Kept as a LightGBM categorical like
  `card1`; `uid_count` (via the existing count-encoding pipeline, now
  `COUNT_ENCODE_COLS = [card1, addr1, P_emaildomain, uid]`) is its
  training-slice population count.
- **Frequency maps saved to `aws-lambda/`**, the same way the existing count
  maps are: `aws-lambda/count_maps.json` is the updated
  `outputs/checkpoints/freq_maps.json` (now includes `uid` alongside
  `card1`/`addr1`/`P_emaildomain`); `aws-lambda/group_stats.json` is the new
  group-aggregate maps. `aws-lambda/lambda_function.py`'s pure-numpy
  reimplementation and `aws-lambda/feature_columns.json` have since been
  updated to match the new feature set (see "How the handler mirrors
  score.py" below); `cat_code_maps.json` / `isotonic_thresholds.json` /
  `examples.json` still need regenerating from the F3 booster once one is
  trained in an environment with S3/model-checkpoint access -- see
  `aws-lambda/DEPLOY.md`. Deploy is still the owner's call after review.
- Retraining used the identical split, seed (26), `scale_pos_weight`, and
  isotonic-calibration procedure as before -- only the feature set changed.

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
  `DeviceType`, `DeviceInfo`, `M4`) plus `uid` (feature pass F3) come
  directly from the `pandas_categorical` section embedded in the trained
  model file itself -- the exact codes LightGBM's own `_data_from_pandas`
  uses at predict time (`.cat.set_categories(category)` then
  `.cat.codes`, confirmed by reading `lightgbm/basic.py` in the installed
  package) -- not recomputed, so encoding is byte-identical by
  construction. `aws-lambda/cat_code_maps.json`.
- **Count-encoded columns** (`card1_count`, `addr1_count`,
  `P_emaildomain_count`, `uid_count`): looked up from
  `aws-lambda/count_maps.json` (the same `outputs/checkpoints/freq_maps.json`
  fit on the training slice by `fraud_radar.model.train()`), downloaded
  from S3 like the other assets. A value not present in the map -- unseen
  in the training slice, or a missing field -- looks up 0, matching
  `fraud_radar.features.apply_freq_maps()` exactly. (Previously these were
  hard-coded to `1.0` to match a `score.py` bug where the map was
  recomputed on whatever frame was at hand -- a single row at serving
  time. Fixed in both places together; see README "Design decisions".)
- **Group aggregates, `uid`, and time deltas** (feature pass F3): the
  `card1`/`card1_addr1`/`card1_P_emaildomain` count and mean(`amt_log`)
  aggregates are looked up from `aws-lambda/group_stats.json` (downloaded
  from S3, not bundled -- see `aws-lambda/DEPLOY.md`) the same way
  `fraud_radar.features.apply_group_stats()` does, with the same unseen-key
  defaults (`0` count / `NaN` mean); `uid`'s composite key
  (`card1||addr1||account_start_day`) is built the same way
  `_compute_uid()` builds it and then looked up in `cat_code_maps.json`
  like any other categorical; `time_since_prev_card1`/
  `time_since_prev_card1_addr1` are hard-coded `NaN`, matching
  `score.py`'s own single-row `score_one()` (a lone transaction has no
  history to diff against either way). `tests/test_lambda_features.py`
  checks all of this against the real `build_features()` with synthetic
  maps -- no trained model required for that check.
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

**This "0.0 max abs diff" number predates feature pass F3** and needs a
fresh run: the handler and `feature_columns.json` now compute the F3
features, but `cat_code_maps.json`/`isotonic_thresholds.json`/
`examples.json`/`verification_rows.json` above are still the pre-F3 files
(regenerating them needs the real trained F3 booster, which the F3-handler
session had no access to -- no local checkpoint, no AWS credentials, and
retraining was out of scope for that task). Run
`aws-lambda/build_verification.py` then `aws-lambda/verify_lambda_local.py`
on a machine with `outputs/checkpoints/` populated from the F3 training run
before deploying -- see `aws-lambda/DEPLOY.md`.

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

## Live Eval canary

A scheduled job (`.github/workflows/canary.yml`, cron `17 */6 * * *` plus
manual `workflow_dispatch`) scores a fixed set of real transactions against
the LIVE `/score` endpoint above -- not a local copy of the model -- and
compares the observed ROC-AUC to the recorded holdout value in
`outputs/metrics.json` (0.8935, tolerance 0.030). This is what feeds
giggitai.com's public Live Eval tab.

- **What runs**: `src/fraud_radar/canary.py` sends one POST per row in
  `canary/rows.json` to the live endpoint, computes ROC-AUC of the returned
  `calibrated_probability` against each row's true `isFraud` label, and
  prints one JSON record to stdout. No fallback numbers -- a request that
  errors (timeout, bad response, missing field) counts toward `errors` and
  forces `match: false` for that run; the canary reports what the live
  endpoint does right now, not what the model would have said.
- **Canary set**: 500 real transactions from the time-ordered HOLDOUT split
  (`outputs/split_info.json` defines the cutoff), 50 of them fraud (well
  above the required floor of 40), picked deterministically -- evenly
  spaced by `TransactionDT` within each label, not random or hand-picked --
  by `scripts/build_canary_rows.py`. `tests/test_canary_rows.py` proves
  every row's `TransactionDT` falls on the holdout side of the recorded
  split cutoff.
- **When**: every 6 hours (`17 */6 * * *`, UTC), or on demand via
  `workflow_dispatch`.
- **Where the ledger is**: every run appends one JSON record to
  `ledger/runs.jsonl` and overwrites `ledger/latest.json` on this repo's
  `ledger` branch -- an orphan branch with no relation to `main`'s history,
  committed as `Alpha N <45754668+alphan-ml@users.noreply.github.com>`.
  giggitai.com reads both files from `raw.githubusercontent.com`.
- **How to read `match`**: `true` means the live endpoint's observed
  ROC-AUC on the canary set was within `tolerance` (0.030) of the recorded
  value and every row scored without error. `false` means either the
  metric drifted past tolerance or at least one request failed -- check
  `errors` and `observed` in the record to tell which.

Example record (real run against the live endpoint, 2026-09-17):

```json
{"ts": "2026-09-17T23:03:03Z", "system": "fraud-radar", "kind": "canary", "release": "6d97b60", "endpoint": "https://r3skxlusm4.execute-api.us-east-1.amazonaws.com/score", "n": 500, "metric": "roc_auc", "recorded": 0.8935432901709163, "observed": 0.895422, "tolerance": 0.03, "match": true, "p50_ms": 121, "p95_ms": 154, "errors": 0, "duration_s": 61.5}
```

Run it by hand: `python3 -m fraud_radar.canary` (prints the record to
stdout; add `--n 5` to try a handful of rows first).

## What this repo does not do

No file on giggitai.com is written by this repo (the live API below is
called by the site, not the other way around). No push to a remote git
host is performed by anything in this repo unless `git remote origin`
already exists. See `CONTEXT.md` for open items.
