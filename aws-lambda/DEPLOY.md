# Deploying the F3 Model

This is the owner's checklist for shipping the feature-pass-F3 model to the
live Lambda. Nothing here is run automatically by this repo's automation --
see "Do not" on the originating issue.

## Before You Start

`aws-lambda/lambda_function.py` now computes the F3 features (the group
aggregates from `group_stats.json`, `uid`/`uid_count`,
`amt_to_card1_mean_ratio`), and `aws-lambda/feature_columns.json` has been
regenerated to match (verified against `fraud_radar.features.build_features`
column-for-column in `tests/test_lambda_features.py`, no trained model
needed for that check).

`aws-lambda/cat_code_maps.json`, `aws-lambda/isotonic_thresholds.json`,
`aws-lambda/examples.json`, and `aws-lambda/verification_rows.json` are
still the pre-F3 files -- they can only be regenerated from the real
trained F3 booster (`outputs/checkpoints/lgb_model.txt` /
`calibrator.joblib`), which this session had no access to: no local
checkpoint, no AWS credentials, and retraining was out of scope. Run step 1
below first.

1. On a machine with the F3 `outputs/checkpoints/` present:
   ```bash
   python3 aws-lambda/build_verification.py
   ```
   This regenerates `aws-lambda/cat_code_maps.json`,
   `aws-lambda/isotonic_thresholds.json`, `aws-lambda/feature_columns.json`,
   `aws-lambda/count_maps.json`, `aws-lambda/review_policy.json`,
   `aws-lambda/verification_rows.json`, and `aws-lambda/examples.json` from
   the current model. It also copies `outputs/checkpoints/group_stats.json`
   to `aws-lambda/group_stats.json`.

2. Run the local verification script and confirm it passes:
   ```bash
   python3 aws-lambda/verify_lambda_local.py
   ```
   It scores the ten `aws-lambda/examples.json` rows through the handler's
   own code path (no S3, no network) and asserts each
   `calibrated_probability` matches `examples.json` to six decimals. Stop
   and fix the handler before deploying if this fails -- do not deploy a
   handler that disagrees with what the site will show.

## Files and Where They Go

### S3 `models/` (downloaded by the Lambda at cold start, not in the zip)

| File | Local source |
|---|---|
| `lgb_model.txt` | `outputs/checkpoints/lgb_model.txt` |
| `isotonic_thresholds.json` | `aws-lambda/isotonic_thresholds.json` |
| `count_maps.json` | `aws-lambda/count_maps.json` |
| `group_stats.json` | `aws-lambda/group_stats.json` |
| `review_policy.json` | `aws-lambda/review_policy.json` |
| `feature_columns.json` | `aws-lambda/feature_columns.json` |
| `cat_code_maps.json` | `aws-lambda/cat_code_maps.json` |
| `examples.json` | `aws-lambda/examples.json` |

`count_maps.json` and `group_stats.json` are several MB each -- both are
fetched from S3 at cold start instead of riding in the deploy zip, the same
way the model file itself already is (this is what keeps the zip under the
Lambda size limit -- see the originating issue's "Stop if"). The other
files are small enough to also ride in the zip (below); they still go to S3
too because `_load_models()` always prefers the S3 copy over whatever
shipped in the zip.

```bash
export MODEL_BUCKET=giggit-fraud-radar-models
aws s3 cp outputs/checkpoints/lgb_model.txt "s3://$MODEL_BUCKET/models/lgb_model.txt"
aws s3 cp aws-lambda/isotonic_thresholds.json "s3://$MODEL_BUCKET/models/isotonic_thresholds.json"
aws s3 cp aws-lambda/count_maps.json "s3://$MODEL_BUCKET/models/count_maps.json"
aws s3 cp aws-lambda/group_stats.json "s3://$MODEL_BUCKET/models/group_stats.json"
aws s3 cp aws-lambda/review_policy.json "s3://$MODEL_BUCKET/models/review_policy.json"
aws s3 cp aws-lambda/feature_columns.json "s3://$MODEL_BUCKET/models/feature_columns.json"
aws s3 cp aws-lambda/cat_code_maps.json "s3://$MODEL_BUCKET/models/cat_code_maps.json"
aws s3 cp aws-lambda/examples.json "s3://$MODEL_BUCKET/models/examples.json"
```

### Handler Zip (`lambda-code/deploy.zip` in the model bucket)

`.github/workflows/deploy-lambda.yml` already does this part on every merge
to `main` that touches `aws-lambda/**` -- no manual step needed unless
you're re-running it by hand (`workflow_dispatch`):

- `lambda_function.py`
- `cat_code_maps.json`
- `examples.json`
- `feature_columns.json`
- `isotonic_thresholds.json`

The workflow downloads the existing `lambda-code/deploy.zip` (which already
has the numpy/lightgbm/scipy dependency layer -- see README "Live scoring
API"), swaps in the five files above, re-uploads the zip, then runs:

```bash
aws lambda update-function-code --function-name "$FUNCTION_NAME" \
  --s3-bucket "$MODEL_BUCKET" --s3-key lambda-code/deploy.zip --output text --query 'LastModified'
```

where `FUNCTION_NAME=fraud-score-transaction` and
`MODEL_BUCKET=giggit-fraud-radar-models` (both set as `env:` in the
workflow).

## Order of Operations

1. Upload the eight `models/` files above -- the new model won't serve
   correctly without them, and the handler reads whatever is already in S3,
   so do this before merging the code change.
2. Merge this PR to `main` -- the deploy workflow swaps in the handler zip
   and points the Lambda at it automatically.
3. Smoke test: `curl "$API_URL/health"`, then a real `POST /score` and
   compare against the matching row in `aws-lambda/examples.json`.
