"""Scores the ten served examples through the Lambda handler's own code path
(no S3, no network -- every asset is staged locally at the /tmp paths
_load_models() expects) and checks the handler reproduces the
calibrated_probability already recorded in examples.json to six decimals.
examples.json is what the site's "Score a Transaction" receipt compares
against, so this is the check that the served scores match the served
examples after a deploy -- see the originating issue's "Stop if"."""
import importlib.util
import json
import os
import shutil
import sys

spec = importlib.util.spec_from_file_location("lambda_function", "aws-lambda/lambda_function.py")
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)

os.makedirs("/tmp", exist_ok=True)
for fname, src in [
    ("lgb_model.txt", "outputs/checkpoints/lgb_model.txt"),
    ("feature_columns.json", "aws-lambda/feature_columns.json"),
    ("cat_code_maps.json", "aws-lambda/cat_code_maps.json"),
    ("count_maps.json", "aws-lambda/count_maps.json"),
    ("group_stats.json", "aws-lambda/group_stats.json"),
    ("isotonic_thresholds.json", "aws-lambda/isotonic_thresholds.json"),
    ("review_policy.json", "aws-lambda/review_policy.json"),
]:
    if not os.path.exists(src):
        sys.exit(
            f"missing {src} -- run this on a machine with the real trained model "
            "(outputs/checkpoints/) present; see aws-lambda/DEPLOY.md"
        )
    shutil.copy(src, f"/tmp/{fname}")

with open("aws-lambda/examples.json") as f:
    examples = json.load(f)

max_abs_diff = 0.0
mismatches = []
for ex in examples:
    result = lf.score_one(ex["input"])
    diff = abs(result["calibrated_probability"] - ex["calibrated_probability"])
    max_abs_diff = max(max_abs_diff, diff)
    flag_match = result["review_flag"] == ex["review_flag"]
    if diff != 0.0 or not flag_match:
        mismatches.append((ex["example_id"], result, ex["calibrated_probability"], ex["review_flag"]))

print("max_abs_diff:", max_abs_diff)
print("n_examples:", len(examples))
print("n_mismatches:", len(mismatches))
for m in mismatches[:10]:
    print(m)

if mismatches:
    sys.exit(1)
assert len(examples) == 10, f"expected 10 examples, got {len(examples)}"
print("OK: handler matches examples.json on all", len(examples), "examples to 6 decimals")
