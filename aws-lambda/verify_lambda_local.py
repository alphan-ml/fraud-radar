import importlib.util
import json
import shutil

spec = importlib.util.spec_from_file_location("lambda_function", "aws-lambda/lambda_function.py")
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)

# stage assets at the /tmp paths _load_models expects, skipping S3
import os

os.makedirs("/tmp", exist_ok=True)
for fname, src in [
    ("lgb_model.txt", "outputs/checkpoints/lgb_model.txt"),
    ("feature_columns.json", "aws-lambda/feature_columns.json"),
    ("cat_code_maps.json", "aws-lambda/cat_code_maps.json"),
    ("count_maps.json", "aws-lambda/count_maps.json"),
    ("isotonic_thresholds.json", "aws-lambda/isotonic_thresholds.json"),
]:
    shutil.copy(src, f"/tmp/{fname}")

with open("aws-lambda/verification_rows.json") as f:
    rows = json.load(f)

max_abs_diff = 0.0
mismatches = []
for r in rows:
    result = lf.score_one(r["input"])
    diff = abs(result["calibrated_probability"] - r["local_calibrated_probability"])
    max_abs_diff = max(max_abs_diff, diff)
    flag_match = result["review_flag"] == r["local_review_flag"]
    if diff != 0.0 or not flag_match:
        mismatches.append((r["transaction_id"], result, r["local_calibrated_probability"], r["local_review_flag"]))

print("max_abs_diff:", max_abs_diff)
print("n_rows:", len(rows))
print("n_mismatches:", len(mismatches))
for m in mismatches[:10]:
    print(m)
