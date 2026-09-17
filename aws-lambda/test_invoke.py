import json
import subprocess

with open("aws-lambda/verification_rows.json") as f:
    rows = json.load(f)
max_abs_diff = 0.0
mismatches = []
for r in rows:
    event = {
        "rawPath": "/score",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps(r["input"]),
        "isBase64Encoded": False,
    }
    with open("/tmp/invoke_payload.json", "w") as f:
        json.dump(event, f)
    subprocess.run(
        [
            "aws", "lambda", "invoke",
            "--function-name", "fraud-score-transaction",
            "--region", "us-east-1",
            "--payload", "fileb:///tmp/invoke_payload.json",
            "/tmp/invoke_out.json",
        ],
        check=True,
        capture_output=True,
    )
    with open("/tmp/invoke_out.json") as f:
        out = json.load(f)
    body = json.loads(out["body"])
    diff = abs(body["calibrated_probability"] - r["local_calibrated_probability"])
    max_abs_diff = max(max_abs_diff, diff)
    if diff != 0.0 or body["review_flag"] != r["local_review_flag"]:
        mismatches.append((r["transaction_id"], body, r["local_calibrated_probability"], r["local_review_flag"]))

print("max_abs_diff:", max_abs_diff)
print("n_rows:", len(rows))
print("n_mismatches:", len(mismatches))
for m in mismatches[:10]:
    print(m)

# print first row's request/response for README
r0 = rows[0]
print("SAMPLE_REQUEST", json.dumps(r0["input"]))
event = {
    "rawPath": "/score",
    "requestContext": {"http": {"method": "POST"}},
    "body": json.dumps(r0["input"]),
    "isBase64Encoded": False,
}
with open("/tmp/invoke_payload.json", "w") as f:
    json.dump(event, f)
subprocess.run(
    [
        "aws", "lambda", "invoke",
        "--function-name", "fraud-score-transaction",
        "--region", "us-east-1",
        "--payload", "fileb:///tmp/invoke_payload.json",
        "/tmp/invoke_out.json",
    ],
    check=True,
    capture_output=True,
)
with open("/tmp/invoke_out.json") as f:
    out = json.load(f)
print("SAMPLE_RESPONSE", out["body"])
