"""The Lambda serves aws-lambda/examples.json to browsers. A browser's
JSON.parse rejects Python's NaN token, so the file must be strict JSON with
null for missing values. This is what broke the site's example picker."""
from __future__ import annotations

import json
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "aws-lambda" / "examples.json"


def _reject_constant(name):
    raise ValueError(f"non-JSON constant in examples.json: {name}")


def test_examples_json_is_strict_json():
    text = EXAMPLES.read_text()
    rows = json.loads(text, parse_constant=_reject_constant)
    assert len(rows) == 10
    assert all("input" in r and "example_id" in r for r in rows)


def test_examples_missing_values_are_null_not_nan():
    text = EXAMPLES.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "null" in text
