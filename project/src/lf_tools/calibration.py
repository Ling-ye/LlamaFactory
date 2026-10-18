# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Compare overlapping human and model reviews; never infer a quality gate from too few labels."""

from pathlib import Path

from .artifacts import new_directory, read_rows, seal, verify, write_json
from .evaluation import wilson


def calibrate(model_reviews, human_reviews, output):
    for path in (model_reviews, human_reviews):
        verify(path)
    model = {
        r["case_id"]: r for r in read_rows(Path(model_reviews) / "reviews.jsonl") if r["reviewer_type"] == "model"
    }
    human = {
        r["case_id"]: r for r in read_rows(Path(human_reviews) / "reviews.jsonl") if r["reviewer_type"] == "human"
    }
    common = sorted(model.keys() & human.keys())
    if not common:
        raise ValueError("No overlapping human/model review cases")
    errors = {key: [] for key in ("identity", "style", "facts", "naturalness")}
    agreement = 0
    for key in common:
        a, b = model[key], human[key]
        if a["case_sha256"] != b["case_sha256"]:
            raise ValueError("Calibration reviews refer to different anonymous answers")
        agreement += a["winner"] == b["winner"]
        for dimension in errors:
            for side in ("A", "B"):
                left, right = a[side][dimension], b[side][dimension]
                if left is not None and right is not None:
                    errors[dimension].append(abs(left - right))
    result = {
        "paired_cases": len(common),
        "winner_agreement": agreement / len(common),
        "agreement_wilson95": wilson(agreement, len(common)),
        "mean_absolute_score_difference": {k: sum(v) / len(v) if v else None for k, v in errors.items()},
        "warning": "descriptive calibration only; inspect disagreements before relying on model scores",
    }
    output = new_directory(output)
    write_json(output / "calibration.json", result)
    seal(output, {"kind": "judge_calibration"})
    return output
