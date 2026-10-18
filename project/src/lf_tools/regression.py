# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Strict short-answer regression scoring, separate from role likeness."""

import re

from .evaluation import wilson


def normalized_answer(text):
    return re.sub(r"\s+", "", text).strip("。.!！")


def score_regression(predictions, references):
    if len(predictions) != len(references) or not references:
        raise ValueError("Regression needs matching, nonempty predictions and references")
    totals, passed, details = {}, {}, []
    for answer, reference in zip(predictions, references, strict=True):
        category = reference["category"]
        ok = normalized_answer(answer) in {normalized_answer(x) for x in reference["answers"]}
        totals[category] = totals.get(category, 0) + 1
        passed[category] = passed.get(category, 0) + int(ok)
        details.append({"id": reference["id"], "passed": ok, "prediction": answer})
    correct = sum(passed.values())
    return {
        "accuracy": 100 * correct / len(references),
        **{k: 100 * passed[k] / n for k, n in totals.items()},
        "wilson95": wilson(correct, len(references)),
        "details": details,
    }
