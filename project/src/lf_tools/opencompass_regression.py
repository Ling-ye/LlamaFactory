# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Score actual saved local model inference through OpenCompass."""

import json
from pathlib import Path

from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .artifacts import read_rows, verify
from .regression import score_regression


@LOAD_DATASET.register_module()
class RegressionDataset(BaseDataset):
    @staticmethod
    def load(protocol):
        verify(protocol)
        cases = read_rows(Path(protocol) / "cases.jsonl")
        if any(len(c["users"]) != 1 or not c.get("answers") for c in cases):
            raise ValueError("Regression requires single-turn cases with accepted short answers")
        return Dataset.from_list(
            [
                {
                    "prompt": c["users"][0],
                    "reference": json.dumps(
                        {"id": c["id"], "category": c["category"], "answers": c["answers"]}, ensure_ascii=False
                    ),
                }
                for c in cases
            ]
        )


@ICL_EVALUATORS.register_module()
class RegressionEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        result = score_regression(predictions, [json.loads(r) if isinstance(r, str) else r for r in references])
        low, high = result.pop("wilson95")
        result.update(accuracy_ci_low=100 * low, accuracy_ci_high=100 * high)
        return result
