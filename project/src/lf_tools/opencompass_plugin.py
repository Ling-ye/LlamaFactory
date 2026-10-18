# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Optional OpenCompass plugins. These use the same rule semantics as local evaluation."""

import json

from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .artifacts import read_json, read_rows, verify
from .evaluation import rule_result


@LOAD_DATASET.register_module()
class PosttrainDataset(BaseDataset):
    @staticmethod
    def load(protocol):
        verify(protocol)
        from pathlib import Path

        frozen = read_json(Path(protocol) / "protocol.json")
        cases = read_rows(Path(protocol) / "cases.jsonl")
        # Explicitly reject flattening multi-turn tests into one-turn benchmarks.
        if any(len(c["users"]) != 1 for c in cases):
            raise ValueError("Use lf_tools eval for autoregressive multi-turn cases")
        return Dataset.from_list(
            [
                {
                    "prompt": c["users"][0],
                    "reference": json.dumps(
                        {"user": c["users"][0], "rules": (frozen["character"] or {}).get("rules", [])},
                        ensure_ascii=False,
                    ),
                }
                for c in cases
            ]
        )


@ICL_EVALUATORS.register_module()
class PosttrainRuleEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Missing benchmark predictions")
        results, totals, details = {}, {}, []
        for prediction, reference in zip(predictions, references, strict=True):
            reference = json.loads(reference) if isinstance(reference, str) else reference
            checks = {r["id"]: rule_result(r, prediction, reference["user"]) for r in reference["rules"]}
            for key, passed in checks.items():
                if passed is not None:
                    results[key] = results.get(key, 0) + int(passed)
                    totals[key] = totals.get(key, 0) + 1
            details.append({"prediction": prediction, "rules": checks})
        return {**{key: 100 * count / totals[key] for key, count in results.items()}, "details": details}
