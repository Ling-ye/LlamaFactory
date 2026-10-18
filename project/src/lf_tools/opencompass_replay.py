# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Replay recorded first-turn answers to verify OpenCompass integration; no new inference."""

import json
from pathlib import Path

from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.models.base import BaseModel
from opencompass.registry import LOAD_DATASET, MODELS

from .artifacts import read_json, read_rows, verify


@LOAD_DATASET.register_module()
class RecordedDataset(BaseDataset):
    @staticmethod
    def load(path):
        verify(path)
        path = Path(path)
        character = read_json(path / "protocol.json")["character"]
        rows = read_rows(path / "responses.jsonl")
        return Dataset.from_list(
            [
                {
                    "prompt": row["messages"][0]["content"],
                    "reference": json.dumps(
                        {"user": row["messages"][0]["content"], "rules": (character or {}).get("rules", [])},
                        ensure_ascii=False,
                    ),
                }
                for row in rows
                if row["turn"] == 0
            ]
        )


@MODELS.register_module()
class RecordedModel(BaseModel):
    def __init__(self, path, **kwargs):
        super().__init__(path=path, **kwargs)
        verify(path)
        rows = [r for r in read_rows(Path(path) / "responses.jsonl") if r["turn"] == 0]
        self.answers = {r["messages"][0]["content"].strip(): r["text"] for r in rows}
        if len(self.answers) != len(rows):
            raise ValueError("Replay prompts are ambiguous")

    def generate(self, inputs, max_out_len):
        return [self.answers[prompt.strip()] for prompt in inputs]

    def get_token_len(self, prompt):
        return len(prompt)  # recorded-string sizing, not a tokenizer/performance measurement
