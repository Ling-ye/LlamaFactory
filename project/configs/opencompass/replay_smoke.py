# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Integration test on recorded answers, not fresh model inference."""

from pathlib import Path

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from lf_tools.opencompass_plugin import PosttrainRuleEvaluator
from lf_tools.opencompass_replay import RecordedDataset, RecordedModel

run_path = str(Path(__file__).resolve().parents[4] / "artifacts/evals/layout-base-empty")
models = [
    dict(
        type=RecordedModel,
        path=run_path,
        abbr="recorded-base-empty",
        max_seq_len=4096,
        batch_size=1,
        run_cfg=dict(num_gpus=0, num_procs=1),
    )
]
datasets = [
    dict(
        type=RecordedDataset,
        path=run_path,
        abbr="recorded-role-smoke",
        reader_cfg=dict(input_columns=["prompt"], output_column="reference"),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="{prompt}"),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=256),
        ),
        eval_cfg=dict(evaluator=dict(type=PosttrainRuleEvaluator)),
    )
]
