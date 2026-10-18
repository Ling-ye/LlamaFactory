# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Score previously generated real answers; lf_tools eval performs model inference."""

from pathlib import Path

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from lf_tools.opencompass_regression import RegressionDataset, RegressionEvaluator
from lf_tools.opencompass_replay import RecordedModel

workspace_path = str(Path(__file__).resolve().parents[4] / "artifacts")
models = [
    dict(
        type=RecordedModel,
        path=str(Path(workspace_path) / "evals" / name),
        abbr=name,
        max_seq_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=0, num_procs=1),
    )
    for name in ("regression-base", "regression-sft")
]
datasets = [
    dict(
        type=RegressionDataset,
        protocol=str(Path(workspace_path) / "protocols/regression"),
        abbr="authored-regression-12",
        reader_cfg=dict(input_columns=["prompt"], output_column="reference"),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="{prompt}"),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=48),
        ),
        eval_cfg=dict(evaluator=dict(type=RegressionEvaluator)),
    )
]
