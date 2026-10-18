# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Strict configs; filesystem paths are relative to their YAML file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Runtime(Strict):
    factory_root: Path
    training_python: Path
    factory_commit: str | None = None
    gpu_lock: Path
    opencompass_python: Path | None = None
    tracking: Literal["mlflow", "none"] = "none"
    tracking_db: Path | None = None


class Model(Strict):
    name: str
    local_path: Path | None = None
    repo_id: str | None = None
    revision: str | None = None
    template: str
    enable_thinking: bool = False

    @model_validator(mode="after")
    def check_source(self):
        if not self.local_path and not self.repo_id:
            raise ValueError("local_path or repo_id is required")
        if self.repo_id and (
            not self.revision or len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision)
        ):
            raise ValueError("Hub downloads require an immutable, 40-character commit revision")
        return self


class Rule(Strict):
    id: str
    kind: Literal["contains_any", "forbid_regex", "suffix_regex", "max_chars", "exact_suffix_library"]
    values: list[str] = Field(default_factory=list)
    pattern: str | None = None
    limit: int | None = Field(default=None, gt=0)
    library: Path | None = None
    trigger: str | None = None
    threshold: float = Field(default=0.95, ge=0, le=1)

    @model_validator(mode="after")
    def check_parameters(self):
        import re

        if self.kind == "contains_any" and not self.values:
            raise ValueError("contains_any requires values")
        if self.kind in {"forbid_regex", "suffix_regex"} and not self.pattern:
            raise ValueError("regex rules require a pattern")
        if self.kind == "max_chars" and not self.limit:
            raise ValueError("max_chars requires a limit")
        if self.kind == "exact_suffix_library" and not self.library:
            raise ValueError("exact_suffix_library requires a library")
        for pattern in (self.pattern, self.trigger):
            if pattern:
                re.compile(pattern)
        return self


class Character(Strict):
    id: str
    language: str = "zh"
    identity: str
    world: str = ""
    timeline: str = ""
    relationships: list[str] = Field(default_factory=list)
    expression: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    short_prompt: str
    full_prompt: str
    sources: list[Path] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    mask_patterns: list[str] = Field(default_factory=list)
    semantic_checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_rules(self):
        if len({r.id for r in self.rules}) != len(self.rules):
            raise ValueError("Duplicate rule ID")
        return self


class Source(Strict):
    path: Path
    format: Literal["messages", "alpaca", "persona_records", "preferences"] = "messages"
    split: Literal["train", "validation", "test"] | None = None
    provenance: Path | None = None
    kind: Literal["observed", "translated", "rewritten", "synthetic", "unknown"] = "unknown"


class Data(Strict):
    name: str
    stage: Literal["sft", "dpo"] = "sft"
    sources: list[Source]
    seed: int = 42
    empty_system_ratio: float | None = Field(default=None, ge=0, le=1)
    targets: dict[str, int] = Field(default_factory=dict)
    max_records_per_split: dict[str, int] = Field(default_factory=dict)
    category_weights: dict[str, float] = Field(default_factory=dict)
    near_duplicate_threshold: float = Field(default=0.9, gt=0, le=1)
    parent_adapter: Path | None = None

    @model_validator(mode="after")
    def check_stage(self):
        if self.stage == "dpo" and (self.empty_system_ratio is not None or not self.parent_adapter):
            raise ValueError("DPO requires parent_adapter and preserves the original shared prompt")
        for limits in (self.targets, self.max_records_per_split):
            if any(k not in {"train", "validation", "test"} or v < 0 for k, v in limits.items()):
                raise ValueError("Invalid split target")
        if self.category_weights and (
            any(v < 0 for v in self.category_weights.values()) or abs(sum(self.category_weights.values()) - 1) > 1e-6
        ):
            raise ValueError("Category weights must be nonnegative and sum to one")
        return self


class Evaluation(Strict):
    name: str
    cases: Path
    purpose: Literal["development", "heldout", "smoke"] = "development"
    system_mode: Literal["empty", "short", "full"] = "empty"
    seed: int = 42
    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    expected_cases: int | None = Field(default=None, gt=0)


class Experiment(Strict):
    name: str
    workspace: Path
    runtime: Path
    data: Path
    training: Path
    evaluation: Path
    character: Path | None = None


def _resolve(value, base):
    if isinstance(value, Path):
        return (base / value.expanduser()).absolute() if not value.is_absolute() else value
    if isinstance(value, BaseModel):
        for key in type(value).model_fields:
            setattr(value, key, _resolve(getattr(value, key), base))
    elif isinstance(value, list):
        return [_resolve(x, base) for x in value]
    return value


def read_config(path: Path, cls):
    path = Path(path).absolute()
    value = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return _resolve(value, path.parent)


class Bundle:
    """Auxiliary metadata references an unchanged, native Factory training YAML."""

    def __init__(self, path):
        from types import SimpleNamespace

        from .native import check_native, read_native

        self.path = Path(path).absolute()
        self.experiment = read_config(self.path, Experiment)
        self.runtime = read_config(self.experiment.runtime, Runtime)
        self.native = read_native(self.experiment.training, self.runtime.factory_root)
        check_native(self.native)
        self.model = Model(
            name=Path(self.native["model_name_or_path"]).name,
            local_path=Path(self.native["model_name_or_path"]),
            template=self.native["template"],
            enable_thinking=self.native.get("enable_thinking", False),
        )
        self.data = read_config(self.experiment.data, Data)
        self.evaluation = read_config(self.experiment.evaluation, Evaluation)
        self.character = read_config(self.experiment.character, Character) if self.experiment.character else None
        self.training = SimpleNamespace(
            stage=self.native.get("stage", "sft"),
            seed=self.native.get("seed", 42),
            cutoff_len=self.native.get("cutoff_len", 2048),
            parent_adapter=Path(self.native["adapter_name_or_path"])
            if self.native.get("adapter_name_or_path")
            else None,
        )
        if self.training.stage != self.data.stage:
            raise ValueError("Dataset and native training stages differ")
        if self.data.empty_system_ratio is not None and not self.character:
            raise ValueError("System prompt mixing requires a character pack")
        if self.data.stage == "dpo" and self.data.parent_adapter.resolve() != self.training.parent_adapter.resolve():
            raise ValueError("DPO data and training must bind the same SFT parent")

    def snapshot(self):
        value = {
            k: getattr(self, k).model_dump(mode="json") if getattr(self, k) is not None else None
            for k in ("experiment", "runtime", "model", "data", "evaluation", "character")
        }
        value["training"] = self.native
        return value
