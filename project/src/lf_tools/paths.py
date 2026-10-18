# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Resolve workspace assets and old recorded paths without changing frozen records."""

import os
from pathlib import Path


def workspace_root():
    return Path(__file__).resolve().parents[4]


def assets_root(root=None):
    root = Path(root) if root is not None else workspace_root()
    configured = os.environ.get("LLM_TRAINING_ASSETS")
    return Path(configured).expanduser().absolute() if configured else root / "llm-training-assets"


def recorded_path(path, root=None):
    path = Path(os.path.abspath(Path(path).expanduser()))
    if path.exists():
        return path
    root = Path(root).absolute() if root is not None else workspace_root()
    assets = assets_root(root)
    mappings = [
        (root.parent / "llm-training-assets", assets),
        (root / "persona-data/workspace/runtime", root / "persona-loom/workspace/runtime"),
        (root / "models", assets / "models"),
        (root / "artifacts", assets / "artifacts"),
        (root / "archive", assets / "archive"),
        (root / "lab", assets / "archive/lab"),
        (root / "saves", assets / "artifacts/legacy-saves"),
        (root / "posttrain/workspace", assets / "artifacts/legacy-posttrain"),
        (root / "persona-data", assets / "archive/legacy-persona-data"),
    ]
    for old, new in mappings:
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    return path
