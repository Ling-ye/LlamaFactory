# Copyright 2026. Licensed under the Apache License, Version 2.0.
from pathlib import Path

import pytest

from lf_tools import paths
from lf_tools.artifacts import read_json, read_rows, seal, verify, write_json, write_rows
from lf_tools.config import Bundle
from lf_tools.models import identify, identify_adapter


def external_fixture(tmp_path, monkeypatch):
    workspace = tmp_path / "llm-training"
    workspace.mkdir()
    assets = tmp_path / "llm-training-assets"
    assets.mkdir()
    monkeypatch.setattr(paths, "workspace_root", lambda: workspace)
    monkeypatch.delenv("LLM_TRAINING_ASSETS", raising=False)
    return workspace, assets


def test_frozen_records_remain_verifiable_at_recorded_old_path(tmp_path, monkeypatch):
    workspace, assets = external_fixture(tmp_path, monkeypatch)
    directory = assets / "artifacts/dataset"
    write_json(directory / "metadata.json", {"original_path": str(workspace / "models/base")})
    write_rows(directory / "records.jsonl", [{"id": "a"}])
    manifest = seal(directory, {"kind": "dataset"})
    old = workspace / "artifacts/dataset"
    assert verify(old) == manifest
    assert read_json(old / "metadata.json")["original_path"] == str(workspace / "models/base")
    assert read_rows(old / "records.jsonl") == [{"id": "a"}]
    assert not (workspace / "artifacts").exists()


def test_legacy_adapter_base_path_matches_relocated_base(tmp_path, monkeypatch):
    workspace, assets = external_fixture(tmp_path, monkeypatch)
    base = assets / "models/base"
    write_json(base / "config.json", {"model_type": "test"})
    (base / "model.safetensors").write_bytes(b"base")
    adapter = assets / "artifacts/adapter"
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": str(workspace / "models/base")})
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    assert identify_adapter(adapter, identify(base))["path"] == str(adapter)
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": str(workspace / "models/other")})
    with pytest.raises(ValueError, match="base path"):
        identify_adapter(adapter, identify(base))


def test_paths_outside_workspace_are_not_remapped(tmp_path, monkeypatch):
    workspace, assets = external_fixture(tmp_path, monkeypatch)
    other = tmp_path / "unrelated/models/missing"
    assert paths.recorded_path(other) == other
    existing = workspace / "file"
    existing.write_text("keep")
    assert paths.recorded_path(existing) == existing
    assert paths.recorded_path(workspace / "posttrain/workspace/run") == assets / "artifacts/legacy-posttrain/run"


def test_active_configs_use_external_assets():
    project = Path(__file__).resolve().parents[1]
    workspace = project.parents[1]
    assets = workspace.parent / "llm-training-assets"
    for name in ("isekaijoucho-smoke", "isekaijoucho", "general-qa", "starlamp", "dpo-smoke"):
        bundle = Bundle(project / "configs" / (name + ".yaml"))
        assert bundle.experiment.workspace.resolve().is_relative_to(assets)
        assert bundle.model.local_path.resolve().is_relative_to(assets / "models")
        assert Path(bundle.native["output_dir"]).is_relative_to(assets / "artifacts")
        assert bundle.runtime.gpu_lock.resolve().is_relative_to(assets)
        assert bundle.runtime.tracking_db.resolve().is_relative_to(assets)
