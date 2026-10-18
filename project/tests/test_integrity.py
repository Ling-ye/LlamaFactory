# Copyright 2026. Licensed under the Apache License, Version 2.0.

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lf_tools.artifacts import gpu_lock, seal, verify, write_json
from lf_tools.config import Data, Source
from lf_tools.opencompass import run


def test_unlisted_artifact_files_rejected(tmp_path):
    write_json(tmp_path / "data.json", {"original": True})
    seal(tmp_path, {"kind": "test"})
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="Artifact changed"):
        verify(tmp_path)


def test_gpu_jobs_are_exclusive(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock(lock):
        with pytest.raises(RuntimeError, match="Another"):
            with gpu_lock(lock):
                pytest.fail("Second job must not acquire the GPU")
    with gpu_lock(lock):
        pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_records_per_split": {"train": -1}},
        {"targets": {"unknown": 2}},
        {"category_weights": {"a": 0.9, "b": 0.5}},
        {"category_weights": {"a": 1.1, "b": -0.1}},
    ],
)
def test_invalid_quotas_rejected(kwargs):
    with pytest.raises(ValidationError):
        Data(name="test", sources=[Source(path=Path("unused"))], **kwargs)


def test_opencompass_zero_exit_without_scores_is_failure(tmp_path, monkeypatch):
    config = tmp_path / "config.py"
    config.write_text("models=[]\ndatasets=[]\n")
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(opencompass_python=Path("/python"), gpu_lock=tmp_path / "gpu.lock")
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="no scored result"):
        run(bundle, config, tmp_path / "output")


def test_opencompass_reports_failure_in_result(tmp_path, monkeypatch):
    config = tmp_path / "config.py"
    config.write_text("models=[]\ndatasets=[]\n")
    output = tmp_path / "output"
    bundle = SimpleNamespace(
        runtime=SimpleNamespace(opencompass_python=Path("/python"), gpu_lock=tmp_path / "gpu.lock")
    )

    def fake(*args, **kwargs):
        write_json(output / "results" / "timestamp" / "results" / "model" / "dataset.json", {"error": "failed"})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="evaluation error"):
        run(bundle, config, output)


def test_local_tracking_has_no_repository_root_artifacts(tmp_path):
    from lf_tools.tracking import log_run

    runtime = SimpleNamespace(tracking="mlflow", tracking_db=tmp_path / "tracking" / "tracking.sqlite")
    bundle = SimpleNamespace(
        runtime=runtime,
        experiment=SimpleNamespace(name="test-local"),
        model=SimpleNamespace(name="test"),
        training=SimpleNamespace(stage="sft", seed=42),
        character=None,
        snapshot=lambda: {"test": True},
    )
    output = tmp_path / "run"
    output.mkdir()
    write_json(output / "result.json", {"steps": 1})
    seal(output, {"kind": "test"})
    log_run(bundle, output, "test", {"steps": 1})
    import mlflow

    experiment = mlflow.get_experiment_by_name("test-local")
    assert str(tmp_path / "tracking" / "artifacts") in experiment.artifact_location
    assert (tmp_path / "run.tracking.json").exists()
    verify(output)
