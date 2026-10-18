# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Optional local MLflow: native artifact records remain the source of truth."""

import json
from pathlib import Path

from .artifacts import write_json


def log_run(bundle, directory, kind, metrics=None):
    if bundle.runtime.tracking == "none":
        return
    if not bundle.runtime.tracking_db:
        raise ValueError("MLflow requires a local tracking_db")
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install factory-project-tools[tracking] in the orchestration environment") from exc
    db = bundle.runtime.tracking_db
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(db))
    experiment = mlflow.get_experiment_by_name(bundle.experiment.name)
    if experiment is None:
        from .artifacts import digest

        artifacts = db.parent / "artifacts" / digest(bundle.experiment.name)[:16]
        artifacts.mkdir(parents=True, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            bundle.experiment.name, artifact_location=artifacts.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    directory = Path(directory)
    with mlflow.start_run(run_name=directory.name) as run:
        mlflow.set_tags({"lf_tools.kind": kind, "lf_tools.artifact": str(directory.absolute())})
        mlflow.log_params(
            {
                "base_model": bundle.model.name,
                "stage": bundle.training.stage,
                "character": bundle.character.id if bundle.character else "none",
                "seed": bundle.training.seed,
            }
        )
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_text(json.dumps(bundle.snapshot(), ensure_ascii=False, indent=2), "configuration.json")
        manifest = directory / "manifest.json"
        if manifest.exists():
            mlflow.log_artifact(str(manifest))
        # Keep mutable tracker metadata outside the immutable experiment.
        write_json(directory.parent / f"{directory.name}.tracking.json", {"run_id": run.info.run_id})
