# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Run a user-selected OpenCompass benchmark in its own environment."""

import os
import subprocess
from pathlib import Path

from .artifacts import file_hash, gpu_lock, new_directory, read_json, seal, write_json


def run(bundle, config, output):
    if not bundle.runtime.opencompass_python:
        raise ValueError("Set runtime.opencompass_python to an isolated OpenCompass environment")
    config = Path(config).absolute()
    output = new_directory(Path(output).resolve())
    command = [
        str(bundle.runtime.opencompass_python),
        "-m",
        "opencompass.cli.main",
        str(config),
        "--work-dir",
        str(output / "results"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    write_json(output / "request.json", {"command": command, "config_sha256": file_hash(config)})
    with gpu_lock(bundle.runtime.gpu_lock), (output / "opencompass.log").open("w") as stream:
        subprocess.run(command, env=env, cwd=output, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result_files = list((output / "results").glob("*/results/**/*.json"))
    if not result_files:
        raise RuntimeError("OpenCompass produced no scored result files; inspect opencompass.log")
    for path in result_files:
        result = read_json(path)
        if isinstance(result, dict) and (result.get("error") or result.get("errors")):
            raise RuntimeError(f"OpenCompass reported an evaluation error: {path}")
    seal(
        output,
        {
            "kind": "opencompass_evaluation",
            "configuration_sha256": file_hash(config),
            "score_files": [str(p.relative_to(output)) for p in result_files],
        },
    )
    return output
