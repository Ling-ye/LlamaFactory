# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Model registry binds local bytes even when their upstream revision is unknown."""

from pathlib import Path

from .artifacts import digest, read_json, tree_hashes


def identify(path):
    path = Path(path)
    if not (path / "config.json").is_file():
        raise ValueError(f"Missing model config: {path}")
    files = tree_hashes(path, {".safetensors", ".bin", ".json", ".model", ".jinja", ".txt"})
    if not any(p.endswith((".safetensors", ".bin")) for p in files):
        raise ValueError("No model weights found")
    return {"path": str(path.absolute()), "files": files, "sha256": digest(files)}


def identify_adapter(path, base=None):
    path = Path(path)
    config = read_json(path / "adapter_config.json")
    files = tree_hashes(path, {".safetensors", ".bin", ".json"})
    if not any(p in files for p in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Missing adapter weights")
    receipt = path / "posttrain-binding.json"
    if base is not None:
        if receipt.exists():
            if read_json(receipt)["base_sha256"] != base["sha256"]:
                raise ValueError("Adapter belongs to a different base model")
        elif Path(config.get("base_model_name_or_path", "")).resolve() != Path(base["path"]).resolve():
            raise ValueError("Legacy adapter base path does not match; register the correct base first")
    return {"path": str(path.absolute()), "sha256": digest(files), "files": files}


def model_path(model):
    if model.local_path and model.local_path.exists():
        return model.local_path
    if not model.repo_id:
        raise ValueError("Local model is missing; download it with the official hf CLI first")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model.repo_id, revision=model.revision, local_files_only=True))
