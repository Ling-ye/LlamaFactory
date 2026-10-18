# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Prepare native Factory YAML and record external official CLI runs; never launch training."""

import math
from pathlib import Path

import yaml

from .artifacts import (
    digest,
    factory_receipt,
    gpu_lock,
    launch_worker,
    new_directory,
    now,
    read_json,
    seal,
    tree_hashes,
    verify,
    write_json,
)
from .data import validate
from .models import identify, identify_adapter, model_path
from .tracking import log_run

PATH_FIELDS = (
    "model_name_or_path",
    "adapter_name_or_path",
    "ref_model",
    "ref_model_adapters",
    "dataset_dir",
    "tokenized_path",
    "output_dir",
    "export_dir",
    "resume_from_checkpoint",
)


def read_native(path, factory_root):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Native Factory YAML must be a mapping")
    for key in PATH_FIELDS:
        if cfg.get(key) and isinstance(cfg[key], str):
            value = Path(cfg[key]).expanduser()
            cfg[key] = str((Path(factory_root) / value).resolve()) if not value.is_absolute() else str(value)
    return cfg


def check_native(cfg):
    if cfg.get("stage", "sft") not in {"sft", "dpo"}:
        raise ValueError("Auxiliary preparation supports native SFT/DPO")
    if cfg.get("quantization_bit") and cfg.get("quantization_method", "bnb") != "bnb":
        raise ValueError("Use the native bnb quantization method")
    if cfg.get("stage") == "dpo":
        parent = cfg.get("adapter_name_or_path")
        if (
            not parent
            or cfg.get("ref_model_adapters") != parent
            or cfg.get("ref_model") != cfg.get("model_name_or_path")
        ):
            raise ValueError("DPO reference must be the same selected SFT parent")
        if cfg.get("ref_model_quantization_bit") != cfg.get("quantization_bit"):
            raise ValueError("DPO reference quantization must match the explicit recipe")
    if cfg.get("train_on_prompt") or cfg.get("packing"):
        raise ValueError("Audited text preparation requires train_on_prompt=false and packing=false")


def prepare(bundle, dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if not validate(dataset)["complete"]:
        raise ValueError("Dataset has unfilled targets")
    manifest = verify(dataset)
    if manifest["stage"] != bundle.training.stage or not manifest["counts"].get("train"):
        raise ValueError("Missing train split or stage mismatch")
    base = identify(model_path(bundle.model))
    parent = identify_adapter(bundle.training.parent_adapter, base) if bundle.training.parent_adapter else None
    if bundle.training.stage == "dpo" and manifest["parent"]["sha256"] != parent["sha256"]:
        raise ValueError("DPO preference parent differs from selected SFT adapter")
    output = new_directory(output)
    cfg = dict(bundle.native)
    cfg.update(dataset_dir=str(dataset), dataset="train", tokenized_path=str(output / "tokenized"))
    check_native(cfg)
    binding = {
        "base": base,
        "parent": parent,
        "dataset_manifest": digest(manifest),
        "configuration": bundle.snapshot(),
        "factory": factory_receipt(bundle.runtime),
        "orchestrator_sha256": digest(tree_hashes(Path(__file__).parent, {".py"})),
    }
    request = {
        "config": cfg,
        "dataset": str(dataset),
        "output": str(output),
        "base": base,
        "fingerprint": digest(binding),
        "model": bundle.model.model_dump(mode="json"),
    }
    write_json(output / "binding.json", binding)
    write_json(output / "request.json", request)
    write_json(output / "source.json", {"config": str(bundle.path)})
    launch_worker(bundle, "prepare", output / "request.json", output / "prepare.log")
    (output / "train.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    seal(output, {"kind": "native_preparation", "binding_sha256": digest(binding), "labels_verified": True})
    return output


def preflight(bundle, prepared, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    manifest = verify(prepared)
    if manifest.get("kind") != "native_preparation":
        raise ValueError("Expected prepared native YAML")
    request = read_json(prepared / "request.json")
    request["output"] = str(new_directory(output))
    write_json(output / "request.json", request)
    with gpu_lock(bundle.runtime.gpu_lock):
        launch_worker(bundle, "preflight", output / "request.json", output / "preflight.log")
    seal(output, {"kind": "preflight", "preparation_sha256": digest(manifest)})
    return output


def start_receipt(prepared, checked):
    prepared, checked = Path(prepared).resolve(), Path(checked).resolve()
    manifest, check = verify(prepared), verify(checked)
    if check.get("preparation_sha256") != digest(manifest):
        raise ValueError("Preflight belongs to another preparation")
    if not read_json(checked / "preflight-result.json").get("within_memory_budget"):
        raise ValueError("Preflight did not pass the physical-memory budget")
    request = read_json(prepared / "request.json")
    adapter = Path(request["config"]["output_dir"])
    if adapter.name != "adapter":
        raise ValueError("Project native output_dir must end with RUN/adapter")
    run = new_directory(adapter.parent)
    binding = read_json(prepared / "binding.json")
    write_json(run / "binding.json", binding)
    write_json(run / "request.json", {**request, "output": str(run)})
    write_json(run / "started.json", {"at": now(), "event": "receipt_before_official_cli"})
    write_json(
        run / "native-receipt.json",
        {
            "prepared": str(prepared),
            "preparation_sha256": digest(manifest),
            "preflight": str(checked),
            "official_command": ["llamafactory-cli", "train", str(prepared / "train.yaml")],
        },
    )
    write_json(run / "status.json", {"status": "awaiting_official_cli"})
    return run


def check_resume(run, checkpoint):
    run, checkpoint = Path(run).resolve(), Path(checkpoint).resolve()
    if (run / "manifest.json").exists():
        raise ValueError("Completed runs are immutable")
    receipt = read_json(run / "native-receipt.json")
    if digest(verify(receipt["prepared"])) != receipt["preparation_sha256"]:
        raise ValueError("Resume preparation changed")
    if not checkpoint.is_relative_to(run / "adapter"):
        raise ValueError("Resume checkpoint belongs to another run")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        if not (checkpoint / name).is_file():
            raise ValueError("Incomplete resume checkpoint: " + name)
    return {"checkpoint": str(checkpoint), "native_override": "resume_from_checkpoint=" + str(checkpoint)}


def finish_receipt(run):
    from .config import Bundle

    run = Path(run).resolve()
    if (run / "manifest.json").exists():
        raise ValueError("Completed runs are immutable")
    receipt = read_json(run / "native-receipt.json")
    prepared = Path(receipt["prepared"])
    if digest(verify(prepared)) != receipt["preparation_sha256"]:
        raise ValueError("Native configuration or encoded data changed")
    binding = read_json(run / "binding.json")
    bundle = Bundle(read_json(prepared / "source.json")["config"])
    if factory_receipt(bundle.runtime) != binding["factory"]:
        raise ValueError("Factory source or dependency versions changed")
    if identify(binding["base"]["path"]) != binding["base"]:
        raise ValueError("Base model changed")
    if binding["parent"] and identify_adapter(binding["parent"]["path"]) != binding["parent"]:
        raise ValueError("Parent adapter changed")
    adapter = run / "adapter"
    state = read_json(adapter / "trainer_state.json")
    if state["global_step"] < 1:
        raise ValueError("Official CLI did not complete an optimizer step")
    for row in state.get("log_history", []):
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in row.values()):
            raise ValueError("Non-finite training metric")
    adapter_binding = {
        "base_sha256": binding["base"]["sha256"],
        "training_binding": digest(binding),
        "template": bundle.model.template,
        "enable_thinking": bundle.model.enable_thinking,
    }
    write_json(adapter / "posttrain-binding.json", adapter_binding)
    for checkpoint in adapter.glob("checkpoint-*"):
        write_json(checkpoint / "posttrain-binding.json", adapter_binding)
    identify_adapter(adapter, binding["base"])
    result = {"optimizer_steps": state["global_step"], "native_cli": True, "preflight": receipt["preflight"]}
    write_json(run / "training-result.json", result)
    write_json(run / "status.json", {"status": "completed", "at": now()})
    seal(
        run,
        {
            "kind": "training",
            "stage": bundle.training.stage,
            "binding_sha256": digest(binding),
            "execution": "official_cli_external",
        },
    )
    log_run(bundle, run, "training", result)
    return run


def verify_reload(bundle, output, model=None, adapter=None):
    base = identify(model or model_path(bundle.model))
    selected = identify_adapter(adapter, base) if adapter else None
    output = new_directory(Path(output).resolve())
    write_json(
        output / "request.json",
        {"base": base, "adapter": selected, "model": bundle.model.model_dump(mode="json"), "output": str(output)},
    )
    with gpu_lock(bundle.runtime.gpu_lock):
        launch_worker(bundle, "reload", output / "request.json", output / "reload.log")
    if not read_json(output / "reload-check.json")["loaded"]:
        raise ValueError("Reload failed")
    seal(output, {"kind": "reload_verification", "base": base, "adapter": selected})
    return output
