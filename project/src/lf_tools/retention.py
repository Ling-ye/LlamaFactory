# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Read-only migration and protected-asset verification. No deletion API."""

from pathlib import Path

from .artifacts import file_hash


def translated(path, mapping, root):
    root, path = Path(root).resolve(), Path(path).absolute()
    if not path.is_relative_to(root):
        raise ValueError("Historical path escapes workspace")
    matches = []
    for item in mapping:
        old, new = Path(item["old"]), Path(item["new"])
        if path.is_relative_to(old):
            matches.append((len(old.parts), new / path.relative_to(old)))
    result = max(matches, key=lambda item: item[0])[1] if matches else path
    if not result.resolve().is_relative_to(root):
        raise ValueError("Migration destination escapes workspace")
    return result


def verify_protected(root, inventory, mapping):
    root = Path(root).resolve()
    checked = []
    for relative, expected in inventory.items():
        destination = translated(root / relative, mapping, root)
        if not destination.is_file() or destination.stat().st_size != expected["size"]:
            raise ValueError("Protected asset missing or size changed: " + relative)
        if file_hash(destination) != expected["sha256"]:
            raise ValueError("Protected asset hash changed: " + relative)
        checked.append(
            {
                "original": relative,
                "current": str(destination.relative_to(root)),
                "sha256": expected["sha256"],
                "size": expected["size"],
            }
        )
    return {"passed": True, "files_checked": len(checked), "files": checked}
