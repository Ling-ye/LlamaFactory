#!/usr/bin/env python3
"""One-time same-filesystem relocation; preserve frozen records and verify assets."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inventory(directory):
    rows = {}
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            rows[str(path.relative_to(directory))] = {
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "inode": info.st_ino,
                "symlink": os.readlink(path) if path.is_symlink() else None,
            }
    return rows


def main(workspace, destination):
    workspace, destination = workspace.resolve(), destination.resolve()
    if destination.parent != workspace.parent or destination == workspace or destination.exists():
        raise ValueError("Use a new, explicitly named sibling asset directory")
    if not all((workspace / name / ".git").is_dir() for name in ("persona-loom", "llama-factory")):
        raise ValueError("Expected two independent Git repositories")
    sources = [workspace / name for name in ("models", "artifacts", "archive")]
    if any(not path.is_dir() or path.is_symlink() for path in sources):
        raise ValueError("Expected real asset directories")
    if any(path.stat().st_dev != destination.parent.stat().st_dev for path in sources):
        raise ValueError("Relocation must be an atomic rename on the same filesystem")
    original = json.loads(
        (workspace / "archive/migration/20261018-layout/protected-verification-after-cleanup.json").read_text()
    )
    for item in original["files"]:
        if sha(workspace / item["current"]) != item["sha256"]:
            raise ValueError("Protected asset changed before migration: " + item["current"])
    print("Verified protected assets before move:", len(original["files"]), flush=True)
    before = {source.name: inventory(source) for source in sources}
    repositories = {
        name: {
            "head": subprocess.check_output(
                ["git", "-C", str(workspace / name), "rev-parse", "HEAD"], text=True
            ).strip(),
            "status": subprocess.check_output(
                ["git", "-C", str(workspace / name), "status", "--porcelain"], text=True
            ),
        }
        for name in ("persona-loom", "llama-factory")
    }
    with (workspace / "artifacts/gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination.mkdir()
        moved = []
        try:
            for source in sources:
                target = destination / source.name
                if not source.resolve().is_relative_to(workspace) or target.parent != destination:
                    raise ValueError("Resolved move leaves the intended roots")
                source.rename(target)
                moved.append(source.name)
            for name in moved:
                if inventory(destination / name) != before[name]:
                    raise ValueError("File metadata changed during rename: " + name)
        except BaseException:
            for name in reversed(moved):
                (destination / name).rename(workspace / name)
            destination.rmdir()
            raise
        record = destination / "archive/migration/20261018-external-assets"
        record.mkdir(parents=True)
        save(record / "inventory-before.json", before)
        save(record / "repositories-before.json", repositories)
        latest = destination / "artifacts/legacy-saves/qwen35-4b/isekaijoucho-latest"
        previous_link = os.readlink(latest)
        target = Path("isekaijoucho-persona-experience-v4-c611a8a-bf16/checkpoint-100")
        if not (latest.parent / target).is_dir():
            raise ValueError("Current v9 adapter is missing")
        latest.unlink()
        latest.symlink_to(target, target_is_directory=True)
        checks = []
        for item in original["files"]:
            path = destination / item["current"]
            if sha(path) != item["sha256"]:
                raise ValueError("Protected asset changed after migration: " + item["current"])
            checks.append({**item, "current": str(path.relative_to(destination))})
        save(
            record / "protected-assets.json",
            {"passed": True, "files_checked": len(checks), "paths_relative_to": "assets_root", "files": checks},
        )
        # Preserve the original database before rebasing only its live tracking locations.
        db = destination / "artifacts/tracking/mlflow.sqlite"
        rebased_rows = 0
        if db.exists():
            with sqlite3.connect(db) as connection:
                with sqlite3.connect(record / "mlflow-before.sqlite") as backup:
                    connection.backup(backup)
                for table, column in (
                    ("experiments", "artifact_location"),
                    ("runs", "artifact_uri"),
                    ("tags", "value"),
                ):
                    for name in ("models", "artifacts", "archive"):
                        old, new = str(workspace / name), str(destination / name)
                        cursor = connection.execute(
                            f'UPDATE "{table}" SET "{column}"=replace("{column}",?,?) WHERE instr("{column}",?)>0',
                            (old, new, old),
                        )
                        rebased_rows += cursor.rowcount
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("MLflow database integrity check failed")
        if (workspace / "README.md").is_file():
            (workspace / "README.md").rename(record / "workspace-README-before.md")
        report = {
            "passed": True,
            "workspace": str(workspace),
            "assets_root": str(destination),
            "moves": [{"old": str(workspace / name), "new": str(destination / name)} for name in moved],
            "files": sum(len(value) for value in before.values()),
            "regular_file_bytes": sum(
                row["size"] for group in before.values() for row in group.values() if row["symlink"] is None
            ),
            "same_filesystem_rename": True,
            "metadata_inventory_identical_before_link_repair": True,
            "protected_files_verified": len(checks),
            "current_model_symlink": {"before": previous_link, "after": str(target)},
            "mlflow_rows_rebased": rebased_rows,
            "mlflow_backup": "mlflow-before.sqlite",
            "historical_manifests_rewritten": False,
            "current_model": "v9",
            "v10": "paused",
        }
        save(record / "relocation.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "moves"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    main(args.workspace, args.assets)
