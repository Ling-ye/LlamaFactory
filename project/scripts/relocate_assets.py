#!/usr/bin/env python3
"""Relocate shared assets between the two supported workspace layouts."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def verify_protected(directory, rows):
    for item in rows:
        path = (directory / item["current"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("Protected asset missing or changed: " + item["current"])


def main(workspace, source, target):
    workspace, source, target = workspace.resolve(), source.resolve(), target.resolve()
    allowed = {workspace / "llm-training-assets", workspace.parent / "llm-training-assets"}
    if source not in allowed or target not in allowed or source == target or target.exists():
        raise ValueError("Choose distinct supported asset locations; the destination must be new")
    if not source.is_dir() or source.is_symlink() or (source / ".git").exists():
        raise ValueError("Expected a real asset directory, not a repository")
    if not all((workspace / name / ".git").is_dir() for name in ("persona-loom", "llama-factory")):
        raise ValueError("Expected two independent Git repositories")
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise ValueError("Relocation must use a same-filesystem directory rename")
    proof = source / "archive/migration/20261018-external-assets/protected-assets.json"
    rows = json.loads(proof.read_text(encoding="utf-8"))["files"]
    verify_protected(source, rows)
    print("Protected assets verified before relocation:", len(rows), flush=True)
    before = inventory(source)
    with (source / "artifacts/gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source.rename(target)
        try:
            if inventory(target) != before:
                raise ValueError("File metadata changed during directory rename")
            verify_protected(target, rows)
        except BaseException:
            target.rename(source)
            raise
        name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") + "-workspace-assets"
        record = target / "archive/migration" / name
        index = 1
        while record.exists():
            record = target / "archive/migration" / (name + "-" + str(index))
            index += 1
        record.mkdir(parents=True)
        save(record / "inventory-before.json", before)
        save(
            record / "protected-assets.json",
            {
                "passed": True,
                "files_checked": len(rows),
                "paths_relative_to": "assets_root",
                "files": rows,
            },
        )
        rebased_rows = 0
        db = target / "artifacts/tracking/mlflow.sqlite"
        if db.exists():
            with sqlite3.connect(db) as connection:
                with sqlite3.connect(record / "mlflow-before.sqlite") as backup:
                    connection.backup(backup)
                for table, column in (
                    ("experiments", "artifact_location"),
                    ("runs", "artifact_uri"),
                    ("tags", "value"),
                ):
                    cursor = connection.execute(
                        f'UPDATE "{table}" SET "{column}"=replace("{column}",?,?) WHERE instr("{column}",?)>0',
                        (str(source), str(target), str(source)),
                    )
                    rebased_rows += cursor.rowcount
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("MLflow database integrity check failed")
        report = {
            "passed": True,
            "workspace": str(workspace),
            "source": str(source),
            "assets_root": str(target),
            "file_entries": len(before),
            "regular_file_bytes": sum(row["size"] for row in before.values() if row["symlink"] is None),
            "same_filesystem_rename": True,
            "metadata_identical_before_tracking_rebase": True,
            "protected_files_verified": len(rows),
            "mlflow_rows_rebased": rebased_rows,
            "historical_manifests_rewritten": False,
            "current_model": "v9",
            "v10": "paused",
        }
        save(record / "relocation.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    main(args.workspace, args.source, args.assets)
