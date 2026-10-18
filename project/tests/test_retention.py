# Copyright 2026. Licensed under the Apache License, Version 2.0.
import pytest

from lf_tools.artifacts import file_hash
from lf_tools.retention import translated, verify_protected


def test_protected_assets_follow_most_specific_mapping(tmp_path):
    target = tmp_path / "artifacts/v9"
    target.parent.mkdir()
    target.write_bytes(b"protected")
    mapping = [
        {"old": str(tmp_path / "saves"), "new": str(tmp_path / "archive")},
        {"old": str(tmp_path / "saves/v9"), "new": str(target)},
    ]
    inventory = {"saves/v9": {"sha256": file_hash(target), "size": 9}}
    assert verify_protected(tmp_path, inventory, mapping)["files_checked"] == 1
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash changed"):
        verify_protected(tmp_path, inventory, mapping)


def test_migration_cannot_escape_workspace(tmp_path):
    mapping = [{"old": str(tmp_path / "old"), "new": str(tmp_path.parent / "outside")}]
    with pytest.raises(ValueError, match="escapes workspace"):
        translated(tmp_path / "old/file", mapping, tmp_path)
