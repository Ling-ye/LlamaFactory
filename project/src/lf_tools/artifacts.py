# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Hash-bound immutable artifacts and process-scoped local GPU locking."""

import contextlib
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .paths import recorded_path

_GPU_LOCK_FD = None


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(recorded_path(path).read_text(encoding="utf-8"))


def read_rows(path):
    path = recorded_path(path)
    if path.suffix == ".json":
        value = read_json(path)
        if isinstance(value, dict) and "cases" in value:
            value = value["cases"]
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array: {path}")
        return value
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def tree_hashes(path, suffixes=None):
    path = Path(path)
    return {
        str(p.relative_to(path)): file_hash(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and (suffixes is None or p.suffix in suffixes)
        and not any(part.startswith(".") for part in p.relative_to(path).parts)
    }


def seal(path, metadata):
    path = Path(path)
    files = tree_hashes(path)
    files.pop("manifest.json", None)
    manifest = {**metadata, "files": files, "content_sha256": digest(files)}
    write_json(path / "manifest.json", manifest)
    return manifest


def verify(path):
    path = recorded_path(path)
    manifest = read_json(path / "manifest.json")
    if digest(manifest["files"]) != manifest["content_sha256"]:
        raise ValueError("Invalid manifest digest")
    actual = tree_hashes(path)
    actual.pop("manifest.json", None)
    if actual != manifest["files"]:
        raise ValueError("Artifact changed: file set or contents differ")
    for name, expected in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_hash(target) != expected:
            raise ValueError(f"Artifact changed: {name}")
    return manifest


def now():
    return datetime.now(timezone.utc).isoformat()


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


@contextlib.contextmanager
def gpu_lock(path):
    global _GPU_LOCK_FD
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another lf_tools GPU operation is running") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            _GPU_LOCK_FD = handle.fileno()
            yield
        finally:
            _GPU_LOCK_FD = None
            fcntl.flock(handle, fcntl.LOCK_UN)


def factory_receipt(runtime):
    root = runtime.factory_root
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if runtime.factory_commit and actual != runtime.factory_commit:
        raise ValueError(f"Factory commit mismatch: expected {runtime.factory_commit}, found {actual}")
    sources = tree_hashes(root / "src" / "llamafactory", {".py"})
    package_script = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({n:m.version(n) for n in "
        '["torch","transformers","datasets","peft","trl"]}))'
    )
    versions = json.loads(subprocess.check_output([str(runtime.training_python), "-c", package_script], text=True))
    return {"commit": actual, "source_sha256": digest(sources), "packages": versions}


def launch_worker(bundle, operation, request, log):
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src), str(bundle.runtime.factory_root / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env.update(WANDB_DISABLED="true", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    with Path(log).open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(bundle.runtime.training_python), "-m", "lf_tools.worker", operation, str(request)],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(_GPU_LOCK_FD,) if _GPU_LOCK_FD is not None else (),
        )
        try:
            code = process.wait()
        except BaseException:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"{operation} failed (exit {code}); see {log}")
