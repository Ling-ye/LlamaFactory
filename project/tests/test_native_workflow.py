# Copyright 2026. Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace

import pytest

from lf_tools import native
from lf_tools.artifacts import digest, read_json, seal, write_json
from lf_tools.config import Source
from lf_tools.data import normalize_record


def prepared_pair(tmp_path):
    prepared, checked, run = (tmp_path / n for n in ("prepared", "checked", "run"))
    write_json(prepared / "request.json", {"config": {"output_dir": str(run / "adapter")}})
    write_json(prepared / "binding.json", {"base": "fixed"})
    preparation = seal(prepared, {"kind": "native_preparation"})
    write_json(checked / "preflight-result.json", {"within_memory_budget": True})
    seal(checked, {"kind": "preflight", "preparation_sha256": digest(preparation)})
    return prepared, checked, run


def test_receipt_records_official_cli_and_refuses_reuse(tmp_path):
    prepared, checked, run = prepared_pair(tmp_path)
    assert native.start_receipt(prepared, checked) == run
    assert read_json(run / "native-receipt.json")["official_command"][:2] == ["llamafactory-cli", "train"]
    assert read_json(run / "status.json")["status"] == "awaiting_official_cli"
    with pytest.raises(FileExistsError):
        native.start_receipt(prepared, checked)


def test_receipt_rejects_preflight_for_other_data(tmp_path):
    prepared, checked, _ = prepared_pair(tmp_path)
    seal(checked, {"kind": "preflight", "preparation_sha256": "other"})
    with pytest.raises(ValueError, match="another preparation"):
        native.start_receipt(prepared, checked)


def test_resume_requires_checkpoint_state_and_immutable_input(tmp_path):
    prepared, checked, run = prepared_pair(tmp_path)
    native.start_receipt(prepared, checked)
    checkpoint = run / "adapter" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete"):
        native.check_resume(run, checkpoint)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (checkpoint / name).write_text("{}")
    assert native.check_resume(run, checkpoint)["checkpoint"] == str(checkpoint)
    with pytest.raises(ValueError, match="another run"):
        native.check_resume(run, tmp_path / "other")
    write_json(prepared / "request.json", {"tampered": True})
    with pytest.raises(ValueError, match="Artifact changed"):
        native.check_resume(run, checkpoint)


def test_native_paths_resolve_from_factory_root(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model_name_or_path: ../models/base\noutput_dir: ../artifacts/run/adapter\n")
    cfg = native.read_native(path, tmp_path / "factory")
    assert cfg["model_name_or_path"] == str(tmp_path / "models/base")
    assert cfg["output_dir"] == str(tmp_path / "artifacts/run/adapter")


def test_failed_history_never_enters_targets(tmp_path):
    raw = {
        "split": "train",
        "failure_case": {"id": "failed"},
        "messages": [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "错误历史"},
            {"role": "user", "content": "请纠正"},
            {"role": "assistant", "content": "正确目标"},
        ],
    }
    with pytest.raises(ValueError, match="Failure histories"):
        normalize_record(raw, Source(path=tmp_path / "snapshot"), 0)
    raw["supervision"] = "last_assistant"
    result = normalize_record(raw, Source(path=tmp_path / "snapshot"), 0)
    assert result["messages"][1]["content"] == "错误历史"
    assert result["supervision"] == "last_assistant"


def test_bridge_excludes_heldout_and_preserves_supervision(tmp_path, monkeypatch):
    import json
    import sys

    from lf_tools import bridge

    records = [
        {
            "sample_id": "train",
            "split": "train",
            "supervision": "last_assistant",
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "修复问题"},
                {"role": "assistant", "content": "正确"},
            ],
        },
        {
            "sample_id": "heldout",
            "split": "test",
            "messages": [
                {"role": "user", "content": "封存场景"},
                {"role": "assistant", "content": "封存回答"},
            ],
        },
    ]
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in records))
    monkeypatch.setattr(bridge, "tokenizer_bundle", lambda r: (None, None, None))
    seen = []

    def encode(record, *args):
        seen.append(record["id"])
        return {"input_ids": [1, 2], "labels": [-100, 2]}

    monkeypatch.setattr(bridge, "encode_sft", encode)

    class FakeDict(dict):
        def save_to_disk(self, path):
            assert set(self) == {"train"}

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows), DatasetDict=FakeDict),
    )
    report = bridge.prepare(source, tmp_path / "model", tmp_path / "tokens")
    assert seen == ["train"]
    assert report["rows"][0]["supervised_message_indices"] == [2]
    with pytest.raises(ValueError, match="truncation"):
        bridge.prepare(source, tmp_path / "model", tmp_path / "short", cutoff_len=1)
