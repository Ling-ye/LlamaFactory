# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Compatibility bridge for PersonaLoom exports, using the single Factory encoder."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .artifacts import digest, file_hash, read_rows, write_json
from .data import normalize_record, overlap_report
from .worker import encode_sft, tokenizer_bundle


def prepare(records_path, model_path, output, cutoff_len=2048, template_name="qwen3_5"):
    from datasets import Dataset, DatasetDict

    output = Path(output)
    if output.exists():
        raise ValueError("Use a new tokenized output directory")
    source = SimpleNamespace(path=Path(records_path), format="messages", split=None, kind="synthetic")
    records = [normalize_record(raw, source, i) for i, raw in enumerate(read_rows(records_path))]
    overlap_report(records)
    bundle, args, template = tokenizer_bundle(
        {
            "base": {"path": str(model_path)},
            "model": {"template": template_name, "enable_thinking": False},
        }
    )
    splits, audits = {}, []
    for record in records:
        if record["split"] == "test":
            continue
        value = encode_sft(record, bundle, args, template)
        tokens = len(value["input_ids"])
        if tokens > cutoff_len:
            raise ValueError(f"Overlength {record['id']}: {tokens}; truncation is forbidden")
        targets = [i for i, m in enumerate(record["messages"]) if m["role"] == "assistant"]
        if record["supervision"] == "last_assistant":
            targets = targets[-1:]
        splits.setdefault(record["split"], []).append(value)
        audits.append(
            {
                "sample_id": record["id"],
                "split": record["split"],
                "tokens": tokens,
                "supervised_message_indices": targets,
                "encoded_sha256": digest(value),
            }
        )
    if not splits.get("train"):
        raise ValueError("No training records")
    DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}).save_to_disk(str(output))
    report = {
        "passed": True,
        "input_sha256": file_hash(records_path),
        "template": template_name,
        "thinking": False,
        "cutoff_len": cutoff_len,
        "counts": {k: len(v) for k, v in splits.items()},
        "rows": audits,
        "max_tokens": max(a["tokens"] for a in audits),
        "no_truncation": True,
        "heldout_excluded": True,
    }
    write_json(output.parent / (output.name + "-validation.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("records", "model", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--template", default="qwen3_5")
    args = parser.parse_args()
    result = prepare(args.records, args.model, args.output, args.cutoff_len, args.template)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
