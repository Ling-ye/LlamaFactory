# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Offline snapshots, source preservation and split isolation."""

import copy
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .artifacts import digest, file_hash, new_directory, read_rows, seal, verify, write_json, write_rows
from .models import identify_adapter


def normalized(text):
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", text).lower())


def user_signature(messages):
    return "\n".join(normalized(m["content"]) for m in messages if m["role"] == "user")


def check_messages(messages, ends="assistant"):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Empty or invalid messages")
    body = messages[1:] if messages[0].get("role") == "system" else messages
    if not body:
        raise ValueError("Missing conversation")
    for i, message in enumerate(messages):
        if set(message) != {"role", "content"} or not isinstance(message["content"], str):
            raise ValueError("Text messages must contain only role/content")
        if not message["content"].strip() and message["role"] != "system":
            raise ValueError("Empty conversation message")
        if message["role"] == "system" and i != 0:
            raise ValueError("system is only allowed at the beginning")
    for i, message in enumerate(body):
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Conversations must alternate user and assistant")
    if body[-1]["role"] != ends:
        raise ValueError(f"Conversation must end with {ends}")


def _ngrams(text):
    return {text[i : i + 3] for i in range(max(1, len(text) - 2))}


def overlap_report(records, threshold=0.9):
    groups, ids, signatures, seen = {}, set(), {}, []
    overlaps = []
    for row in records:
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        ids.add(row["id"])
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Source/scenario group crosses splits: {group}")
        groups[group] = row["split"]
        sig = user_signature(row.get("messages", row.get("prompt")))
        if sig in signatures and signatures[sig]["split"] != row["split"]:
            raise ValueError(f"Identical user scenario crosses splits: {row['id']}")
        tokens = _ngrams(sig)
        for previous, other_sig, other_tokens in seen:
            if previous["split"] == row["split"] or len(sig) < 16 or len(other_sig) < 16:
                continue
            similarity = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
            if similarity >= threshold:
                raise ValueError(f"Near-duplicate scenario crosses splits: {previous['id']} / {row['id']}")
        if sig in signatures:
            overlaps.append({"left": signatures[sig]["id"], "right": row["id"], "kind": "same_split_context"})
        signatures[sig] = row
        seen.append((row, sig, tokens))
    return {"cross_split_overlap": 0, "within_split_variants": overlaps}


def normalize_record(raw, source, index):
    messages = copy.deepcopy(raw.get("messages"))
    if source.format == "alpaca":
        messages = []
        if raw.get("system"):
            messages.append({"role": "system", "content": raw["system"]})
        for user, assistant in raw.get("history", []):
            messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        text = raw.get("instruction", "")
        if raw.get("input"):
            text += "\n" + raw["input"]
        messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": raw["output"]}])
    split = source.split or raw.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Every sample needs an explicit split; existing groups are never randomly reshuffled")
    if source.split and raw.get("split") and source.split != raw["split"]:
        raise ValueError("Source split conflicts with record split")
    row = {
        "id": str(raw.get("id", raw.get("sample_id", raw.get("pair_id", digest(raw))))),
        "split": split,
        "kind": raw.get("kind", "synthetic" if raw.get("synthetic") else source.kind),
        "category": raw.get("category", raw.get("topic", "unspecified")),
        "supervision": raw.get("supervision", "all_assistant"),
        "evidence_ids": raw.get("evidence_ids", raw.get("claim_evidence_ids", [])),
        "review": raw.get("review", {}),
        "human_verified": raw.get("human_verified", raw.get("human_reviewed", False)),
        "origin": {"file": str(source.path), "row": index, "sha256": digest(raw)},
        "failure_case_ids": raw.get("failure_case_ids", raw.get("response_case_ids", [])),
    }
    if row["review"].get("accepted") is False:
        raise ValueError(f"Rejected sample cannot enter training: {row['id']}")
    if source.format == "preferences":
        prompt = copy.deepcopy(raw["prompt"])
        check_messages(prompt, "user")
        responses = {}
        for name in ("chosen", "rejected"):
            response = raw[name]
            if isinstance(response, list):
                if len(response) != 1 or response[0].get("role") != "assistant":
                    raise ValueError("chosen/rejected must each contain one assistant completion")
                response = response[0]["content"]
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Invalid preference completion")
            responses[name] = response
        if responses["chosen"] == responses["rejected"]:
            raise ValueError("Identical preference branches")
        if max(map(len, responses.values())) > 2 * min(map(len, responses.values())):
            raise ValueError("Preference length ratio exceeds 2:1")
        if not raw.get("review", {}).get("accepted") or not raw["review"].get("reason"):
            raise ValueError("Preferences require an accepted review with a reason")
        if raw.get("prompt_sha256") != digest(prompt):
            raise ValueError("Preference shared prompt hash mismatch")
        row.update(prompt=prompt, **responses, parent_adapter_sha256=raw.get("parent_adapter_sha256"))
        messages = prompt
    else:
        check_messages(messages)
        if row["supervision"] not in {"all_assistant", "last_assistant"}:
            raise ValueError("Unsupported supervision policy")
        if raw.get("failure_case") and row["supervision"] != "last_assistant":
            raise ValueError("Failure histories must never be supervised")
        row["messages"] = messages
    row["group_id"] = str(
        raw.get("group_id", raw.get("source_group_id", raw.get("scenario_family", digest(user_signature(messages)))))
    )
    return row


def build(bundle, output):
    records, originals, sources = [], [], []
    data = bundle.data
    for source in data.sources:
        sources.append(
            {
                "config": source.model_dump(mode="json"),
                "sha256": file_hash(source.path),
                "provenance_sha256": file_hash(source.provenance) if source.provenance else None,
            }
        )
        for index, raw in enumerate(read_rows(source.path)):
            row = normalize_record(raw, source, index)
            if (data.stage == "dpo") != ("prompt" in row):
                raise ValueError("Dataset stage and input format disagree")
            records.append(row)
            originals.append({"sample_id": row["id"], "record": raw})
    if data.max_records_per_split:
        counts = Counter()
        limited = []
        for record in records:
            limit = data.max_records_per_split.get(record["split"], 0)
            if counts[record["split"]] < limit:
                limited.append(record)
                counts[record["split"]] += 1
        records = limited
        ids = {r["id"] for r in records}
        originals = [r for r in originals if r["sample_id"] in ids]
    if not records:
        raise ValueError("No source records")
    audit = overlap_report(records, data.near_duplicate_threshold)
    parent = identify_adapter(data.parent_adapter) if data.parent_adapter else None
    if parent and any(r["parent_adapter_sha256"] != parent["sha256"] for r in records):
        raise ValueError("Preference data was not generated for this selected SFT parent")
    if data.empty_system_ratio is not None:
        for split in ("train", "validation", "test"):
            subset = sorted([r for r in records if r["split"] == split], key=lambda r: r["id"])
            random.Random(data.seed).shuffle(subset)
            empty_count = round(len(subset) * data.empty_system_ratio)
            for i, row in enumerate(subset):
                body = [m for m in row["messages"] if m["role"] != "system"]
                row["messages"] = (
                    body if i < empty_count else [{"role": "system", "content": bundle.character.short_prompt}, *body]
                )
                row["system_mode"] = "empty" if i < empty_count else "short"
    counts = Counter(r["split"] for r in records)
    gaps = {k: max(0, target - counts[k]) for k, target in data.targets.items()}
    output = new_directory(output)
    write_rows(output / "records.jsonl", records)
    write_rows(output / "provenance.jsonl", originals)
    for i, source in enumerate(data.sources):
        if source.provenance:
            import shutil

            shutil.copyfile(source.provenance, output / f"source-provenance-{i}.json")
    write_json(output / "sources.json", sources)
    write_json(output / "recipe.json", data.model_dump(mode="json"))
    write_json(output / "overlap.json", audit)
    write_json(
        output / "retired-cases.json",
        sorted({x for r in records if r["split"] == "train" for x in r["failure_case_ids"]}),
    )
    factory_info = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in records if r["split"] == split]
        if not subset:
            continue
        if data.stage == "sft":
            exported = [{"messages": r["messages"]} for r in subset]
            info = {
                "formatting": "sharegpt",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
        else:
            exported = [
                {
                    "messages": [m for m in r["prompt"] if m["role"] != "system"],
                    "system": next((m["content"] for m in r["prompt"] if m["role"] == "system"), ""),
                    "chosen": {"role": "assistant", "content": r["chosen"]},
                    "rejected": {"role": "assistant", "content": r["rejected"]},
                }
                for r in subset
            ]
            info = {
                "formatting": "sharegpt",
                "ranking": True,
                "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected", "system": "system"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        write_rows(output / f"{split}.jsonl", exported)
        factory_info[split] = {"file_name": f"{split}.jsonl", **info}
    write_json(output / "dataset_info.json", factory_info)
    return seal(
        output,
        {
            "kind": "dataset",
            "stage": data.stage,
            "counts": dict(counts),
            "gaps": gaps,
            "complete": not any(gaps.values()),
            "parent": parent,
            "requires_label_audit": True,
            "categories": dict(Counter(r["category"] for r in records if r["split"] == "train")),
            "system_modes": dict(Counter(r.get("system_mode", "preserved") for r in records)),
            "category_weight_policy": "report_only_for_imported_data",
            "requested_category_weights": data.category_weights,
            "semantic_quality": "imported_reviews_not_independently_reverified",
            "recipe_sha256": digest(data.model_dump(mode="json")),
        },
    )


def validate(directory):
    manifest = verify(directory)
    rows = read_rows(Path(directory) / "records.jsonl")
    for row in rows:
        check_messages(row.get("messages", row.get("prompt")), "user" if "prompt" in row else "assistant")
    from .artifacts import read_json

    recipe = read_json(Path(directory) / "recipe.json")
    audit = overlap_report(rows, recipe["near_duplicate_threshold"])
    if dict(Counter(r["split"] for r in rows)) != manifest["counts"]:
        raise ValueError("Manifest row counts disagree")
    return {"passed": True, "complete": manifest["complete"], "counts": manifest["counts"], **audit}
