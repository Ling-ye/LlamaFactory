# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Frozen evaluations, actual model histories, independent rule scores and blind review."""

import copy
import html
import math
import random
import re
from pathlib import Path

from .artifacts import (
    digest,
    factory_receipt,
    gpu_lock,
    launch_worker,
    new_directory,
    now,
    read_json,
    read_rows,
    seal,
    verify,
    write_json,
    write_rows,
)
from .data import normalized, overlap_report
from .models import identify, identify_adapter, model_path
from .tracking import log_run


def library_lines(path):
    raw = read_json(path)
    values = []

    def visit(value):
        if isinstance(value, str):
            values.extend(value.splitlines())
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if "full_text" in value:
                visit(value["full_text"])
            elif "text" in value:
                visit(value["text"])
            else:
                for key in ("songs", "lines", "lyrics", "entries"):
                    if key in value:
                        visit(value[key])

    visit(raw)
    result = sorted({normalized(v) for v in values if normalized(v)})
    if not result:
        raise ValueError(f"No lines found in rule library: {path}")
    return result


def expanded_character(character):
    if character is None:
        return None
    result = character.model_dump(mode="json")
    for rule in result["rules"]:
        if rule["library"]:
            rule["values"] = library_lines(rule["library"])
            rule["library"] = None
    return result


def validate_cases(cases):
    if not cases:
        raise ValueError("Empty evaluation cases")
    ids, signatures = set(), set()
    for case in cases:
        if not isinstance(case.get("id"), str) or case["id"] in ids:
            raise ValueError("Missing or duplicate evaluation case ID")
        users = case.get("users")
        if not isinstance(users, list) or not users or any(not isinstance(u, str) or not u.strip() for u in users):
            raise ValueError("Cases must contain a nonempty list of user turns")
        signature = digest(users)
        if signature in signatures:
            raise ValueError("Duplicate evaluation scenario")
        ids.add(case["id"])
        signatures.add(signature)
        if case.get("knowledge_status", "not_applicable") not in {
            "known_fact_new_context",
            "unseen_fact",
            "not_applicable",
            "unverified_exposure",
        }:
            raise ValueError("Unknown knowledge exposure classification")


def check_case_overlap(cases, dataset, threshold=0.9):
    rows = [r for r in read_rows(Path(dataset) / "records.jsonl") if r["split"] == "train"]
    retired = set(read_json(Path(dataset) / "retired-cases.json"))
    cases_rows = []
    for case in cases:
        if case["id"] in retired:
            raise ValueError(f"Evaluation case retired into training: {case['id']}")
        cases_rows.append(
            {
                "id": "eval:" + case["id"],
                "split": "test",
                "group_id": case.get("group_id", "case:" + case["id"]),
                "messages": [{"role": "user", "content": u} for u in case["users"]],
            }
        )
    overlap_report(rows + cases_rows, threshold)


def freeze(bundle, output, dataset=None):
    cases = read_rows(bundle.evaluation.cases)
    validate_cases(cases)
    if bundle.evaluation.expected_cases and len(cases) != bundle.evaluation.expected_cases:
        raise ValueError("Evaluation count does not match the protocol")
    if dataset:
        verify(dataset)
        check_case_overlap(cases, dataset)
    elif bundle.evaluation.purpose == "heldout":
        raise ValueError("Heldout protocol requires --dataset for leakage checks")
    character = expanded_character(bundle.character)
    if bundle.evaluation.system_mode != "empty" and not character:
        raise ValueError("short/full system mode requires a character")
    system = "" if bundle.evaluation.system_mode == "empty" else character[bundle.evaluation.system_mode + "_prompt"]
    output = new_directory(output)
    write_rows(output / "cases.jsonl", cases)
    protocol = {
        "evaluation": bundle.evaluation.model_dump(mode="json"),
        "character": character,
        "system": system,
        "frozen_at": now(),
        "cases_sha256": digest(cases),
        "dataset_manifest": digest(verify(dataset)) if dataset else None,
    }
    write_json(output / "protocol.json", protocol)
    seal(output, {"kind": "evaluation_protocol"})
    return output


def execute_cases(request, generate):
    """Persist each generated turn. Resuming never substitutes reference answers."""
    output = Path(request["output"])
    cases = request["cases"]
    previous = (
        read_json(output / "progress.json")
        if (output / "progress.json").exists()
        else {"fingerprint": request["fingerprint"], "rows": []}
    )
    if previous["fingerprint"] != request["fingerprint"]:
        raise ValueError("Resume protocol/model/decoding changed")
    saved = previous["rows"]
    cursor, rows = 0, []
    for case in cases:
        messages = []
        for turn, user in enumerate(case["users"]):
            messages.append({"role": "user", "content": user})
            seed = (request["decoding_seed"] + int(digest([case["id"], turn])[:8], 16)) % (2**32)
            identity = {
                "case_id": case["id"],
                "group_id": case.get("group_id", case["id"]),
                "turn": turn,
                "messages": copy.deepcopy(messages),
                "seed": seed,
            }
            if cursor < len(saved):
                row = saved[cursor]
                checksum = row.get("row_sha256")
                if checksum != digest({k: v for k, v in row.items() if k != "row_sha256"}):
                    raise ValueError("Saved response was modified")
                if any(row.get(k) != v for k, v in identity.items()):
                    raise ValueError("Saved response has mismatched case/history")
            else:
                answer = generate(copy.deepcopy(messages), seed)
                row = {**identity, **answer, "category": case.get("category", "unspecified")}
                row["row_sha256"] = digest(row)
            rows.append(row)
            messages.append({"role": "assistant", "content": row["text"]})
            cursor += 1
            write_json(output / "progress.json", {"fingerprint": request["fingerprint"], "rows": rows})
    if len(saved) > len(rows):
        raise ValueError("Saved progress contains unexpected extra turns")
    write_rows(output / "responses.jsonl", rows)


def rule_result(rule, text, user):
    if rule.get("trigger") and not re.search(rule["trigger"], user):
        return None
    kind = rule["kind"]
    if kind == "contains_any":
        return any(v in text for v in rule["values"])
    if kind == "forbid_regex":
        return re.search(rule["pattern"], text) is None
    if kind == "suffix_regex":
        return re.search(rule["pattern"], text.rstrip()) is not None
    if kind == "max_chars":
        return len(text) <= rule["limit"]
    if kind == "exact_suffix_library":
        match = re.search(rule.get("pattern") or r"♪\s*([^\n]+)\s*$", text)
        return bool(match and normalized(match.group(1)) in rule["values"])
    raise ValueError(f"Unsupported rule: {kind}")


def wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0, center - width), min(1, center + width)]


def score_rows(rows, character):
    rules = character["rules"] if character else []
    details = []
    for row in rows:
        detail = {
            "case_id": row["case_id"],
            "group_id": row.get("group_id", row["case_id"]),
            "turn": row["turn"],
            "rules": {r["id"]: rule_result(r, row["text"], row["messages"][-1]["content"]) for r in rules},
            "empty": not row["text"].strip(),
            "truncated": row.get("finish_reason") == "length",
            "replacement_char": "\ufffd" in row["text"],
            "repeated": bool(re.search(r"(.{8,}?)\1\1", row["text"], flags=re.S)),
        }
        details.append(detail)
    metrics = {}
    for rule in rules:
        values = [d["rules"][rule["id"]] for d in details if d["rules"][rule["id"]] is not None]
        metrics[rule["id"]] = {
            "passed": sum(values),
            "applicable": len(values),
            "rate": sum(values) / len(values) if values else None,
            "wilson95_descriptive_turns": wilson(sum(values), len(values)),
            "threshold": rule["threshold"],
            "target_met": sum(values) / len(values) >= rule["threshold"] if values else None,
        }
    return {
        "rules": metrics,
        "turns": len(rows),
        "cases": len({r["case_id"] for r in rows}),
        "empty": sum(d["empty"] for d in details),
        "truncated": sum(d["truncated"] for d in details),
        "repeated": sum(d["repeated"] for d in details),
        "details": details,
        "semantic_identity": "pending_independent_review",
        "persona_similarity": "pending_independent_review",
        "facts": "pending_independent_review",
        "naturalness": "pending_independent_review",
        "release_accepted": False,
    }


def evaluate(bundle, protocol_path, output, adapter=None, resume=False):
    protocol_path, output = Path(protocol_path).absolute(), Path(output).absolute()
    verify(protocol_path)
    frozen = read_json(protocol_path / "protocol.json")
    cases = read_rows(protocol_path / "cases.jsonl")
    if digest(cases) != frozen["cases_sha256"]:
        raise ValueError("Frozen case hash mismatch")
    base = identify(model_path(bundle.model))
    selected = identify_adapter(adapter, base) if adapter else None
    if adapter:
        adapter_path = Path(adapter).absolute()
        run_root = adapter_path.parent
        if adapter_path.name.startswith("checkpoint-"):
            run_root = adapter_path.parent.parent
        binding = run_root / "binding.json"
        if frozen["evaluation"]["purpose"] == "heldout" and not binding.exists():
            raise ValueError(
                "Heldout adapter evaluation requires original training receipts; use development for legacy baselines"
            )
        if binding.exists():
            training = read_json(binding)
            if frozen["evaluation"]["purpose"] == "heldout":
                run_manifest = verify(run_root)
                if run_manifest.get("kind") != "training":
                    raise ValueError("Heldout candidates must be completed training runs")
            dataset = training["configuration"]["experiment"]["workspace"]
            # Dataset path is recorded independently in the original training request.
            request_path = run_root / "request.json"
            if request_path.exists():
                dataset = read_json(request_path)["dataset"]
                check_case_overlap(cases, dataset)
                if frozen["evaluation"]["purpose"] == "heldout":
                    # Protocol must predate the training directory creation receipt.
                    first = read_json(run_root / "started.json") if (run_root / "started.json").exists() else None
                    if not first or frozen["frozen_at"] > first["at"]:
                        raise ValueError("Heldout protocol was not frozen before this training run")
    ev = frozen["evaluation"]
    decoding = {
        "max_new_tokens": ev["max_new_tokens"],
        "temperature": ev["temperature"],
        "top_p": ev["top_p"],
        "top_k": 20,
        "do_sample": ev["temperature"] > 0,
    }
    from .artifacts import tree_hashes

    binding = {
        "orchestrator_sha256": digest(tree_hashes(Path(__file__).parent, {".py"})),
        "base": base,
        "adapter": selected,
        "model": bundle.model.model_dump(mode="json"),
        "protocol_sha256": digest(frozen),
        "factory": factory_receipt(bundle.runtime),
    }
    fingerprint = digest(binding)
    if resume:
        if read_json(output / "binding.json") != binding or (output / "manifest.json").exists():
            raise ValueError("Cannot resume changed or completed evaluation")
    else:
        new_directory(output)
        write_json(output / "binding.json", binding)
    write_json(output / "protocol.json", frozen)
    request = {
        "base": base,
        "adapter": selected,
        "model": binding["model"],
        "output": str(output),
        "cases": cases,
        "system": frozen["system"],
        "protocol": ev,
        "decoding": decoding,
        "decoding_seed": ev["seed"],
        "fingerprint": fingerprint,
    }
    write_json(output / "request.json", request)
    write_json(output / "status.json", {"status": "running", "at": now()})
    try:
        with gpu_lock(bundle.runtime.gpu_lock):
            launch_worker(bundle, "infer", output / "request.json", output / "inference.log")
        rows = read_rows(output / "responses.jsonl")
        score = score_rows(rows, frozen["character"])
        write_json(output / "scores.json", score)
        write_json(output / "status.json", {"status": "completed", "at": now()})
        seal(output, {"kind": "evaluation", "binding_sha256": fingerprint})
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "error": str(exc), "at": now()})
        raise
    log_run(bundle, output, "evaluation", {k: v for k, v in score.items() if isinstance(v, (int, float))})
    return output


def cluster_delta(left, right, key, seed=42, repeats=2000):
    groups = sorted({d.get("group_id", d["case_id"]) for d in left})

    def group_counts(details):
        return {
            g: [
                d["rules"][key]
                for d in details
                if d.get("group_id", d["case_id"]) == g and d["rules"][key] is not None
            ]
            for g in groups
        }

    a, b = group_counts(left), group_counts(right)

    def delta(sample):
        av, bv = [x for g in sample for x in a[g]], [x for g in sample for x in b[g]]
        return sum(bv) / len(bv) - sum(av) / len(av) if av and bv else None

    point = delta(groups)
    if point is None:
        return {"right_minus_left": None, "cluster_bootstrap95": None}
    rng = random.Random(seed)
    samples = sorted(v for _ in range(repeats) if (v := delta(rng.choices(groups, k=len(groups)))) is not None)
    return {
        "right_minus_left": point,
        "cluster_bootstrap95": [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]],
        "clusters": len(groups),
    }


def compare(left, right, output, allow_system_difference=False):
    left, right = Path(left), Path(right)
    verify(left)
    verify(right)
    a, b = read_json(left / "protocol.json"), read_json(right / "protocol.json")
    pa, pb = copy.deepcopy(a), copy.deepcopy(b)
    for obj in (pa, pb):
        obj.pop("frozen_at", None)
        obj["evaluation"].pop("name", None)
        obj["evaluation"].pop("cases", None)
        if allow_system_difference:
            obj.pop("system", None)
            obj["evaluation"].pop("system_mode", None)
    if pa != pb:
        raise ValueError("Comparisons require the same cases, decoding, rules and protocol")
    la, lb = read_rows(left / "responses.jsonl"), read_rows(right / "responses.jsonl")
    if [(x["case_id"], x["turn"]) for x in la] != [(x["case_id"], x["turn"]) for x in lb]:
        raise ValueError("Comparison has missing/different turns")
    output = new_directory(output)
    rows, mapping = [], {}
    masks = (a["character"] or {}).get("mask_patterns", [])
    rng = random.Random(42)
    for case_id in dict.fromkeys(r["case_id"] for r in la):
        switched = bool(rng.getrandbits(1))
        pair = [[r for r in response if r["case_id"] == case_id] for response in (la, lb)]
        if switched:
            pair.reverse()
        row = {
            "case_id": case_id,
            "users": [r["messages"][-1]["content"] for r in pair[0]],
            "A": [r["text"] for r in pair[0]],
            "B": [r["text"] for r in pair[1]],
        }
        for label in ("A", "B"):
            masked = row[label]
            for pattern in masks:
                masked = [re.sub(pattern, "", text) for text in masked]
            row[label + "_masked"] = masked
        row["case_sha256"] = digest(row)
        rows.append(row)
        mapping[case_id] = {"A": "right" if switched else "left", "B": "left" if switched else "right"}
    write_rows(output / "blind.jsonl", rows)
    write_json(output / "private" / "mapping.json", mapping)
    sa, sb = read_json(left / "scores.json"), read_json(right / "scores.json")
    rule_deltas = {key: cluster_delta(sa["details"], sb["details"], key) for key in sa["rules"]}
    report = {
        "left": str(left.absolute()),
        "right": str(right.absolute()),
        "rules": rule_deltas,
        "system_difference": allow_system_difference,
        "semantic_review": "pending",
        "release_accepted": False,
    }
    write_json(output / "comparison.json", report)
    blocks = [
        '<!doctype html><meta charset="utf-8"><title>匿名对话评审</title>',
        "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto}pre{white-space:pre-wrap}"
        "td{vertical-align:top;width:50%;padding:16px;border:1px solid #ddd}</style>",
        "<h1>匿名对话评审</h1><p>先看遮蔽版本判断人物感，再看原回答判断规则、事实和自然度。</p>",
    ]
    for row in rows:
        blocks.append(
            "<h2>"
            + html.escape(row["case_id"])
            + "</h2><p>"
            + html.escape(" / ".join(row["users"]))
            + "</p><table><tr>"
        )
        for label in ("A", "B"):
            blocks.append(
                "<td><h3>"
                + label
                + "</h3><pre>"
                + html.escape("\n\n".join(row[label]))
                + "</pre><details><summary>遮蔽版本</summary><pre>"
                + html.escape("\n\n".join(row[label + "_masked"]))
                + "</pre></details></td>"
            )
        blocks.append("</tr></table>")
    (output / "review.html").write_text("\n".join(blocks), encoding="utf-8")
    seal(output, {"kind": "blind_comparison"})
    return output


def import_reviews(comparison, reviews_path, output):
    comparison = Path(comparison)
    verify(comparison)
    cases = {r["case_id"]: r for r in read_rows(comparison / "blind.jsonl")}
    mapping = read_json(comparison / "private" / "mapping.json")
    reviews = read_rows(reviews_path)
    seen, results = set(), []
    for r in reviews:
        if r["case_id"] in seen or r["case_id"] not in cases:
            raise ValueError("Duplicate or unknown review case")
        seen.add(r["case_id"])
        if r.get("case_sha256") != cases[r["case_id"]]["case_sha256"]:
            raise ValueError("Review does not bind the actual anonymous answers")
        if r.get("winner") not in {"A", "B", "tie"} or r.get("reviewer_type") not in {"human", "model"}:
            raise ValueError("Invalid winner or reviewer_type")
        if not r.get("reviewer") or not r.get("reason"):
            raise ValueError("Reviewer and reasoning are required")
        for label in ("A", "B"):
            for dimension in ("identity", "style", "facts", "naturalness"):
                value = r.get(label, {}).get(dimension)
                if dimension == "facts" and dimension in r.get(label, {}) and value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
                    raise ValueError("Each answer needs independent 1–5 scores; facts may be null without evidence")
        winner = "tie" if r["winner"] == "tie" else mapping[r["case_id"]][r["winner"]]
        results.append({**r, "unblinded_winner": winner})
    if not results:
        raise ValueError("No reviews supplied")
    output = new_directory(output)
    write_rows(output / "reviews.jsonl", results)
    wins = sum(r["unblinded_winner"] == "right" for r in results)
    ties = sum(r["unblinded_winner"] == "tie" for r in results)
    write_json(
        output / "summary.json",
        {
            "reviewed": len(results),
            "total": len(cases),
            "right_win_rate": wins / len(results),
            "tie_rate": ties / len(results),
            "right_win_wilson95": wilson(wins, len(results)),
            "human_reviewed": sum(r["reviewer_type"] == "human" for r in results),
            "complete": len(results) == len(cases),
            "release_accepted": False,
        },
    )
    seal(output, {"kind": "semantic_reviews", "comparison_sha256": digest(verify(comparison))})
    return output
