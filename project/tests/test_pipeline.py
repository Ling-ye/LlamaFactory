# Copyright 2026. Licensed under the Apache License, Version 2.0.

import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lf_tools import data
from lf_tools.artifacts import digest, read_json, read_rows, seal, verify, write_json, write_rows
from lf_tools.config import Character, Data, Model, Rule, Source, read_config
from lf_tools.evaluation import (
    check_case_overlap,
    cluster_delta,
    compare,
    execute_cases,
    import_reviews,
    rule_result,
    score_rows,
    validate_cases,
)
from lf_tools.models import identify, identify_adapter
from lf_tools.native import check_native


def message(text, answer="回答"):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]


def row(id, split, text, **extra):
    return {"id": id, "split": split, "messages": message(text), **extra}


@pytest.fixture
def character():
    return Character(
        id="demo",
        identity="原创",
        short_prompt="你是星灯",
        full_prompt="完整角色卡",
        rules=[Rule(id="name", kind="contains_any", values=["星灯"])],
    )


def bundle(source, character=None, **kwargs):
    return SimpleNamespace(data=Data(name="test", sources=[Source(path=source)], **kwargs), character=character)


def test_message_validation():
    data.check_messages(message("问题"))
    data.check_messages([{"role": "system", "content": ""}, *message("问题")])
    data.check_messages([{"role": "user", "content": "问题"}], "user")


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "A"}],
        [{"role": "user", "content": " "}, {"role": "assistant", "content": "A"}],
        [{"role": "user", "content": "Q"}, {"role": "system", "content": "S"}],
        [{"role": "user", "content": "Q", "weight": 1}, {"role": "assistant", "content": "A"}],
    ],
)
def test_reject_malformed_messages(messages):
    with pytest.raises(ValueError):
        data.check_messages(messages)


def test_build_preserves_error_history_and_mask(tmp_path, character):
    source = tmp_path / "source.jsonl"
    original = row("a", "train", "第一问题", supervision="last_assistant")
    original["messages"] += message("请修正此前回答", "正确答案")
    write_rows(source, [original, row("b", "validation", "完全不同的开发场景")])
    output = tmp_path / "release"
    data.build(bundle(source, character, empty_system_ratio=1), output)
    records = read_rows(output / "records.jsonl")
    assert records[0]["messages"] == original["messages"]
    assert records[0]["supervision"] == "last_assistant"
    assert read_rows(output / "provenance.jsonl")[0]["record"] == original
    assert data.validate(output)["passed"]


def test_exact_system_mix_and_provenance(tmp_path, character):
    source = tmp_path / "source.jsonl"
    write_rows(source, [row(str(i), "train", f"主题{i}") for i in range(10)])
    output = tmp_path / "release"
    result = data.build(bundle(source, character, empty_system_ratio=0.9), output)
    assert result["system_modes"] == {"empty": 9, "short": 1}
    assert all("system_mode" not in r["record"] for r in read_rows(output / "provenance.jsonl"))


def test_detect_overlap_despite_different_system_and_answers(tmp_path):
    source = tmp_path / "input.jsonl"
    a = row("a", "train", "这个问题不能同时出现在两个集合中", group_id="one")
    b = row("b", "test", "这个问题不能同时出现在两个集合中", group_id="two")
    b["messages"].insert(0, {"role": "system", "content": "different"})
    b["messages"][-1]["content"] = "different"
    write_rows(source, [a, b])
    with pytest.raises(ValueError, match="crosses splits"):
        data.build(bundle(source), tmp_path / "release")


def test_source_groups_cannot_be_reshuffled():
    rows = [
        row("a", "train", "第一种问法", group_id="same"),
        row("b", "validation", "截然不同的表述", group_id="same"),
    ]
    with pytest.raises(ValueError, match="group crosses"):
        data.overlap_report(rows)


def test_near_duplicates_detected():
    text = "请解释在只有十五分钟的情况下如何规划一个完整的学习任务并保留休息时间"
    rows = [row("a", "train", text, group_id="a"), row("b", "test", text + "呢", group_id="b")]
    with pytest.raises(ValueError, match="Near-duplicate"):
        data.overlap_report(rows)


def test_reject_rejected_and_conflicting_splits(tmp_path):
    source = Source(path=tmp_path / "unused", split="train")
    with pytest.raises(ValueError, match="conflicts"):
        data.normalize_record(row("a", "test", "Q"), source, 0)
    with pytest.raises(ValueError, match="Rejected"):
        data.normalize_record(row("a", "train", "Q", review={"accepted": False}), source, 0)


def test_incomplete_dataset_reports_gap(tmp_path):
    source = tmp_path / "input.jsonl"
    write_rows(source, [row("a", "train", "Q")])
    result = data.build(bundle(source, targets={"train": 3}), tmp_path / "release")
    assert result["gaps"] == {"train": 2}
    assert result["complete"] is False


def test_immutable_and_tamper_detection(tmp_path):
    source = tmp_path / "input.jsonl"
    write_rows(source, [row("a", "train", "Q")])
    output = tmp_path / "release"
    data.build(bundle(source), output)
    with pytest.raises(FileExistsError):
        data.build(bundle(source), output)
    (output / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Artifact changed"):
        verify(output)


def test_alpaca_with_history(tmp_path):
    source = Source(path=tmp_path / "unused", format="alpaca", split="train")
    raw = {"instruction": "新问题", "input": "补充信息", "output": "回复", "history": [["旧问题", "旧回答"]]}
    result = data.normalize_record(raw, source, 0)
    assert len(result["messages"]) == 4
    assert result["messages"][-2]["content"] == "新问题\n补充信息"


def preference(source):
    prompt = [{"role": "user", "content": "请回应我的困惑"}]
    return {
        "id": "p",
        "split": "train",
        "prompt": prompt,
        "chosen": "这是一条好回答",
        "rejected": "这是一条差回答",
        "prompt_sha256": digest(prompt),
        "review": {"accepted": True, "reason": "表达和事实独立核对"},
        "parent_adapter_sha256": "a" * 64,
    }


def test_preferences_same_prompt_and_review(tmp_path):
    source = Source(path=tmp_path / "unused", format="preferences")
    raw = preference(source)
    assert data.normalize_record(raw, source, 0)["prompt"] == raw["prompt"]
    raw["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="prompt hash"):
        data.normalize_record(raw, source, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"chosen": "这是一条差回答"},
        {"chosen": "很" * 100},
        {"review": {"accepted": True}},
        {"chosen": [{"role": "user", "content": "bad"}]},
    ],
)
def test_bad_preferences_rejected(tmp_path, change):
    source = Source(path=tmp_path / "unused", format="preferences")
    raw = preference(source)
    raw.update(change)
    with pytest.raises(ValueError):
        data.normalize_record(raw, source, 0)


def test_dpo_fixed_reference(tmp_path):
    cfg = {
        "stage": "dpo",
        "model_name_or_path": "/models/base",
        "adapter_name_or_path": "/selected-sft",
        "ref_model": "/models/base",
        "ref_model_adapters": "/selected-sft",
        "quantization_bit": 4,
        "ref_model_quantization_bit": 4,
        "quantization_method": "bnb",
    }
    check_native(cfg)
    cfg["ref_model_adapters"] = "/wrong-parent"
    with pytest.raises(ValueError, match="reference"):
        check_native(cfg)


@pytest.mark.parametrize("kwargs", [{"repo_id": "Qwen/x", "revision": "main"}, {"repo_id": "Qwen/x"}, {}])
def test_download_requires_pin(kwargs):
    with pytest.raises(ValidationError):
        Model(name="x", template="qwen", **kwargs)


def test_paths_relative_to_each_yaml(tmp_path):
    cfg = tmp_path / "model.yaml"
    cfg.write_text("name: x\nlocal_path: weights\ntemplate: qwen\n")
    assert read_config(cfg, Model).local_path == tmp_path / "weights"


def test_adapter_base_binding(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    write_json(base / "config.json", {})
    (base / "model.safetensors").write_bytes(b"model")
    fingerprint = identify(base)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": str(base)})
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    identify_adapter(adapter, fingerprint)
    write_json(adapter / "posttrain-binding.json", {"base_sha256": "wrong"})
    with pytest.raises(ValueError, match="different base"):
        identify_adapter(adapter, fingerprint)


def inference_request(tmp_path):
    return {
        "output": str(tmp_path),
        "fingerprint": "fixed",
        "decoding_seed": 42,
        "cases": [{"id": "a", "users": ["问题一", "追问"]}],
    }


def test_multiturn_uses_model_answer_and_resume(tmp_path):
    request = inference_request(tmp_path)
    calls = []

    def generate(messages, seed):
        calls.append(messages)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return {"text": "模型自己的第一轮回答", "finish_reason": "stop"}

    with pytest.raises(RuntimeError):
        execute_cases(request, generate)
    assert len(read_json(tmp_path / "progress.json")["rows"]) == 1
    resumed = []

    def finish(messages, seed):
        resumed.append(messages)
        return {"text": "第二轮真实回答", "finish_reason": "stop"}

    execute_cases(request, finish)
    assert len(resumed) == 1
    assert resumed[0][1] == {"role": "assistant", "content": "模型自己的第一轮回答"}
    assert len(read_rows(tmp_path / "responses.jsonl")) == 2


def test_tampered_resume_rejected(tmp_path):
    request = inference_request(tmp_path)
    execute_cases(request, lambda messages, seed: {"text": "answer"})
    progress = read_json(tmp_path / "progress.json")
    progress["rows"][0]["text"] = "tampered"
    write_json(tmp_path / "progress.json", progress)
    with pytest.raises(ValueError, match="modified"):
        execute_cases(request, lambda *args: pytest.fail("must not generate"))


def test_changed_resume_protocol_rejected(tmp_path):
    request = inference_request(tmp_path)
    execute_cases(request, lambda messages, seed: {"text": "answer"})
    request["fingerprint"] = "different"
    with pytest.raises(ValueError, match="changed"):
        execute_cases(request, lambda *args: pytest.fail("must not generate"))


@pytest.mark.parametrize(
    "rule,text,user,expected",
    [
        ({"kind": "contains_any", "values": ["星灯"]}, "星灯在这里", "Q", True),
        ({"kind": "contains_any", "values": ["星灯"]}, "别的人", "Q", False),
        ({"kind": "suffix_regex", "pattern": "灯还亮着。$"}, "灯还亮着。\n", "Q", True),
        ({"kind": "max_chars", "limit": 2}, "三个字", "Q", False),
        ({"kind": "forbid_regex", "pattern": "换角色"}, "不换角色", "Q", False),
        ({"kind": "contains_any", "values": ["星灯"], "trigger": "名字"}, "没有", "天气", None),
        (
            {"kind": "exact_suffix_library", "values": [data.normalized("真っ新な画を描いて")]},
            "你好\n♪ 真っ新な画を描いて",
            "Q",
            True,
        ),
    ],
)
def test_independent_rules(rule, text, user, expected):
    assert rule_result(rule, text, user) is expected


def test_rules_do_not_claim_semantic_quality(character):
    rows = [{"case_id": "a", "turn": 0, "text": "星灯星灯星灯", "messages": message("Q")[:1]}]
    score = score_rows(rows, character.model_dump(mode="json"))
    assert score["rules"]["name"]["rate"] == 1
    assert score["persona_similarity"] == "pending_independent_review"
    assert score["release_accepted"] is False


def test_no_applicable_rule_is_not_success():
    char = {"rules": [{"id": "trigger", "kind": "max_chars", "limit": 1, "trigger": "点歌", "threshold": 0.95}]}
    rows = [{"case_id": "a", "turn": 0, "text": "长回答", "messages": message("天气")[:1]}]
    assert score_rows(rows, char)["rules"]["trigger"]["target_met"] is None


def test_cluster_bootstrap_paired_by_case():
    left = [
        {"case_id": "a", "rules": {"rule": False}},
        {"case_id": "a", "rules": {"rule": False}},
        {"case_id": "b", "rules": {"rule": True}},
    ]
    right = copy.deepcopy(left)
    right[0]["rules"]["rule"] = True
    result = cluster_delta(left, right, "rule", repeats=100)
    assert result["right_minus_left"] == pytest.approx(1 / 3)
    assert result["clusters"] == 2


def test_retired_test_case_cannot_return(tmp_path):
    write_rows(tmp_path / "records.jsonl", [row("a", "train", "新主题", group_id="a")])
    write_json(tmp_path / "retired-cases.json", ["old-case"])
    with pytest.raises(ValueError, match="retired"):
        check_case_overlap([{"id": "old-case", "users": ["其他话题"]}], tmp_path)


def test_bad_case_bank_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases([{"id": "x", "users": ["a"]}, {"id": "x", "users": ["b"]}])


def make_eval(path, text, character, temperature=0):
    path.mkdir()
    protocol = {
        "evaluation": {"name": "x", "cases": "unused", "system_mode": "empty", "temperature": temperature, "seed": 42},
        "character": character.model_dump(mode="json"),
        "system": "",
        "cases_sha256": "fixed",
    }
    rows = [{"case_id": "a", "turn": 0, "text": text, "messages": [{"role": "user", "content": "问题"}]}]
    write_json(path / "protocol.json", protocol)
    write_rows(path / "responses.jsonl", rows)
    write_json(path / "scores.json", score_rows(rows, protocol["character"]))
    seal(path, {"kind": "evaluation"})


def test_blind_comparison_and_bound_reviews(tmp_path, character):
    left, right = tmp_path / "left", tmp_path / "right"
    make_eval(left, "普通回复", character)
    make_eval(right, "星灯的回复", character)
    output = compare(left, right, tmp_path / "comparison")
    cases = read_rows(output / "blind.jsonl")
    assert "left" not in cases[0] and "right" not in cases[0]
    reviews = tmp_path / "reviews.jsonl"
    write_rows(
        reviews,
        [
            {
                "case_id": "a",
                "case_sha256": cases[0]["case_sha256"],
                "winner": "B",
                "reviewer_type": "human",
                "reviewer": "test",
                "reason": "独立评审",
                "A": {"identity": 3, "style": 3, "facts": 4, "naturalness": 4},
                "B": {"identity": 4, "style": 4, "facts": 4, "naturalness": 4},
            }
        ],
    )
    import_reviews(output, reviews, tmp_path / "scored")
    assert read_json(tmp_path / "scored" / "summary.json")["complete"]
    cases[0]["case_sha256"] = "wrong"
    raw = read_rows(reviews)
    raw[0]["case_sha256"] = "wrong"
    write_rows(reviews, raw)
    with pytest.raises(ValueError, match="bind"):
        import_reviews(output, reviews, tmp_path / "bad")


def test_comparison_rejects_changed_decoding(tmp_path, character):
    make_eval(tmp_path / "left", "A", character, 0)
    make_eval(tmp_path / "right", "B", character, 0.7)
    with pytest.raises(ValueError, match="same cases"):
        compare(tmp_path / "left", tmp_path / "right", tmp_path / "out")
