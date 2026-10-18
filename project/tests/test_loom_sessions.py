import copy

import pytest

from lf_tools.evaluation import execute_cases, validate_cases


def case():
    from persona_data.loom.contracts import AgentContext, BehaviorPack
    pack = BehaviorPack(id="test", self_reference=None, lyric_policy="optional",
                        song_library_sha256=None, public_experience_each_turn=False,
                        min_traits_each_turn=0, distinctive_trait_each_turn=False, identity="ヰ世界情緒")
    return {"id": "case", "users": ["我先说旧信息", "我把信息更新了", "新会话，记得更新吗？"],
            "sessions": [
                {"start_turn": 0, "context": AgentContext().model_dump(mode="json"), "memory_from_turns": []},
                {"start_turn": 2, "context": AgentContext().model_dump(mode="json"), "memory_from_turns": [1]}],
            "behavior_pack": pack.model_dump(), "behavior_sha256": pack.fingerprint}


def test_cross_session_uses_actual_responses_and_resumes(tmp_path):
    c = case()
    validate_cases([c])
    calls = []
    request = {"output": str(tmp_path), "cases": [c], "fingerprint": "f", "decoding_seed": 42,
               "protocol": {"system_mode": "short"}}
    def generate(messages, seed, system_override=None):
        calls.append((copy.deepcopy(messages), system_override))
        return {"text": "实际回复" + str(len(calls))}
    execute_cases(request, generate)
    assert len(calls[2][0]) == 1
    assert "实际回复2" in calls[2][1]
    assert "实际回复1" not in calls[2][1]
    execute_cases(request, generate)
    assert len(calls) == 3


def test_future_memory_is_rejected():
    c = case()
    c["sessions"][1]["memory_from_turns"] = [2]
    with pytest.raises(ValueError, match="future"):
        validate_cases([c])


def test_worker_preserves_shared_renderer_import_path(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lf_tools import artifacts
    captured = {}
    class Process:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs["env"])
        def wait(self):
            return 0
    monkeypatch.setenv("PYTHONPATH", "/explicit/persona-loom/tool/src")
    monkeypatch.setattr(artifacts.subprocess, "Popen", Process)
    bundle = SimpleNamespace(runtime=SimpleNamespace(factory_root=tmp_path, training_python="python"))
    artifacts.launch_worker(bundle, "infer", tmp_path / "request.json", tmp_path / "log")
    assert captured["PYTHONPATH"].endswith("/explicit/persona-loom/tool/src")
