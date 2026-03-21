from agent.application.agent_center.keys import conversation_key, scope_signature, session_key
from agent.application.agent_center.memory import persist_turn_memory
from agent.application.agent_center.prompting import with_language_hint


def test_agent_center_keys():
    assert session_key("p1", "s1", "leader") == "leader:p1:s1"
    assert conversation_key("p1", "s1") == "p1:s1"
    assert scope_signature([{"uid": "b"}, {"uid": "a"}, {"uid": ""}]) == "a,b"


def test_with_language_hint():
    assert "English" in with_language_hint("hello", lambda _x: "en")
    assert "中文回答" in with_language_hint("你好", lambda _x: "zh")
    assert with_language_hint("hola", lambda _x: "es") == "hola"


def test_persist_turn_memory(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "agent.application.agent_center.memory.classify_turn_memory_type",
        lambda _q, _a: "semantic",
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.ttl_for_memory_type",
        lambda _t: "2099-01-01T00:00:00",
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.upsert_project_memory_item",
        lambda **kwargs: captured.update(kwargs),
    )
    persist_turn_memory(
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        prompt="question",
        answer="answer",
    )
    assert captured["uuid"] == "u1"
    assert captured["project_uid"] == "p1"
    assert captured["session_uid"] == "s1"
