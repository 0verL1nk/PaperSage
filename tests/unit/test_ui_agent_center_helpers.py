from types import SimpleNamespace

from agent.application.agent_center.keys import conversation_key, scope_signature, session_key
from agent.application.agent_center.memory import persist_turn_memory
from agent.application.agent_center.prompting import build_routing_context, with_language_hint


def test_agent_center_keys():
    assert session_key("p1", "s1", "leader") == "leader:p1:s1"
    assert conversation_key("p1", "s1") == "p1:s1"
    assert scope_signature([{"uid": "b"}, {"uid": "a"}, {"uid": ""}]) == "a,b"


def test_with_language_hint():
    assert "English" in with_language_hint("hello", lambda _x: "en")
    assert "中文回答" in with_language_hint("你好", lambda _x: "zh")
    assert with_language_hint("hola", lambda _x: "es") == "hola"


def test_build_routing_context(monkeypatch):
    monkeypatch.setattr(
        "agent.application.agent_center.prompting.load_agent_settings",
        lambda: SimpleNamespace(
            agent_routing_context_recent_limit=2,
            agent_routing_context_max_chars=500,
            agent_routing_context_item_max_chars=20,
            agent_routing_context_reason_max_chars=20,
            agent_routing_context_roles_preview_count=2,
        ),
    )
    context = build_routing_context(
        [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": "A1",
                "mindmap_html": "<html>very long html payload</html>",
                "policy_decision": {"plan_enabled": True, "team_enabled": False, "reason": "r"},
                "team_execution": {"enabled": False, "rounds": 0, "roles": []},
            },
        ],
        "summary",
    )
    assert "[会话压缩摘要]" in context
    assert "[上一轮执行策略]" in context
    assert "very long html payload" not in context


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
