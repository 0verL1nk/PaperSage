from types import SimpleNamespace

from agent.application.agent_center.keys import conversation_key, scope_signature, session_key
from agent.application.agent_center.memory import persist_turn_memory, query_turn_memory
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
    assert context.startswith("RTv1\n")
    assert "\nS:summary\n" in context
    assert "\nP:plan=True,team=False,reason=r\n" in context
    assert "\nT:enabled=False,rounds=0,roles=\n" in context
    assert "H1:user:Q1" in context
    assert "very long html payload" not in context


def test_build_routing_context_keeps_stable_prefix_under_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.application.agent_center.prompting.load_agent_settings",
        lambda: SimpleNamespace(
            agent_routing_context_recent_limit=6,
            agent_routing_context_max_chars=140,
            agent_routing_context_item_max_chars=60,
            agent_routing_context_reason_max_chars=40,
            agent_routing_context_roles_preview_count=2,
        ),
    )
    context = build_routing_context(
        [
            {
                "role": "assistant",
                "content": "A",
                "policy_decision": {
                    "plan_enabled": True,
                    "team_enabled": True,
                    "reason": "long reason",
                },
            },
            {"role": "user", "content": "Q1 " * 30},
            {"role": "assistant", "content": "A1 " * 30},
            {"role": "user", "content": "Q2 " * 30},
        ],
        "summary " * 30,
    )
    assert context.startswith("RTv1\nS:")
    assert "\nP:" in context
    assert "\nT:" in context
    assert len(context) <= 140


def test_persist_turn_memory(monkeypatch):
    captured = {}

    class _FakeAdapter:
        def add_resource(self, *, project_uid, content, metadata=None):
            captured["project_uid"] = project_uid
            captured["content"] = content
            captured["metadata"] = metadata or {}
            return "res-memory-1"

    monkeypatch.setattr(
        "agent.application.agent_center.memory.classify_turn_memory_type",
        lambda _q, _a: "semantic",
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.ttl_for_memory_type",
        lambda _t: "2099-01-01T00:00:00",
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.get_openviking_adapter",
        lambda: _FakeAdapter(),
    )
    persist_turn_memory(
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        prompt="question",
        answer="answer",
    )
    assert captured["project_uid"] == "p1"
    assert "Q: question\nA: answer" in captured["content"]
    assert captured["metadata"]["namespace"] == "memory"
    assert captured["metadata"]["user_uuid"] == "u1"
    assert captured["project_uid"] == "p1"
    assert captured["metadata"]["session_uid"] == "s1"


def test_query_turn_memory_uses_openviking_hits(monkeypatch):
    class _FakeHit:
        def __init__(self, *, content, score, metadata):
            self.content = content
            self.score = score
            self.metadata = metadata

    class _FakeAdapter:
        def search(self, request):
            assert request.project_uid == "p1"
            assert request.query == "method"
            return [
                _FakeHit(
                    content="A",
                    score=0.9,
                    metadata={"namespace": "memory", "memory_type": "semantic", "user_uuid": "u1"},
                ),
                _FakeHit(
                    content="B",
                    score=0.8,
                    metadata={"namespace": "rag", "memory_type": "semantic", "user_uuid": "u1"},
                ),
            ]

    monkeypatch.setattr(
        "agent.application.agent_center.memory.get_openviking_adapter",
        lambda: _FakeAdapter(),
    )

    items = query_turn_memory(
        uuid="u1",
        project_uid="p1",
        query="method",
        limit=3,
    )

    assert len(items) == 1
    assert items[0]["memory_type"] == "semantic"
    assert items[0]["content"] == "A"
