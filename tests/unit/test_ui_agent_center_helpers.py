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
            {"role": "assistant", "content": "A", "policy_decision": {"plan_enabled": True, "team_enabled": True, "reason": "long reason"}},
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
    queued = {}
    monkeypatch.setattr(
        "agent.application.agent_center.memory.save_project_memory_episode",
        lambda **kwargs: captured.update(kwargs) or "episode-1",
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.create_task",
        lambda task_id, uid, content_type, db_name="./database.sqlite": queued.update(
            {"task_id": task_id, "uid": uid, "content_type": content_type, "db_name": db_name}
        ),
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.enqueue_task",
        lambda task_func, task_id, episode_uid, user_uuid, db_name="./database.sqlite": queued.update(
            {
                "task_func": getattr(task_func, "__name__", ""),
                "enqueue_task_id": task_id,
                "episode_uid": episode_uid,
                "user_uuid": user_uuid,
                "enqueue_db_name": db_name,
            }
        )
        or {"mode": "queued", "job_id": "job-1"},
    )
    monkeypatch.setattr(
        "agent.application.agent_center.memory.update_task_status",
        lambda task_id, status, job_id=None, db_name="./database.sqlite": queued.update(
            {
                "status_task_id": task_id,
                "status": getattr(status, "value", status),
                "job_id": job_id,
                "status_db_name": db_name,
            }
        ),
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
    assert captured["prompt"] == "question"
    assert captured["answer"] == "answer"
    assert queued["uid"] == "episode-1"
    assert queued["content_type"] == "memory_writer"
    assert queued["task_func"] == "task_memory_writer"
    assert queued["episode_uid"] == "episode-1"
    assert queued["user_uuid"] == "u1"
    assert queued["status"] == "queued"
    assert queued["job_id"] == "job-1"
