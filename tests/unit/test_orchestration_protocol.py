import json

from agent.domain.orchestration import (
    PolicyDecision,
    TeamExecution,
    TeamRole,
    build_trace_event,
    create_trace_context,
)
from agent.orchestration.orchestrator import execute_orchestrated_turn
from agent.orchestration.team_runtime import run_team_tasks


class _FakeLeaderAgent:
    def invoke(self, _payload, config=None):
        assert isinstance(config, dict)
        return {"messages": [{"role": "assistant", "content": "ok"}]}


def test_build_trace_event_contains_required_envelope_fields():
    context = create_trace_context(
        run_id="run-test",
        task_id="task-test",
        channel="internal.test",
    )
    event = build_trace_event(
        context=context,
        sender="user",
        receiver="leader",
        performative="request",
        content="hello",
    )

    assert event["run_id"] == "run-test"
    assert event["task_id"] == "task-test"
    assert event["channel"] == "internal.test"
    assert event["sequence"] == 1
    assert event["span_id"] == "run-test:span:1"
    assert event["parent_span_id"] == ""
    assert event["timestamp"]
    assert isinstance(event.get("a2aMessage"), dict)
    assert event["a2aMessage"]["taskId"] == "task-test"
    assert event["a2aMessage"]["contextId"] == "run-test"
    assert event["a2aMessage"]["metadata"]["performative"] == "request"


def test_build_trace_event_rejects_invalid_route():
    context = create_trace_context(run_id="run-invalid", task_id="task-invalid")
    try:
        build_trace_event(
            context=context,
            sender="leader",
            receiver="leader",
            performative="dispatch",
            content="invalid",
        )
    except ValueError as exc:
        assert "Invalid trace route" in str(exc)
    else:
        raise AssertionError("Expected invalid dispatch route to raise ValueError")


def test_execute_orchestrated_turn_emits_protocolized_trace():
    result = execute_orchestrated_turn(
        prompt="请回答",
        hinted_prompt="请回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        llm=None,
        force_plan=False,
        force_team=False,
    )

    assert result.trace_payload
    sequences = [int(item.get("sequence", 0)) for item in result.trace_payload]
    assert sequences == sorted(sequences)
    assert sequences[0] == 1
    for item in result.trace_payload:
        assert item.get("run_id")
        assert item.get("task_id")
        assert item.get("span_id")
        assert "timestamp" in item
        assert item.get("channel") == "internal.orchestrator"
        assert isinstance(item.get("a2aMessage"), dict)
        assert item["a2aMessage"]["messageId"] == item["span_id"]


def test_run_team_tasks_normalizes_role_name_and_emits_round_metadata(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="Leader", goal="collect evidence")],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **_kwargs: "done",
    )
    events = []

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
        on_event=lambda event: events.append(event),
    )

    assert execution.enabled is True
    assert execution.roles == ["leader_role"]
    assert len(events) == 2
    assert events[0]["performative"] == "dispatch"
    assert events[0]["meta"]["round"] == 1
    assert events[0]["meta"]["role"] == "leader_role"
    assert events[1]["performative"] == "review"
    assert events[1]["meta"]["round"] == 1


def test_run_team_tasks_uses_todo_dependencies_for_ready_queue(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [
            TeamRole(name="researcher", goal="collect evidence"),
            TeamRole(name="reviewer", goal="cross check"),
        ],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 2,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **kwargs: f"{kwargs['role'].name}-done",
    )
    events: list[dict] = []

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=2,
        max_rounds=2,
        on_event=lambda event: events.append(event),
    )

    assert execution.enabled is True
    assert execution.roles == ["researcher", "reviewer"]
    assert execution.rounds == 2
    assert len(execution.todo_records) == 4
    assert execution.todo_stats["done"] == 4
    assert execution.todo_stats["blocked"] == 0

    dispatch_events = [e for e in events if e["performative"] == "dispatch"]
    dispatch_todo_ids = [str(e.get("meta", {}).get("todo_id") or "") for e in dispatch_events]
    assert dispatch_todo_ids == [
        "team_r1_researcher",
        "team_r1_reviewer",
        "team_r2_researcher",
        "team_r2_reviewer",
    ]

    todo_map = {str(item["id"]): item for item in execution.todo_records}
    assert todo_map["team_r1_researcher"]["dependencies"] == []
    assert todo_map["team_r1_reviewer"]["dependencies"] == ["team_r1_researcher"]
    assert todo_map["team_r2_researcher"]["dependencies"] == ["team_r1_researcher"]
    assert todo_map["team_r2_reviewer"]["dependencies"] == [
        "team_r1_reviewer",
        "team_r2_researcher",
    ]


def test_run_team_tasks_blocks_when_dependency_cycle_detected(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="researcher", goal="collect evidence")],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._build_team_todo_records",
        lambda **_kwargs: [
            {
                "id": "task_a",
                "title": "a",
                "status": "todo",
                "assignee": "researcher",
                "dependencies": ["task_b"],
                "history": [],
            },
            {
                "id": "task_b",
                "title": "b",
                "status": "todo",
                "assignee": "researcher",
                "dependencies": ["task_a"],
                "history": [],
            },
        ],
    )

    execution = run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
    )

    assert execution.enabled is True
    assert execution.fallback_reason == "todo_dependency_cycle"
    assert execution.todo_stats["blocked"] == 2
    assert execution.trace_events == []


def test_execute_orchestrated_turn_syncs_team_todo_store(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_TOOLS_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_TODO_FILE", ".agent/todo.json")
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        lambda *_args, **_kwargs: PolicyDecision(
            plan_enabled=False,
            team_enabled=True,
            reason="forced",
            confidence=1.0,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.run_team_tasks",
        lambda **_kwargs: TeamExecution(
            enabled=True,
            roles=["researcher"],
            member_count=1,
            rounds=1,
            summary="ok",
            todo_records=[
                {
                    "id": "team_r1_researcher",
                    "title": "[Round 1] researcher",
                    "status": "done",
                    "assignee": "researcher",
                    "dependencies": [],
                    "history": [],
                }
            ],
            todo_stats={"done": 1},
            trace_events=[],
        ),
    )

    execute_orchestrated_turn(
        prompt="请回答",
        hinted_prompt="请回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        llm=object(),
    )

    todo_path = tmp_path / ".agent" / "todo.json"
    assert todo_path.exists()
    payload = json.loads(todo_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["id"] == "team_r1_researcher"
