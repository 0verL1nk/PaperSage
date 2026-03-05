from agent.domain.orchestration import (
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
