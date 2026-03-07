from agent.application.agent_center.turn_execution import (
    TurnRuntimeInputs,
    execute_turn_with_runtime,
    resolve_turn_runtime_inputs,
)


def test_resolve_turn_runtime_inputs_validates_agent_and_normalizes():
    state = {
        "paper_agent": "agent",
        "paper_agent_runtime_config": {"configurable": {"thread_id": "t1"}},
        "paper_leader_llm": "llm",
        "paper_evidence_retriever": lambda _q: {"evidences": []},
    }
    runtime = resolve_turn_runtime_inputs(state)
    assert runtime.leader_agent == "agent"
    assert runtime.leader_runtime_config["configurable"]["thread_id"] == "t1"
    assert runtime.leader_llm == "llm"
    assert callable(runtime.search_document_evidence_fn)

    state = {
        "paper_agent": "agent",
        "paper_agent_runtime_config": "bad",
        "paper_evidence_retriever": "bad",
    }
    runtime = resolve_turn_runtime_inputs(state)
    assert runtime.leader_runtime_config == {}
    assert runtime.search_document_evidence_fn is None


def test_resolve_turn_runtime_inputs_raises_when_missing_agent():
    try:
        resolve_turn_runtime_inputs({})
    except ValueError as exc:
        assert "Leader agent" in str(exc)
        return
    raise AssertionError("Expected ValueError")


def test_execute_turn_with_runtime_delegates(monkeypatch):
    captured = {}

    def _fake_execute_turn_core(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok"}

    monkeypatch.setattr(
        "agent.application.agent_center.turn_execution.execute_turn_core",
        _fake_execute_turn_core,
    )
    runtime = TurnRuntimeInputs(
        leader_agent="agent",
        leader_runtime_config={"configurable": {"thread_id": "t1"}},
        leader_llm="llm",
        search_document_evidence_fn=None,
    )
    result = execute_turn_with_runtime(
        prompt="p",
        hinted_prompt="hp",
        runtime_inputs=runtime,
        force_plan=True,
        force_team=False,
        routing_context="ctx",
        on_event=lambda _item: None,
    )
    assert result["answer"] == "ok"
    assert captured["prompt"] == "p"
    assert captured["hinted_prompt"] == "hp"
    assert captured["leader_agent"] == "agent"
    assert captured["leader_runtime_config"]["configurable"]["thread_id"] == "t1"
    assert captured["force_plan"] is True
    assert captured["force_team"] is False
    assert captured["routing_context"] == "ctx"
    assert captured["emit_tool_load_event"] is True


def test_execute_turn_with_runtime_passes_emit_tool_load_flag(monkeypatch):
    captured = {}

    def _fake_execute_turn_core(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok"}

    monkeypatch.setattr(
        "agent.application.agent_center.turn_execution.execute_turn_core",
        _fake_execute_turn_core,
    )
    runtime = TurnRuntimeInputs(
        leader_agent="agent",
        leader_runtime_config={},
        leader_llm="llm",
        search_document_evidence_fn=None,
    )
    execute_turn_with_runtime(
        prompt="p",
        hinted_prompt="hp",
        runtime_inputs=runtime,
        emit_tool_load_event=False,
    )
    assert captured["emit_tool_load_event"] is False
