from agent.application.agent_center.facade import (
    AgentCenterRuntimeDeps,
    AgentCenterTurnRequest,
    execute_agent_center_turn,
)


def test_execute_agent_center_turn_delegates_to_turn_engine(monkeypatch):
    captured = {}

    def _fake_execute_turn_core(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok"}

    monkeypatch.setattr(
        "agent.application.agent_center.facade.execute_turn_core",
        _fake_execute_turn_core,
    )

    result = execute_agent_center_turn(
        request=AgentCenterTurnRequest(
            prompt="p",
            hinted_prompt="hp",
            force_plan=True,
            routing_context="ctx",
        ),
        deps=AgentCenterRuntimeDeps(
            leader_agent="agent",
            leader_runtime_config={"configurable": {"thread_id": "t1"}},
            leader_llm="llm",
            policy_llm="policy-llm",
            leader_tool_specs=[{"name": "search_document"}],
        ),
    )

    assert result["answer"] == "ok"
    assert captured["prompt"] == "p"
    assert captured["hinted_prompt"] == "hp"
    assert captured["leader_agent"] == "agent"
    assert captured["leader_runtime_config"]["configurable"]["thread_id"] == "t1"
    assert captured["policy_llm"] == "policy-llm"
    assert captured["force_plan"] is True
    assert captured["routing_context"] == "ctx"
    assert captured["leader_tool_specs"] == [{"name": "search_document"}]

