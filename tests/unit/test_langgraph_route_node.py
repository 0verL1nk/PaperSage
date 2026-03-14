from __future__ import annotations

from agent.a2a.coordinator import WORKFLOW_PLAN_ACT, WORKFLOW_REACT, WORKFLOW_TEAM
from agent.a2a.router import _policy_to_workflow_mode
from agent.domain.orchestration import ExecutionPlan, PlanStep, PolicyDecision, TeamExecution
from agent.orchestration.orchestrator import execute_orchestrated_turn


class _FakeLeaderAgent:
    def invoke(self, _payload, config=None):
        assert isinstance(config, dict)
        return {"messages": [{"role": "assistant", "content": "ok"}]}


def _decision(*, plan_enabled: bool, team_enabled: bool) -> PolicyDecision:
    return PolicyDecision(
        plan_enabled=plan_enabled,
        team_enabled=team_enabled,
        reason="test",
        confidence=1.0,
        source="llm",
    )


def test_route_node_maps_policy_to_react_mode() -> None:
    from agent.orchestration.langgraph_route_node import route_from_policy_decision

    assert (
        route_from_policy_decision(_decision(plan_enabled=False, team_enabled=False))
        == WORKFLOW_REACT
    )


def test_route_node_maps_policy_to_plan_act_mode() -> None:
    from agent.orchestration.langgraph_route_node import route_from_policy_decision

    assert (
        route_from_policy_decision(_decision(plan_enabled=True, team_enabled=False))
        == WORKFLOW_PLAN_ACT
    )


def test_route_node_maps_team_policy_to_team_mode() -> None:
    from agent.orchestration.langgraph_route_node import route_from_policy_decision

    assert (
        route_from_policy_decision(_decision(plan_enabled=True, team_enabled=True)) == WORKFLOW_TEAM
    )


def test_route_node_mapping_keeps_parity_with_router_mapping() -> None:
    from agent.orchestration.langgraph_route_node import route_from_policy_decision

    for plan_enabled in (False, True):
        for team_enabled in (False, True):
            decision = _decision(plan_enabled=plan_enabled, team_enabled=team_enabled)
            assert route_from_policy_decision(decision) == _policy_to_workflow_mode(
                plan_enabled=plan_enabled,
                team_enabled=team_enabled,
            )


def test_orchestrator_directly_executes_plan_path_from_route_node(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        lambda *_args, **_kwargs: _decision(plan_enabled=True, team_enabled=False),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.build_execution_plan",
        lambda *_args, **_kwargs: ExecutionPlan(
            goal="回答问题",
            steps=[PlanStep(id="step_1", title="检索证据", done_when="有结果")],
        ),
    )

    result = execute_orchestrated_turn(
        prompt="请回答",
        hinted_prompt="请回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        llm=None,
    )

    performatives = [str(item.get("performative") or "") for item in result.trace_payload]
    assert "plan" in performatives
    assert "step_dispatch" in performatives


def test_orchestrator_runs_plan_then_team_for_team_route(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        lambda *_args, **_kwargs: _decision(plan_enabled=True, team_enabled=True),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.build_execution_plan",
        lambda *_args, **_kwargs: ExecutionPlan(
            goal="回答问题",
            steps=[PlanStep(id="step_1", title="检索证据", done_when="有结果")],
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.run_team_tasks",
        lambda **_kwargs: TeamExecution(
            enabled=True,
            roles=["researcher"],
            member_count=1,
            rounds=1,
            summary="team ok",
            todo_records=[],
            todo_stats={"done": 1},
            trace_events=[],
        ),
    )

    result = execute_orchestrated_turn(
        prompt="请回答",
        hinted_prompt="请回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        llm=object(),
    )

    performatives = [str(item.get("performative") or "") for item in result.trace_payload]
    assert result.team_execution.enabled is True
    assert "plan" in performatives
    assert "step_dispatch" in performatives
