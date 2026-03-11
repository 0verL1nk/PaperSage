from types import SimpleNamespace

from agent.a2a.coordinator import WORKFLOW_PLAN_ACT_REPLAN, A2AMultiAgentCoordinator
from agent.a2a.standard import (
    A2A_VERSION_HEADER,
    METHOD_SEND_MESSAGE,
    A2AInMemoryServer,
    build_agent_card,
    build_coordinator_executor,
)
from agent.application.turn_engine import execute_turn_core
from agent.domain.orchestration import TeamRole


class _FakeLeaderAgent:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        if len(self.calls) == 1:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "name": "start_team",
                                "args": {"goal": "团队协作回答", "reason": "need team"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "start_team",
                        "content": '{"type":"mode_activate","mode":"team","goal":"团队协作回答"}',
                    },
                ]
            }
        return {"messages": [{"role": "assistant", "content": "final-from-leader"}]}


def _team_done_output(label: str) -> str:
    return f"[结论]\n{label}\n\n[证据]\nsearch evidence [chunk_1]\n\n[待验证点]\nnone"


class _ScriptedAgent:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append((payload, config))
        content = self._responses.pop(0) if self._responses else ""
        return {"messages": [SimpleNamespace(type="ai", content=content)]}


def test_agent_team_turn_engine_end_to_end(monkeypatch):
    from agent.domain.orchestration import PolicyDecision

    def mock_intercept(ctx, *args, **kwargs):
        return PolicyDecision(
            plan_enabled=True,
            team_enabled=True,
            reason="test",
            confidence=0.9,
            source="llm",
        )

    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        mock_intercept,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime.generate_dynamic_roles",
        lambda *_args, **_kwargs: [
            TeamRole(name="researcher", goal="collect evidence"),
            TeamRole(name="reviewer", goal="cross-check findings"),
        ],
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._decide_team_rounds",
        lambda **_kwargs: 2,
    )
    monkeypatch.setattr(
        "agent.orchestration.team_runtime._invoke_role_agent",
        lambda **kwargs: _team_done_output(f"{kwargs['role'].name}-done"),
    )

    event_log = []
    result = execute_turn_core(
        prompt="请做团队协作回答",
        hinted_prompt="请做团队协作回答",
        leader_agent=_FakeLeaderAgent(),
        leader_runtime_config={},
        leader_llm=object(),
        on_event=lambda event: event_log.append(event),
    )

    assert result["answer"] == "final-from-leader"
    assert result["policy_decision"]["plan_enabled"] is False
    assert result["policy_decision"]["team_enabled"] is True
    assert result["team_execution"]["enabled"] is True
    assert result["team_execution"]["rounds"] == 2
    assert result["team_execution"]["roles"] == ["researcher", "reviewer"]

    performatives = [item["performative"] for item in result["trace_payload"]]
    assert performatives[0] == "request"
    assert performatives[-1] == "final"
    assert performatives.count("dispatch") == 4
    assert performatives.count("member_output") == 4
    assert performatives.count("member_request") == 4
    assert performatives.count("leader_decision") == 4

    assert event_log
    assert all(isinstance(item.get("a2aMessage"), dict) for item in event_log)
    assert all(item["a2aMessage"]["messageId"] == item["span_id"] for item in event_log)


def test_multi_agent_server_replan_pipeline_end_to_end():
    react = _ScriptedAgent([])
    planner = _ScriptedAgent(["plan-1", "plan-2"])
    researcher = _ScriptedAgent(["draft-1", "draft-2"])
    reviewer = _ScriptedAgent(
        [
            "Decision: REVISE\nFeedback: add evidence",
            "Decision: PASS\nFeedback: good",
        ]
    )
    coordinator = A2AMultiAgentCoordinator(
        react_agent=react,
        planner_agent=planner,
        researcher_agent=researcher,
        reviewer_agent=reviewer,
        session_id="e2e-team-session",
    )
    server = A2AInMemoryServer(
        agent_card=build_agent_card(
            name="Paper Agent",
            description="integration e2e",
            url="http://localhost/a2a",
        ),
        execute_message_fn=build_coordinator_executor(
            coordinator,
            workflow_mode=WORKFLOW_PLAN_ACT_REPLAN,
        ),
    )

    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": "e2e-1",
            "method": METHOD_SEND_MESSAGE,
            "headers": {A2A_VERSION_HEADER: "1.0"},
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "请完成多智能体协作"}],
                    "kind": "message",
                    "messageId": "m-e2e",
                }
            },
        }
    )

    assert "result" in response
    task = response["result"]
    assert task["status"]["state"] == "completed"
    final_answer = task["artifacts"][0]["parts"][0]["text"]
    assert final_answer == "draft-2"
    assert len(planner.calls) >= 2
    assert len(reviewer.calls) >= 2
