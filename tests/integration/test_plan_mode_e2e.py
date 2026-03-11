from agent.application.turn_engine import execute_turn_core
from agent.domain.orchestration import ExecutionPlan, PlanStep, PolicyDecision


class _SequencedLeaderAgent:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def invoke(self, payload, config=None):
        self.calls.append({"payload": payload, "config": config})
        response = self._responses.pop(0) if self._responses else "ok"
        if isinstance(response, dict):
            return response
        return {"messages": [{"role": "assistant", "content": response}]}


def test_plan_mode_turn_engine_emits_step_trace_events_end_to_end(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        lambda *_args, **_kwargs: PolicyDecision(
            plan_enabled=True,
            team_enabled=False,
            reason="integration-forced",
            confidence=1.0,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.build_execution_plan",
        lambda *_args, **_kwargs: ExecutionPlan(
            goal="回答问题",
            steps=[
                PlanStep(
                    id="step_1",
                    title="检索证据",
                    done_when="需要文档证据",
                    tool_hints=["search_document"],
                ),
                PlanStep(id="step_2", title="提炼结论", done_when="形成结论"),
            ],
            done_when="输出最终回答",
        ),
    )
    leader = _SequencedLeaderAgent(
        [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "已检索到证据 [chunk_1]",
                        "tool_calls": [{"name": "search_document", "args": {"query": "q"}}],
                    },
                    {
                        "role": "tool",
                        "name": "search_document",
                        "content": "evidence chunk [chunk_1]",
                    },
                ]
            },
            "结论草稿：方法A更稳健。",
            "最终回答：方法A更稳健，证据见 [chunk_1]。",
        ]
    )

    result = execute_turn_core(
        prompt="请给出结论",
        hinted_prompt="请给出结论",
        leader_agent=leader,
        leader_runtime_config={},
        leader_llm=None,
    )

    assert result["policy_decision"]["plan_enabled"] is True
    assert result["policy_decision"]["team_enabled"] is False
    assert result["answer"].startswith("最终回答")
    assert result["runtime_state"] is not None
    assert result["runtime_state"]["completed_step_ids"] == ["step_1", "step_2"]

    performatives = [str(item.get("performative") or "") for item in result["trace_payload"]]
    assert "plan" in performatives
    assert "step_dispatch" in performatives
    assert "step_result" in performatives
    assert "step_verify" in performatives
    assert performatives.count("step_complete") == 2
    assert performatives[-1] == "final"


def test_plan_mode_turn_engine_replans_when_step_verification_fails(monkeypatch):
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.intercept_policy",
        lambda *_args, **_kwargs: PolicyDecision(
            plan_enabled=True,
            team_enabled=False,
            reason="integration-forced",
            confidence=1.0,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.build_execution_plan",
        lambda *_args, **_kwargs: ExecutionPlan(
            goal="回答问题",
            steps=[
                PlanStep(
                    id="step_1",
                    title="检索证据",
                    done_when="需要文档证据",
                    tool_hints=["search_document"],
                )
            ],
            done_when="输出最终回答",
        ),
    )
    monkeypatch.setattr(
        "agent.orchestration.orchestrator.revise_execution_plan",
        lambda **_kwargs: ExecutionPlan(
            goal="回答问题",
            steps=[
                PlanStep(
                    id="step_r1",
                    title="补充证据",
                    done_when="需要文档证据",
                    tool_hints=["search_document"],
                )
            ],
            done_when="输出最终回答",
        ),
    )
    leader = _SequencedLeaderAgent(
        [
            "无证据内容",
            "依然无证据",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "补充后获得证据 [chunk_2]",
                        "tool_calls": [{"name": "search_document", "args": {"query": "q"}}],
                    },
                    {
                        "role": "tool",
                        "name": "search_document",
                        "content": "evidence chunk [chunk_2]",
                    },
                ]
            },
            "最终回答：证据已补充。",
        ]
    )

    result = execute_turn_core(
        prompt="请给出结论",
        hinted_prompt="请给出结论",
        leader_agent=leader,
        leader_runtime_config={},
        leader_llm=None,
    )

    performatives = [str(item.get("performative") or "") for item in result["trace_payload"]]
    assert "step_retry" in performatives
    assert "replan" in performatives
    assert performatives[-1] == "final"
    assert result["runtime_state"] is not None
    assert "step_r1" in result["runtime_state"]["completed_step_ids"]
