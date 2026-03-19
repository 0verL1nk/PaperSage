from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from agent.middlewares.orchestration import OrchestrationMiddleware


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self._response)


def test_orchestration_middleware_injects_plan_guidance_for_complex_tasks():
    middleware = OrchestrationMiddleware(
        llm=_FakeLLM('{"is_complex": true, "needs_team": false, "reason": "multi-step"}')
    )
    state = {"messages": [HumanMessage(content="请先分步骤分析，再给结论。")]}

    update = middleware.before_model(
        state,
        runtime=None,
        config={"configurable": {"state": {}}},
    )

    assert update == {"needs_team": False}

    request = SimpleNamespace(messages=[HumanMessage(content="请先分步骤分析，再给结论。")])
    wrapped = middleware.wrap_model_call(request, lambda req: req)

    assert isinstance(wrapped.messages[0], SystemMessage)
    assert "规划工具" in str(wrapped.messages[0].content)


def test_orchestration_middleware_marks_team_tasks_and_injects_team_guidance():
    middleware = OrchestrationMiddleware(
        llm=_FakeLLM('{"is_complex": true, "needs_team": true, "reason": "multi-role"}')
    )
    state = {"messages": [HumanMessage(content="请从研究、评审、写作三个角色协作完成分析。")]}

    update = middleware.before_model(
        state,
        runtime=None,
        config={"configurable": {"state": {}}},
    )

    assert update == {"needs_team": True}
    assert middleware._last_analysis is not None
    assert middleware._last_analysis["is_complex"] is True
    assert middleware._last_analysis["needs_team"] is True
