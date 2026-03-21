from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agent.middlewares.orchestration import OrchestrationMiddleware


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self._response)


class _FakeStructuredInvoker:
    def __init__(self, response: BaseModel) -> None:
        self._response = response

    def invoke(self, _prompt: str) -> BaseModel:
        return self._response


class _FakeStructuredLLM:
    def __init__(self) -> None:
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return _FakeStructuredInvoker(
            schema(is_complex=True, needs_team=False, reason="structured")
        )

    def invoke(self, _prompt: str) -> SimpleNamespace:
        raise AssertionError("raw invoke should not be used when structured output is available")


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


def test_orchestration_middleware_prefers_structured_output_when_available():
    llm = _FakeStructuredLLM()
    middleware = OrchestrationMiddleware(llm=llm)

    update = middleware.before_model(
        {"messages": [HumanMessage(content="请先分步骤分析，再给结论。")]},
        runtime=None,
        config={"configurable": {"state": {}}},
    )

    assert update == {"needs_team": False}
    assert llm.schema is not None
    assert middleware._last_analysis == {
        "is_complex": True,
        "needs_team": False,
        "reason": "structured",
        "has_plan": False,
    }


def test_orchestration_middleware_parses_last_valid_json_from_mixed_text():
    response = """只返回JSON格式:
{"is_complex": true/false, "needs_team": true/false, "reason": "简短原因"}

分析结果如下:
{"is_complex": true, "needs_team": true, "reason": "需要对比研究"}
"""
    middleware = OrchestrationMiddleware(llm=_FakeLLM(response))

    update = middleware.before_model(
        {"messages": [HumanMessage(content="请对比这两个方案并给出建议。")]},
        runtime=None,
        config={"configurable": {"state": {}}},
    )

    assert update == {"needs_team": True}
    assert middleware._last_analysis is not None
    assert middleware._last_analysis["is_complex"] is True
    assert middleware._last_analysis["needs_team"] is True
    assert middleware._last_analysis["reason"] == "需要对比研究"


def test_orchestration_middleware_does_not_guess_team_need_from_fallback_keywords():
    middleware = OrchestrationMiddleware(llm=_FakeLLM("not-json"))

    update = middleware.before_model(
        {"messages": [HumanMessage(content="请让研究和评审两个角色协作完成分析。")]},
        runtime=None,
        config={"configurable": {"state": {}}},
    )

    assert update == {"needs_team": False}
    assert middleware._last_analysis is not None
    assert middleware._last_analysis["is_complex"] is True
    assert middleware._last_analysis["needs_team"] is False
    assert middleware._last_analysis["reason"] == "fallback heuristic"
