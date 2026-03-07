from agent.orchestration import policy_engine


class _NonDictStructured:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, value):
        self.calls.append(value)
        if isinstance(value, dict):
            raise ValueError("Invalid input type <class 'dict'>")
        return {
            "plan_enabled": True,
            "team_enabled": False,
            "reason": "needs plan",
            "confidence": 0.9,
        }


class _DictOnlyStructured:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, value):
        self.calls.append(value)
        if not isinstance(value, dict):
            raise TypeError("dict required")
        return {
            "plan_enabled": False,
            "team_enabled": False,
            "reason": "simple",
            "confidence": 0.8,
        }


class _LLMStub:
    def __init__(self, structured) -> None:
        self._structured = structured

    def with_structured_output(self, _schema, method=None):
        return self._structured


def test_route_with_llm_fallback_avoids_raw_dict_for_chat_models():
    structured = _NonDictStructured()
    decision = policy_engine._route_with_llm("测试问题", _LLMStub(structured))
    assert decision is not None
    assert decision.source == "llm"
    assert structured.calls
    assert not isinstance(structured.calls[0], dict)


def test_route_with_llm_fallback_keeps_dict_compat_for_simple_mocks():
    structured = _DictOnlyStructured()
    decision = policy_engine._route_with_llm("测试问题", _LLMStub(structured))
    assert decision is not None
    assert decision.source == "llm"
    assert any(isinstance(call, dict) for call in structured.calls)
