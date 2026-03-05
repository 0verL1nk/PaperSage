import pytest

from agent.orchestration import planning_service, policy_engine, team_runtime


class _MethodAwareLLM:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema, method=None):
        self.calls.append((schema, method))
        return {"chain": "ok"}


class _LegacyLLM:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema):
        self.calls.append(schema)
        return {"chain": "legacy"}


@pytest.mark.parametrize(
    ("helper", "schema"),
    [
        (policy_engine._with_structured_output, policy_engine.PolicyRouterOutput),
        (planning_service._with_structured_output, planning_service.PlannerOutput),
        (team_runtime._with_structured_output, team_runtime.RoundRouterOutput),
    ],
)
def test_structured_output_prefers_function_calling(helper, schema):
    llm = _MethodAwareLLM()
    output = helper(llm, schema)
    assert output == {"chain": "ok"}
    assert llm.calls == [(schema, "function_calling")]


@pytest.mark.parametrize(
    ("helper", "schema"),
    [
        (policy_engine._with_structured_output, policy_engine.PolicyRouterOutput),
        (planning_service._with_structured_output, planning_service.PlannerOutput),
        (team_runtime._with_structured_output, team_runtime.RoleRouterOutput),
    ],
)
def test_structured_output_fallbacks_for_legacy_signature(helper, schema):
    llm = _LegacyLLM()
    output = helper(llm, schema)
    assert output == {"chain": "legacy"}
    assert llm.calls == [schema]
