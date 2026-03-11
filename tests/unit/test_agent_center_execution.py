from agent.application.agent_center.runtime_state import build_runtime_deps_from_session_state


def test_build_runtime_deps_from_session_state_normalizes_values():
    deps = build_runtime_deps_from_session_state(
        {
            "paper_agent": "agent",
            "paper_agent_runtime_config": {"configurable": {"thread_id": "t1"}},
            "paper_leader_llm": "llm",
            "paper_policy_router_llm": "policy-llm",
            "paper_evidence_retriever": lambda _q: {"evidences": []},
            "paper_current_tool_specs": [{"name": "search_document"}],
        }
    )

    assert deps.leader_agent == "agent"
    assert deps.leader_runtime_config["configurable"]["thread_id"] == "t1"
    assert deps.leader_llm == "llm"
    assert deps.policy_llm == "policy-llm"
    assert callable(deps.search_document_evidence_fn)
    assert deps.leader_tool_specs == [{"name": "search_document"}]


def test_build_runtime_deps_from_session_state_requires_agent():
    try:
        build_runtime_deps_from_session_state({})
    except ValueError as exc:
        assert "Leader agent" in str(exc)
        return
    raise AssertionError("Expected ValueError")
