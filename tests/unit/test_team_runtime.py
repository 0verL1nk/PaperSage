from agent.domain.orchestration import TeamRole
from agent.orchestration import team_runtime


def test_invoke_role_agent_blocks_spawn_tools_on_load(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeAgent:
        def invoke(self, _payload, config=None):
            assert isinstance(config, dict)
            return {"messages": [{"role": "assistant", "content": "[结论]\nok\n\n[证据]\n[chunk_1]\n\n[待验证点]\nnone"}]}

    def _fake_create_runtime_agent(**kwargs):
        captured["tool_names"] = [tool.name for tool in kwargs["tools"]]
        captured["system_prompt"] = kwargs["system_prompt"]
        return _FakeAgent()

    monkeypatch.setattr(team_runtime, "create_runtime_agent", _fake_create_runtime_agent)

    result = team_runtime._invoke_role_agent(
        llm=object(),
        role=TeamRole(name="researcher", goal="collect evidence"),
        prompt="q",
        plan_text="p",
        notes="",
        search_document_fn=lambda query: f"doc:{query}",
        search_document_evidence_fn=None,
        round_idx=1,
        prior_output="",
        task_details="look up evidence",
    )

    tool_names = set(captured["tool_names"])
    system_prompt = str(captured["system_prompt"])

    assert result.answer.startswith("[结论]")
    assert "start_plan" not in tool_names
    assert "start_team" not in tool_names
    assert "不能再次调用 start_plan/start_team" in system_prompt


def test_run_team_tasks_records_member_tool_events_before_member_output(monkeypatch):
    monkeypatch.setattr(
        team_runtime,
        "generate_dynamic_roles",
        lambda *_args, **_kwargs: [TeamRole(name="researcher", goal="collect evidence")],
    )
    monkeypatch.setattr(
        team_runtime,
        "_decide_team_rounds",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        team_runtime,
        "_invoke_role_agent",
        lambda **_kwargs: team_runtime.TeamTaskAttemptResult(
            answer="[结论]\ndone\n\n[证据]\n[chunk_1]\n\n[待验证点]\nnone",
            tool_names=["search_document"],
            tool_trace_events=[
                {
                    "sender": "leader",
                    "receiver": "search_document",
                    "performative": "tool_call",
                    "content": "{'query': 'q'}",
                },
                {
                    "sender": "search_document",
                    "receiver": "leader",
                    "performative": "tool_result",
                    "content": "evidence [chunk_1]",
                },
            ],
            skill_activation_events=[],
            tool_activation_events=[],
        ),
    )

    execution = team_runtime.run_team_tasks(
        prompt="q",
        plan_text="p",
        llm=object(),
        search_document_fn=lambda _query: "",
        search_document_evidence_fn=None,
        max_members=1,
        max_rounds=1,
    )

    performatives = [str(item.get("performative") or "") for item in execution.trace_events]
    assert "member_request" in performatives
    assert performatives.index("member_request") < performatives.index("tool_call")
    assert performatives.index("tool_call") < performatives.index("member_output")
    assert performatives.index("tool_result") < performatives.index("member_output")
    decision_events = [item for item in execution.trace_events if item.get("performative") == "leader_decision"]
    assert decision_events
