from agent.application.turn_engine import build_search_document_fn, execute_turn_core
from agent.domain.orchestration import OrchestratedTurn, PolicyDecision, TeamExecution


def test_build_search_document_fn_joins_evidence_text():
    search_fn = build_search_document_fn(
        lambda _query: {"evidences": [{"text": "a"}, {"text": "b"}, {"text": "  "}]}
    )
    assert search_fn("q") == "a\nb"


def test_execute_turn_core_with_injected_executor_replaces_evidence():
    events = []

    def _fake_executor(**kwargs):
        on_event = kwargs.get("on_event")
        if callable(on_event):
            on_event(
                {
                    "sender": "policy_engine",
                    "receiver": "leader",
                    "performative": "policy",
                    "content": "plan=False, team=False",
                }
            )
            on_event(
                {
                    "sender": "leader",
                    "receiver": "user",
                    "performative": "final",
                    "content": "结论 [证据]",
                }
            )
        return OrchestratedTurn(
            answer="结论 [证据]",
            policy_decision=PolicyDecision(
                plan_enabled=False,
                team_enabled=False,
                reason="heuristic",
                confidence=None,
                source="heuristic",
            ),
            team_execution=TeamExecution(enabled=False),
            trace_payload=[],
            leader_tool_names=["search_document"],
        )

    result = execute_turn_core(
        prompt="请给结论",
        hinted_prompt="请给结论",
        leader_agent=object(),
        leader_runtime_config={},
        search_document_evidence_fn=lambda _query: {
            "evidences": [{"chunk_id": "c1", "text": "证据文本", "page_no": 1}]
        },
        orchestrated_turn_executor=_fake_executor,
        on_event=lambda item: events.append(item),
    )

    assert result["used_document_rag"] is True
    assert result["evidence_items"]
    assert "[c1|p1]" in result["answer"]
    assert result["trace_payload"]
    assert events


def test_execute_turn_core_without_document_rag_skips_evidence():
    called = {"evidence": 0}

    def _fake_executor(**_kwargs):
        return OrchestratedTurn(
            answer='{"name":"主题","children":[]}',
            policy_decision=PolicyDecision(
                plan_enabled=True,
                team_enabled=False,
                reason="llm",
                confidence=0.8,
                source="llm",
            ),
            team_execution=TeamExecution(enabled=False, rounds=0),
            trace_payload=[],
            leader_tool_names=["search_web"],
        )

    def _evidence_fn(_query: str):
        called["evidence"] += 1
        return {"evidences": [{"chunk_id": "c2", "text": "x"}]}

    result = execute_turn_core(
        prompt="最新进展",
        hinted_prompt="最新进展",
        leader_agent=object(),
        leader_runtime_config={},
        search_document_evidence_fn=_evidence_fn,
        orchestrated_turn_executor=_fake_executor,
    )

    assert called["evidence"] == 0
    assert result["used_document_rag"] is False
    assert result["evidence_items"] == []
    assert isinstance(result["mindmap_data"], dict)


def test_execute_turn_core_emits_tool_load_event_from_tool_specs():
    captured_events: list[dict] = []

    def _fake_executor(**_kwargs):
        return OrchestratedTurn(
            answer="ok",
            policy_decision=PolicyDecision(
                plan_enabled=False,
                team_enabled=False,
                reason="heuristic",
                confidence=None,
                source="heuristic",
            ),
            team_execution=TeamExecution(enabled=False, rounds=0),
            trace_payload=[],
            leader_tool_names=[],
        )

    execute_turn_core(
        prompt="q",
        hinted_prompt="q",
        leader_agent=object(),
        leader_runtime_config={},
        leader_tool_specs=[
            {"name": "search_document"},
            {"name": "use_skill"},
        ],
        orchestrated_turn_executor=_fake_executor,
        on_event=lambda item: captured_events.append(item),
    )

    assert any(
        item.get("performative") == "tool_load"
        and "search_document" in str(item.get("content", ""))
        for item in captured_events
    )
