from agent.orchestration.contracts import OrchestratedTurn, PolicyDecision, TeamExecution
from agent.turn_service import execute_turn_core


def test_execute_turn_core_replaces_evidence_and_collects_trace(monkeypatch):
    def _fake_orchestrated_turn(**kwargs):
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

    monkeypatch.setattr("agent.turn_service.execute_orchestrated_turn", _fake_orchestrated_turn)

    def _evidence_fn(_query: str):
        return {
            "evidences": [
                {"chunk_id": "c1", "text": "证据文本", "page_no": 1},
            ]
        }

    result = execute_turn_core(
        prompt="请给结论",
        hinted_prompt="请给结论",
        leader_agent=object(),
        leader_runtime_config={},
        leader_llm=None,
        search_document_evidence_fn=_evidence_fn,
    )

    assert result["policy_decision"]["plan_enabled"] is False
    assert result["evidence_items"]
    assert "[c1|p1]" in result["answer"]
    assert result["trace_payload"]


def test_execute_turn_core_without_document_rag_skips_evidence(monkeypatch):
    called = {"evidence": 0}

    def _fake_orchestrated_turn(**kwargs):
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

    monkeypatch.setattr("agent.turn_service.execute_orchestrated_turn", _fake_orchestrated_turn)

    def _evidence_fn(_query: str):
        called["evidence"] += 1
        return {"evidences": [{"chunk_id": "c2", "text": "x"}]}

    result = execute_turn_core(
        prompt="最新进展",
        hinted_prompt="最新进展",
        leader_agent=object(),
        leader_runtime_config={},
        leader_llm=None,
        search_document_evidence_fn=_evidence_fn,
    )

    assert called["evidence"] == 0
    assert result["used_document_rag"] is False
    assert result["evidence_items"] == []
    assert isinstance(result["mindmap_data"], dict)
