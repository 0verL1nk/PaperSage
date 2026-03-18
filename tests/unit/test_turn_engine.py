from agent.application.turn_engine import (
    _maybe_to_dict,
    build_search_document_fn,
    execute_turn_core,
)


def test_build_search_document_fn_joins_evidence_text():
    search_fn = build_search_document_fn(
        lambda _query: {"evidences": [{"text": "a"}, {"text": "b"}, {"text": "  "}]}
    )
    assert search_fn("q") == "a\nb"


def test_execute_turn_core_with_injected_executor_replaces_evidence():
    from unittest.mock import Mock

    events = []
    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            Mock(
                content="结论 <evidence>c1|p1|o0-10</evidence>",
                tool_calls=[{"name": "search_document", "args": {"query": "test"}}],
            )
        ]
    }

    result = execute_turn_core(
        prompt="请给结论",
        hinted_prompt="请给结论",
        leader_agent=mock_agent,
        leader_runtime_config={},
        search_document_evidence_fn=lambda _query: {
            "evidences": [{"chunk_id": "c1", "text": "证据文本", "page_no": 1}]
        },
        on_event=lambda item: events.append(item),
    )

    assert result["used_document_rag"] is True
    assert result["evidence_items"]
    assert len(result["evidence_items"]) == 1
    assert result["evidence_items"][0]["chunk_id"] == "c1"


def test_execute_turn_core_without_document_rag_skips_evidence():
    from unittest.mock import Mock

    called = {"evidence": 0}

    def _evidence_fn(_query: str):
        called["evidence"] += 1
        return {"evidences": [{"chunk_id": "c2", "text": "x"}]}

    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            Mock(
                content='{"name":"主题","children":[]}',
                tool_calls=[{"name": "search_web", "args": {"query": "test"}}],
            )
        ]
    }

    result = execute_turn_core(
        prompt="最新进展",
        hinted_prompt="最新进展",
        leader_agent=mock_agent,
        leader_runtime_config={},
        search_document_evidence_fn=_evidence_fn,
    )

    assert called["evidence"] == 0
    assert result["used_document_rag"] is False
    assert result["evidence_items"] == []
    assert result["mindmap_data"] is None or isinstance(result["mindmap_data"], dict)


def test_maybe_to_dict_handles_none_and_noncallable_values():
    assert _maybe_to_dict(None) is None
    assert _maybe_to_dict({"x": 1}) is None

    class _Payload:
        def to_dict(self):
            return {"ok": True}

    assert _maybe_to_dict(_Payload()) == {"ok": True}
