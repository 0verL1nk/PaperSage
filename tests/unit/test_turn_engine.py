from agent.application.turn_engine import (
    _maybe_to_dict,
    build_search_document_fn,
    execute_turn_core,
    try_parse_mindmap,
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
        leader_agent=mock_agent,
        leader_runtime_config={},
        search_document_evidence_fn=_evidence_fn,
    )

    assert called["evidence"] == 0
    assert result["used_document_rag"] is False
    assert result["evidence_items"] == []
    assert result["mindmap_data"] is None or isinstance(result["mindmap_data"], dict)


def test_execute_turn_core_uses_search_document_tool_result_evidence_without_reretrieval():
    from types import SimpleNamespace
    from unittest.mock import Mock

    called = {"evidence": 0}

    def _evidence_fn(_query: str):
        called["evidence"] += 1
        return {"evidences": [{"chunk_id": "other_chunk", "text": "不会被使用"}]}

    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "search_document", "args": {"query": "rag"}}],
            ),
            {
                "role": "tool",
                "name": "search_document",
                "content": (
                    '{"evidences": ['
                    '{"chunk_id": "arxiv:2005.11401:chunk_11", "text": "证据文本", "page_no": 1, "offset_start": 0, "offset_end": 10}'
                    "]} "
                ),
            },
            SimpleNamespace(
                content="结论 <evidence>arxiv:2005.11401:chunk_11|p1|o0-10</evidence>",
                tool_calls=[],
            ),
        ]
    }

    result = execute_turn_core(
        prompt="请概括 RAG 核心结论",
        leader_agent=mock_agent,
        leader_runtime_config={},
        search_document_evidence_fn=_evidence_fn,
    )

    assert called["evidence"] == 0
    assert result["used_document_rag"] is True
    assert result["evidence_items"]
    assert result["evidence_items"][0]["chunk_id"] == "arxiv:2005.11401:chunk_11"


def test_execute_turn_core_matches_plain_doc_uid_citation_to_tool_evidence() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "search_document", "args": {"query": "rag"}}],
            ),
            {
                "role": "tool",
                "name": "search_document",
                "content": (
                    '{"evidences": ['
                    '{"doc_uid": "arxiv:2005.11401", "chunk_id": "arxiv:2005.11401:chunk_11", "text": "证据文本", "page_no": 1, "offset_start": 0, "offset_end": 10}'
                    "]} "
                ),
            },
            SimpleNamespace(
                content="结论引用 arxiv:2005.11401|p1|o0-10，建议优先采用标准 RAG。",
                tool_calls=[],
            ),
        ]
    }

    result = execute_turn_core(
        prompt="请概括 RAG 核心结论",
        leader_agent=mock_agent,
        leader_runtime_config={},
    )

    assert result["used_document_rag"] is True
    assert result["evidence_items"]
    assert result["evidence_items"][0]["chunk_id"] == "arxiv:2005.11401:chunk_11"


def test_execute_turn_core_does_not_count_tool_result_evidence_without_answer_citations():
    from types import SimpleNamespace
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            SimpleNamespace(
                content="",
                tool_calls=[{"name": "search_document", "args": {"query": "rag"}}],
            ),
            {
                "role": "tool",
                "name": "search_document",
                "content": (
                    '{"evidences": ['
                    '{"chunk_id": "arxiv:2005.11401:chunk_11", "text": "证据文本", "page_no": 1}'
                    "]} "
                ),
            },
            SimpleNamespace(
                content="这是没有证据标签的总结。",
                tool_calls=[],
            ),
        ]
    }

    result = execute_turn_core(
        prompt="请概括 RAG 核心结论",
        leader_agent=mock_agent,
        leader_runtime_config={},
    )

    assert result["used_document_rag"] is True
    assert result["evidence_items"] == []


def test_execute_turn_core_infers_final_phase_from_answer_without_final_event():
    from types import SimpleNamespace

    class _Agent:
        def invoke(self, payload, config=None):
            assert payload["messages"][0]["content"] == "请总结"
            if isinstance(config, dict):
                configurable = config.get("configurable")
                if isinstance(configurable, dict):
                    on_event = configurable.get("on_event")
                    if callable(on_event):
                        on_event(
                            {
                                "sender": "leader",
                                "receiver": "leader",
                                "performative": "unknown_internal_phase",
                                "content": "处理中",
                            }
                        )
            return {
                "messages": [
                    SimpleNamespace(
                        content="最终回答",
                        tool_calls=[],
                    )
                ]
            }

    result = execute_turn_core(
        prompt="请总结",
        leader_agent=_Agent(),
        leader_runtime_config={},
    )

    assert result["answer"] == "最终回答"
    assert result["phase_path"].endswith("输出最终答案")


def test_execute_turn_core_sends_raw_user_prompt_and_turn_context() -> None:
    captured: dict[str, object] = {}

    class _Agent:
        def invoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    result = execute_turn_core(
        prompt="真实用户问题",
        turn_context={
            "response_language": "en",
            "memory_items": [{"memory_type": "semantic", "content": "prefers concise answers"}],
        },
        leader_agent=_Agent(),
        leader_runtime_config={"configurable": {"thread_id": "tid"}},
    )

    assert result["answer"] == "ok"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"] == [{"role": "user", "content": "真实用户问题"}]
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["configurable"]["thread_id"] == "tid"
    assert config["configurable"]["turn_context"] == {
        "response_language": "en",
        "memory_items": [{"memory_type": "semantic", "content": "prefers concise answers"}],
    }


def test_execute_turn_core_logs_final_answer(caplog) -> None:
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [
            Mock(
                content="最终回答内容",
                tool_calls=[],
            )
        ]
    }

    with caplog.at_level("INFO"):
        result = execute_turn_core(
            prompt="请总结",
            leader_agent=mock_agent,
            leader_runtime_config={},
        )

    assert result["answer"] == "最终回答内容"
    assert "TURN_FINAL_ANSWER: 最终回答内容" in caplog.text


def test_try_parse_mindmap_accepts_extra_text_after_json_inside_tag() -> None:
    answer = """<mindmap>
{
  "name": "Seed-TTS系统",
  "children": [{"name": "模型", "children": []}]
}
补充说明：这一层表示核心模块。
</mindmap>"""

    parsed = try_parse_mindmap(answer)

    assert parsed == {
        "name": "Seed-TTS系统",
        "children": [{"name": "模型", "children": []}],
    }


def test_try_parse_mindmap_accepts_wrapped_text_around_tag() -> None:
    answer = """下面是导图结果：
<mindmap>
{
  "name": "主题",
  "children": []
}
</mindmap>
以上。"""

    parsed = try_parse_mindmap(answer)

    assert parsed == {"name": "主题", "children": []}


def test_try_parse_mindmap_ignores_second_json_object_inside_tag() -> None:
    answer = """<mindmap>
{"name": "主题", "children": []}
{"other": 1}
</mindmap>"""

    parsed = try_parse_mindmap(answer)

    assert parsed == {"name": "主题", "children": []}


def test_try_parse_mindmap_ignores_trailing_braces_in_comment_text() -> None:
    answer = """<mindmap>
{"name": "主题", "children": []}
注释里还有 {brace}
</mindmap>"""

    parsed = try_parse_mindmap(answer)

    assert parsed == {"name": "主题", "children": []}


def test_try_parse_mindmap_accepts_repeated_mindmap_blocks() -> None:
    answer = """<mindmap>
{"name": "主题A", "children": []}
</mindmap>
<mindmap>
{"name": "主题B", "children": []}
</mindmap>"""

    parsed = try_parse_mindmap(answer)

    assert parsed == {"name": "主题A", "children": []}


def test_maybe_to_dict_handles_none_and_noncallable_values():
    assert _maybe_to_dict(None) is None
    assert _maybe_to_dict({"x": 1}) == {"x": 1}

    class _Payload:
        def to_dict(self):
            return {"ok": True}

    assert _maybe_to_dict(_Payload()) == {"ok": True}


def test_execute_turn_core_exposes_team_handoff_and_scheduler_convenience():
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.invoke.return_value = {
        "messages": [Mock(content="请先执行 todo", tool_calls=[])],
        "needs_team": True,
        "team_handoff": {"mode": "leader_teammate", "reason": "multi-role"},
        "todo_scheduler_hint": {
            "ready_todo_ids": ["todo-2"],
            "blocked_todo_ids": [],
            "completed_todo_ids": ["todo-1"],
            "in_progress_todo_ids": [],
        },
        "todos": [
            {
                "id": "todo-1",
                "content": "检索证据",
                "status": "completed",
                "depends_on": [],
                "assignee": "researcher",
                "execution_backend": "local",
                "result": {"output": "done"},
            },
            {
                "id": "todo-2",
                "content": "整理结论",
                "status": "ready",
                "depends_on": ["todo-1"],
                "assignee": "writer",
                "execution_backend": "a2a",
            },
        ],
    }

    result = execute_turn_core(
        prompt="请协作完成分析",
        hinted_prompt="请协作完成分析",
        leader_agent=mock_agent,
        leader_runtime_config={},
    )

    assert result["policy_decision"]["team_enabled"] is True
    assert result["team_handoff"]["mode"] == "leader_teammate"
    assert result["todo_scheduler_hint"]["ready_todo_ids"] == ["todo-2"]
    assert result["team_execution"]["enabled"] is True
    assert result["team_execution"]["todo_stats"]["done"] == 1
    assert result["team_execution"]["todo_stats"]["todo"] == 1
    assert result["team_execution"]["todo_records"][0]["assignee"] == "researcher"
    assert result["team_execution"]["todo_records"][1]["dependencies"] == ["todo-1"]
