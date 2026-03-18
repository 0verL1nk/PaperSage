from types import SimpleNamespace

from agent.application.agent_center.page_orchestrator import (
    apply_turn_result,
    build_turn_execution_context,
    gate_prompt_and_enqueue,
    prepare_scope_runtime,
)


def test_prepare_scope_runtime_success_and_failure():
    calls = {"prepare": 0, "messages": 0, "compact": 0, "usage": 0}
    result = prepare_scope_runtime(
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        user_uuid="u1",
        project_uid="p1",
        project_name="P",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs=[{"uid": "d1"}],
        load_scope_docs_with_text_fn=lambda **_kwargs: (
            [{"uid": "d1", "file_name": "A", "text": "t"}],
            {"session_hit": 1, "db_restore": 0, "extracted": 0},
            None,
        ),
        build_scope_cache_caption_fn=lambda stats: f"caption:{stats['session_hit']}",
        build_scope_signature_fn=lambda docs: f"sig:{len(docs)}",
        prepare_agent_session_fn=lambda *args, **kwargs: calls.__setitem__("prepare", calls["prepare"] + 1),
        ensure_conversation_messages_fn=lambda **_kwargs: calls.__setitem__("messages", calls["messages"] + 1),
        ensure_compact_summary_fn=lambda **_kwargs: calls.__setitem__("compact", calls["compact"] + 1),
        update_context_usage_fn=lambda *_args: calls.__setitem__("usage", calls["usage"] + 1),
    )
    assert result is not None
    assert result.scope_signature == "sig:1"
    assert result.cache_caption == "caption:1"
    assert calls == {"prepare": 1, "messages": 1, "compact": 1, "usage": 1}
    assert result.scope_docs_with_text == [{"uid": "d1", "file_name": "A"}]

    result = prepare_scope_runtime(
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        user_uuid="u1",
        project_uid="p1",
        project_name="P",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs=[{"uid": "d1"}],
        load_scope_docs_with_text_fn=lambda **_kwargs: ([], {}, "d1"),
        build_scope_cache_caption_fn=lambda _stats: "",
        build_scope_signature_fn=lambda _docs: "sig",
        prepare_agent_session_fn=lambda *_args: None,
        ensure_conversation_messages_fn=lambda **_kwargs: None,
        ensure_compact_summary_fn=lambda **_kwargs: None,
        update_context_usage_fn=lambda *_args: None,
    )
    assert result is None


def test_prepare_scope_runtime_skips_loading_when_cached():
    calls = {"load": 0, "prepare": 0}
    result = prepare_scope_runtime(
        logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        user_uuid="u1",
        project_uid="p1",
        project_name="P",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_docs=[{"uid": "d1", "file_name": "A", "file_path": "/tmp/a"}],
        load_scope_docs_with_text_fn=lambda **_kwargs: calls.__setitem__("load", calls["load"] + 1),
        build_scope_cache_caption_fn=lambda _stats: "",
        build_scope_signature_fn=lambda docs: f"sig:{len(docs)}",
        has_cached_session_fn=lambda *_args: True,
        prepare_agent_session_fn=lambda *args, **kwargs: calls.__setitem__("prepare", calls["prepare"] + 1),
        ensure_conversation_messages_fn=lambda **_kwargs: None,
        ensure_compact_summary_fn=lambda **_kwargs: None,
        update_context_usage_fn=lambda *_args: None,
    )
    assert result is not None
    assert result.scope_signature == "sig:1"
    assert calls == {"load": 0, "prepare": 1}
    assert result.scope_docs_with_text == [{"uid": "d1", "file_name": "A", "file_path": "/tmp/a"}]


def test_gate_prompt_and_enqueue_states():
    session_state = {}
    side_effects = {"clear": 0, "enqueue": 0, "persist": 0}
    result = gate_prompt_and_enqueue(
        session_state=session_state,
        turn_in_progress=True,
        pending_turn={"prompt": "x"},
        prompt_input=None,
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_signature="sig",
        resolve_active_prompt_fn=lambda **_kwargs: (None, "mismatch_pending"),
        clear_turn_lock_fn=lambda _state: side_effects.__setitem__("clear", side_effects["clear"] + 1),
        enqueue_user_turn_fn=lambda **_kwargs: side_effects.__setitem__("enqueue", side_effects["enqueue"] + 1),
        persist_active_conversation_fn=lambda **_kwargs: side_effects.__setitem__("persist", side_effects["persist"] + 1),
        user_uuid="u1",
    )
    assert result.state == "rerun"
    assert side_effects["clear"] == 1

    result = gate_prompt_and_enqueue(
        session_state=session_state,
        turn_in_progress=False,
        pending_turn=None,
        prompt_input="q",
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_signature="sig",
        resolve_active_prompt_fn=lambda **_kwargs: ("q", "new_prompt"),
        clear_turn_lock_fn=lambda _state: side_effects.__setitem__("clear", side_effects["clear"] + 1),
        enqueue_user_turn_fn=lambda **_kwargs: side_effects.__setitem__("enqueue", side_effects["enqueue"] + 1),
        persist_active_conversation_fn=lambda **_kwargs: side_effects.__setitem__("persist", side_effects["persist"] + 1),
        user_uuid="u1",
    )
    assert result.state == "rerun"
    assert side_effects["enqueue"] == 1
    assert side_effects["persist"] == 1

    result = gate_prompt_and_enqueue(
        session_state=session_state,
        turn_in_progress=True,
        pending_turn={"prompt": "q"},
        prompt_input=None,
        project_uid="p1",
        session_uid="s1",
        conversation_key="p1:s1",
        scope_signature="sig",
        resolve_active_prompt_fn=lambda **_kwargs: ("q", "resume_pending"),
        clear_turn_lock_fn=lambda _state: None,
        enqueue_user_turn_fn=lambda **_kwargs: None,
        persist_active_conversation_fn=lambda **_kwargs: None,
        user_uuid="u1",
    )
    assert result.state == "ready"
    assert result.prompt == "q"


def test_build_turn_execution_context():
    context = build_turn_execution_context(
        prompt="q",
        user_uuid="u1",
        project_uid="p1",
        session_state={
            "agent_messages": [{"role": "user", "content": "q"}],
            "paper_agent_runtime_config": {"configurable": {"thread_id": "tid"}},
        },
        build_routing_context_fn=lambda messages, summary: f"{len(messages)}:{summary}",
        build_hinted_prompt_fn=lambda **kwargs: f"{kwargs['prompt']}",
        resolve_runtime_session_id_fn=lambda cfg: str(cfg["configurable"]["thread_id"]),
        resolve_selected_doc_uid_for_logging_fn=lambda docs: str(docs[0]["uid"]),
        scope_docs_with_text=[{"uid": "d1"}],
    )
    assert context.hinted_prompt == "q"
    assert context.routing_context == "1:"
    assert context.session_id == "tid"
    assert context.selected_doc_uid_for_logging == "d1"
    assert context.run_id.startswith("run-")


def test_apply_turn_result():
    warnings: list[str] = []
    stored_metrics = {}
    stored_message = {}
    payload = apply_turn_result(
        logger=SimpleNamespace(warning=lambda msg, exc: warnings.append(f"{msg}:{exc}")),
        user_uuid="u1",
        project_uid="p1",
        session_uid="s1",
        project_name="P",
        conversation_key="p1:s1",
        prompt="q",
        turn_result={
            "answer": "ans",
            "policy_decision": {"plan_enabled": False},
            "team_execution": {"rounds": 0},
            "trace_payload": [{"content": "x"}],
            "evidence_items": [{"text": "e"}],
            "mindmap_data": {"name": "root"},
            "method_compare_data": None,
            "run_latency_ms": 12.0,
            "team_rounds": 0,
            "phase_path": "phase",
        },
        session_state={},
        extract_skill_context_texts_from_trace_fn=lambda _trace: ["ctx"],
        append_skill_context_texts_fn=lambda **_kwargs: None,
        get_doc_metrics_fn=lambda _key: {"turns": 1},
        record_query_metrics_fn=lambda *_args, **_kwargs: {"turns": 2},
        store_turn_metrics_fn=lambda **kwargs: stored_metrics.update(kwargs),
        append_assistant_turn_message_fn=lambda **kwargs: stored_message.update(kwargs),
        persist_turn_memory_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("x")),
        infer_output_type_fn=lambda _prompt, _mindmap: "mindmap",
        serialize_output_content_fn=lambda **_kwargs: "serialized",
        resolve_archive_target_fn=lambda **_kwargs: ("d1", "docA"),
        scope_docs_with_text=[{"uid": "d1", "file_name": "docA"}],
    )
    assert payload.output_type == "mindmap"
    assert payload.serialized_content == "serialized"
    assert payload.doc_uid == "d1"
    assert payload.doc_name == "docA"
    assert stored_metrics["conversation_key"] == "p1:s1"
    assert stored_message["answer"] == "ans"
    assert warnings


