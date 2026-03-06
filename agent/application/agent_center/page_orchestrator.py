from dataclasses import dataclass
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ScopeRuntimeResult:
    scope_docs_with_text: list[dict[str, Any]]
    scope_signature: str
    cache_caption: str


@dataclass(frozen=True)
class PromptGateResult:
    state: str
    prompt: str | None = None


@dataclass(frozen=True)
class TurnExecutionContext:
    hinted_prompt: str
    routing_context: str
    run_id: str
    session_id: str
    selected_doc_uid_for_logging: str


@dataclass(frozen=True)
class ArchivePayload:
    output_type: str
    serialized_content: str
    doc_uid: str | None
    doc_name: str


def prepare_scope_runtime(
    *,
    logger,
    user_uuid: str,
    project_uid: str,
    project_name: str,
    session_uid: str,
    conversation_key: str,
    scope_docs: list[dict[str, Any]],
    load_scope_docs_with_text_fn,
    build_scope_cache_caption_fn,
    build_scope_signature_fn,
    prepare_agent_session_fn,
    ensure_conversation_messages_fn,
    ensure_compact_summary_fn,
    update_context_usage_fn,
) -> ScopeRuntimeResult | None:
    scope_docs_with_text, cache_stats, failed_uid = load_scope_docs_with_text_fn(
        scope_docs=scope_docs,
    )
    if failed_uid:
        logger.warning("Scope document loading interrupted: uid=%s", failed_uid)
        return None

    scope_signature = build_scope_signature_fn(scope_docs_with_text)
    prepare_agent_session_fn(
        project_uid,
        session_uid,
        project_name,
        scope_docs_with_text,
        scope_signature,
    )
    ensure_conversation_messages_fn(
        user_uuid=user_uuid,
        project_uid=project_uid,
        project_name=project_name,
        session_uid=session_uid,
        conversation_key=conversation_key,
        scope_docs_count=len(scope_docs_with_text),
    )
    ensure_compact_summary_fn(
        user_uuid=user_uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        conversation_key=conversation_key,
    )
    update_context_usage_fn(project_uid, conversation_key)
    return ScopeRuntimeResult(
        scope_docs_with_text=scope_docs_with_text,
        scope_signature=scope_signature,
        cache_caption=str(build_scope_cache_caption_fn(cache_stats)),
    )


def gate_prompt_and_enqueue(
    *,
    session_state: dict[str, Any],
    turn_in_progress: bool,
    pending_turn: Any,
    prompt_input: str | None,
    project_uid: str,
    session_uid: str,
    conversation_key: str,
    scope_signature: str,
    resolve_active_prompt_fn,
    clear_turn_lock_fn,
    enqueue_user_turn_fn,
    persist_active_conversation_fn,
    user_uuid: str,
) -> PromptGateResult:
    prompt, prompt_state = resolve_active_prompt_fn(
        turn_in_progress=turn_in_progress,
        pending_turn=pending_turn,
        prompt_input=prompt_input,
        project_uid=project_uid,
        session_uid=session_uid,
        scope_signature=scope_signature,
    )
    if prompt_state == "mismatch_pending":
        clear_turn_lock_fn(session_state)
        return PromptGateResult(state="rerun")
    if prompt_state == "no_prompt":
        return PromptGateResult(state="idle")
    if prompt_state == "new_prompt":
        enqueue_user_turn_fn(
            session_state=session_state,
            prompt=str(prompt),
            project_uid=project_uid,
            session_uid=session_uid,
            conversation_key=conversation_key,
            scope_signature=scope_signature,
        )
        persist_active_conversation_fn(
            user_uuid=user_uuid,
            project_uid=project_uid,
            session_uid=session_uid,
            conversation_key=conversation_key,
        )
        return PromptGateResult(state="rerun")
    return PromptGateResult(state="ready", prompt=str(prompt))


def build_turn_execution_context(
    *,
    prompt: str,
    compact_summary: str,
    user_uuid: str,
    project_uid: str,
    session_state: dict[str, Any],
    build_routing_context_fn,
    build_hinted_prompt_fn,
    resolve_runtime_session_id_fn,
    resolve_selected_doc_uid_for_logging_fn,
    scope_docs_with_text: list[dict[str, Any]],
) -> TurnExecutionContext:
    routing_context = build_routing_context_fn(
        session_state.get("agent_messages", []),
        compact_summary,
    )
    hinted_prompt = build_hinted_prompt_fn(
        prompt=prompt,
        compact_summary=compact_summary,
        user_uuid=user_uuid,
        project_uid=project_uid,
    )
    runtime_config = session_state.get("paper_agent_runtime_config", {})
    session_id = resolve_runtime_session_id_fn(runtime_config)
    selected_doc_uid_for_logging = resolve_selected_doc_uid_for_logging_fn(
        scope_docs_with_text
    )
    return TurnExecutionContext(
        hinted_prompt=hinted_prompt,
        routing_context=routing_context,
        run_id=f"run-{uuid4().hex[:12]}",
        session_id=session_id,
        selected_doc_uid_for_logging=selected_doc_uid_for_logging,
    )


def apply_turn_result(
    *,
    logger,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    project_name: str,
    conversation_key: str,
    prompt: str,
    turn_result: dict[str, Any],
    session_state: dict[str, Any],
    extract_skill_context_texts_from_trace_fn,
    append_skill_context_texts_fn,
    get_doc_metrics_fn,
    record_query_metrics_fn,
    store_turn_metrics_fn,
    append_assistant_turn_message_fn,
    persist_turn_memory_fn,
    infer_output_type_fn,
    serialize_output_content_fn,
    resolve_archive_target_fn,
    scope_docs_with_text: list[dict[str, Any]],
) -> ArchivePayload:
    answer = turn_result["answer"]
    policy_decision = turn_result["policy_decision"]
    team_execution = turn_result["team_execution"]
    trace_payload = turn_result["trace_payload"]
    evidence_items = turn_result["evidence_items"]
    mindmap_data = turn_result["mindmap_data"]
    mindmap_html = turn_result.get("mindmap_html")
    mindmap_render_error = turn_result.get("mindmap_render_error")
    method_compare_data = turn_result["method_compare_data"]
    ask_human_requests = turn_result.get("ask_human_requests", [])
    run_latency_ms = float(turn_result["run_latency_ms"])
    team_rounds = int(turn_result["team_rounds"])
    phase_path = turn_result["phase_path"]

    traced_skill_texts = extract_skill_context_texts_from_trace_fn(trace_payload)
    append_skill_context_texts_fn(
        session_state=session_state,
        conversation_key=conversation_key,
        skill_texts=traced_skill_texts,
    )

    doc_metrics = get_doc_metrics_fn(conversation_key)
    updated_metrics = record_query_metrics_fn(
        doc_metrics,
        latency_ms=run_latency_ms,
        trace_payload=trace_payload,
        policy_decision=policy_decision if isinstance(policy_decision, dict) else None,
        team_execution=team_execution if isinstance(team_execution, dict) else None,
    )
    store_turn_metrics_fn(
        session_state=session_state,
        conversation_key=conversation_key,
        metrics=updated_metrics,
    )

    append_assistant_turn_message_fn(
        session_state=session_state,
        answer=answer,
        trace_payload=trace_payload,
        mindmap_data=mindmap_data,
        mindmap_html=mindmap_html,
        mindmap_render_error=mindmap_render_error,
        method_compare_data=method_compare_data,
        ask_human_requests=ask_human_requests if isinstance(ask_human_requests, list) else [],
        evidence_items=evidence_items,
        policy_decision=policy_decision,
        team_execution=team_execution,
        latency_ms=run_latency_ms,
        team_rounds=team_rounds,
        phase_path=phase_path,
    )
    try:
        persist_turn_memory_fn(
            user_uuid=user_uuid,
            project_uid=project_uid,
            session_uid=session_uid,
            prompt=prompt,
            answer=answer,
        )
    except Exception as exc:
        logger.warning("Persist turn memory failed: %s", exc)

    output_type = infer_output_type_fn(prompt, mindmap_data)
    serialized_content = serialize_output_content_fn(
        answer=answer,
        mindmap_data=mindmap_data,
    )
    archive_doc_uid, archive_doc_name = resolve_archive_target_fn(
        scope_docs_with_text=scope_docs_with_text,
        project_name=project_name,
    )
    return ArchivePayload(
        output_type=output_type,
        serialized_content=serialized_content,
        doc_uid=archive_doc_uid,
        doc_name=archive_doc_name,
    )
