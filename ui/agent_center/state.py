from collections.abc import Callable, MutableMapping
from logging import Logger
from typing import Any


def has_cached_agent_session(
    project_uid: str,
    session_uid: str,
    scope_signature: str,
    *,
    session_state: MutableMapping[str, Any],
    build_session_key_fn: Callable[..., str],
    mode_leader: str,
    has_cached_agent_session_state_fn: Callable[..., bool],
) -> bool:
    return has_cached_agent_session_state_fn(
        session_state=session_state,
        build_session_key_fn=build_session_key_fn,
        mode_leader=mode_leader,
        project_uid=project_uid,
        session_uid=session_uid,
        scope_signature=scope_signature,
    )


def load_document_text(
    selected_uid: str,
    file_path: str,
    *,
    st: Any,
    logger: Logger,
    session_state: MutableMapping[str, Any],
    load_document_text_state_fn: Callable[..., tuple[str | None, str, str | None]],
    load_cached_extraction_fn: Callable[..., str | None],
    extract_document_payload_fn: Callable[[str], tuple[str, str | None]],
    save_cached_extraction_fn: Callable[..., None],
) -> tuple[str | None, str]:
    def _extract_with_spinner(path: str):
        with st.spinner("正在解析文档内容..."):
            return extract_document_payload_fn(path)

    text, source, error = load_document_text_state_fn(
        session_state=session_state,
        logger=logger,
        selected_uid=selected_uid,
        file_path=file_path,
        load_cached_extraction_fn=load_cached_extraction_fn,
        extract_document_fn=_extract_with_spinner,
        save_cached_extraction_fn=save_cached_extraction_fn,
    )
    if error:
        st.error(error)
        return None, "error"
    return text, source


def clear_project_runtime(
    project_uid: str,
    *,
    session_state: MutableMapping[str, Any],
    mode_leader: str,
    clear_project_runtime_state_fn: Callable[..., None],
) -> None:
    clear_project_runtime_state_fn(
        session_state=session_state,
        project_uid=project_uid,
        mode_leader=mode_leader,
    )


def ensure_agent_runtime(
    project_uid: str,
    session_uid: str,
    project_name: str,
    scope_docs: list[dict],
    scope_signature: str,
    *,
    session_state: MutableMapping[str, Any],
    logger: Logger,
    mode_leader: str,
    build_session_key_fn: Callable[..., str],
    clear_project_runtime_fn: Callable[[str], None],
    normalize_evidence_items_fn: Callable[[Any], list[dict[str, Any]]],
    get_user_api_key_fn: Callable[[], str | None],
    get_user_model_name_fn: Callable[[], str | None],
    get_user_base_url_fn: Callable[[], str | None],
    get_user_policy_router_model_name_fn: Callable[[], str | None],
    get_user_policy_router_base_url_fn: Callable[[], str | None],
    get_user_policy_router_api_key_fn: Callable[[], str | None],
    create_chat_model_fn: Callable[..., Any],
    create_project_evidence_retriever_fn: Callable[..., Any],
    create_leader_session_fn: Callable[..., Any],
    ensure_agent_runtime_state_fn: Callable[..., None],
) -> None:
    ensure_agent_runtime_state_fn(
        session_state=session_state,
        logger=logger,
        project_uid=project_uid,
        session_uid=session_uid,
        project_name=project_name,
        scope_docs=scope_docs,
        scope_signature=scope_signature,
        mode_leader=mode_leader,
        build_session_key_fn=build_session_key_fn,
        clear_project_runtime_fn=clear_project_runtime_fn,
        normalize_evidence_items_fn=normalize_evidence_items_fn,
        get_user_api_key_fn=get_user_api_key_fn,
        get_user_model_name_fn=get_user_model_name_fn,
        get_user_base_url_fn=get_user_base_url_fn,
        get_user_policy_router_model_name_fn=get_user_policy_router_model_name_fn,
        get_user_policy_router_base_url_fn=get_user_policy_router_base_url_fn,
        get_user_policy_router_api_key_fn=get_user_policy_router_api_key_fn,
        create_chat_model_fn=create_chat_model_fn,
        create_project_evidence_retriever_fn=create_project_evidence_retriever_fn,
        create_leader_session_fn=create_leader_session_fn,
    )


def prepare_agent_session(
    project_uid: str,
    session_uid: str,
    project_name: str,
    scope_docs: list[dict],
    scope_signature: str,
    *,
    st: Any,
    logger: Logger,
    has_cached_session_fn: Callable[[str, str, str], bool],
    ensure_agent_runtime_fn: Callable[[str, str, str, list[dict], str], None],
    prepare_agent_session_state_fn: Callable[..., None],
) -> None:
    def _run_with_spinner(run_fn):
        with st.spinner("正在构建项目级 RAG 索引（首次会自动下载模型）..."):
            run_fn()

    prepare_agent_session_state_fn(
        logger=logger,
        has_cached_session_fn=has_cached_session_fn,
        ensure_agent_runtime_fn=ensure_agent_runtime_fn,
        cached_caption_fn=lambda: st.caption("项目级 RAG 索引已存在，已复用。"),
        build_captioned_fn=_run_with_spinner,
        project_uid=project_uid,
        session_uid=session_uid,
        project_name=project_name,
        scope_docs=scope_docs,
        scope_signature=scope_signature,
    )


def update_context_usage(
    project_uid: str,
    conversation_key: str,
    *,
    st: Any,
    build_context_usage_snapshot_fn: Callable[..., dict[str, Any]],
    extract_skill_context_texts_from_trace_fn: Callable[..., list[str]],
    update_context_usage_state_fn: Callable[..., None],
) -> None:
    update_context_usage_state_fn(
        st=st,
        build_context_usage_snapshot_fn=build_context_usage_snapshot_fn,
        project_uid=project_uid,
        conversation_key=conversation_key,
        extract_skill_context_texts_from_trace_fn=extract_skill_context_texts_from_trace_fn,
    )


def apply_auto_compact(
    project_uid: str,
    session_uid: str,
    user_uuid: str,
    conversation_key: str,
    *,
    st: Any,
    logger: Logger,
    should_trigger_auto_compact_fn: Callable[..., bool],
    auto_compact_messages_fn: Callable[..., Any],
    build_openai_compatible_chat_model_fn: Callable[..., Any],
    get_user_api_key_fn: Callable[[], str | None],
    get_user_model_name_fn: Callable[[], str | None],
    get_user_base_url_fn: Callable[[], str | None],
    save_project_session_compact_memory_fn: Callable[..., None],
    persist_active_conversation_fn: Callable[..., None],
    apply_auto_compact_state_fn: Callable[..., str],
) -> str:
    return apply_auto_compact_state_fn(
        st=st,
        logger=logger,
        should_trigger_auto_compact_fn=should_trigger_auto_compact_fn,
        auto_compact_messages_fn=auto_compact_messages_fn,
        build_openai_compatible_chat_model_fn=build_openai_compatible_chat_model_fn,
        get_user_api_key_fn=get_user_api_key_fn,
        get_user_model_name_fn=get_user_model_name_fn,
        get_user_base_url_fn=get_user_base_url_fn,
        save_project_session_compact_memory_fn=save_project_session_compact_memory_fn,
        persist_active_conversation_fn=persist_active_conversation_fn,
        project_uid=project_uid,
        session_uid=session_uid,
        user_uuid=user_uuid,
        conversation_key=conversation_key,
    )
