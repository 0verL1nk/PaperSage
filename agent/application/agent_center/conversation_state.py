def persist_active_conversation(
    *,
    st,
    save_project_session_messages_fn,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    conversation_key: str,
) -> None:
    messages = st.session_state.get("agent_messages", [])
    if not isinstance(messages, list):
        return
    save_project_session_messages_fn(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=user_uuid,
        messages=messages,
    )
    messages_map = st.session_state.get("paper_project_messages", {})
    if not isinstance(messages_map, dict):
        messages_map = {}
    messages_map[conversation_key] = messages
    st.session_state.paper_project_messages = messages_map


def ensure_conversation_messages(
    *,
    st,
    list_project_session_messages_fn,
    persist_active_conversation_fn,
    user_uuid: str,
    project_uid: str,
    project_name: str,
    session_uid: str,
    conversation_key: str,
    scope_docs_count: int,
) -> None:
    messages_map = st.session_state.get("paper_project_messages", {})
    if not isinstance(messages_map, dict):
        messages_map = {}

    cached = messages_map.get(conversation_key)
    if isinstance(cached, list) and cached:
        st.session_state.agent_messages = cached
        return

    persisted = list_project_session_messages_fn(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=user_uuid,
    )
    if persisted:
        messages_map[conversation_key] = persisted
        st.session_state.paper_project_messages = messages_map
        st.session_state.agent_messages = persisted
        return

    bootstrap = [
        {
            "role": "assistant",
            "content": (
                f"已加载项目《{project_name}》，当前检索范围 {scope_docs_count} 篇文档。"
                " 工作流将按问题自动路由。"
            ),
        }
    ]
    messages_map[conversation_key] = bootstrap
    st.session_state.paper_project_messages = messages_map
    st.session_state.agent_messages = bootstrap
    persist_active_conversation_fn(
        user_uuid=user_uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        conversation_key=conversation_key,
    )


def ensure_compact_summary(
    *,
    st,
    get_project_session_compact_memory_fn,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    conversation_key: str,
) -> None:
    summary_map = st.session_state.get("paper_project_compact_summaries", {})
    if not isinstance(summary_map, dict):
        summary_map = {}
    if conversation_key in summary_map:
        return

    compact_state = get_project_session_compact_memory_fn(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=user_uuid,
    )
    summary_map[conversation_key] = str(compact_state.get("compact_summary") or "")
    st.session_state.paper_project_compact_summaries = summary_map


def update_context_usage(
    *,
    st,
    build_context_usage_snapshot_fn,
    project_uid: str,
    conversation_key: str,
) -> None:
    compact_summaries = st.session_state.get("paper_project_compact_summaries", {})
    tool_specs_map = st.session_state.get("paper_project_tool_specs", {})
    tool_specs = tool_specs_map.get(project_uid, [])
    skill_texts_map = st.session_state.get("paper_project_skill_context_texts", {})
    skill_context_texts = skill_texts_map.get(conversation_key, [])
    usage_map = st.session_state.get("paper_project_context_usage", {})
    usage_map[conversation_key] = build_context_usage_snapshot_fn(
        messages=st.session_state.get("agent_messages", []),
        compact_summary=str(compact_summaries.get(conversation_key, "") or ""),
        tool_specs=tool_specs if isinstance(tool_specs, list) else [],
        skill_context_texts=(
            skill_context_texts if isinstance(skill_context_texts, list) else []
        ),
    )
    st.session_state.paper_project_context_usage = usage_map


def apply_auto_compact(
    *,
    st,
    logger,
    should_trigger_auto_compact_fn,
    auto_compact_messages_fn,
    build_openai_compatible_chat_model_fn,
    get_user_api_key_fn,
    get_user_model_name_fn,
    get_user_base_url_fn,
    save_project_session_compact_memory_fn,
    persist_active_conversation_fn,
    project_uid: str,
    session_uid: str,
    user_uuid: str,
    conversation_key: str,
) -> str:
    summary_map = st.session_state.get("paper_project_compact_summaries", {})
    current_summary = str(summary_map.get(conversation_key, "") or "")
    llm = None
    messages = st.session_state.get("agent_messages", [])
    if should_trigger_auto_compact_fn(messages):
        llm = st.session_state.get("paper_compactor_llm")
        if llm is None:
            api_key = get_user_api_key_fn()
            model_name = get_user_model_name_fn()
            base_url = get_user_base_url_fn()
            if api_key and model_name:
                llm = build_openai_compatible_chat_model_fn(
                    api_key=api_key,
                    model_name=model_name,
                    base_url=base_url,
                    temperature=0,
                )
                st.session_state.paper_compactor_llm = llm

    result = auto_compact_messages_fn(
        messages,
        current_summary=current_summary,
        llm=llm,
    )
    st.session_state.agent_messages = result.messages
    messages_map = st.session_state.get("paper_project_messages", {})
    if not isinstance(messages_map, dict):
        messages_map = {}
    messages_map[conversation_key] = result.messages
    st.session_state.paper_project_messages = messages_map
    summary_map[conversation_key] = result.summary
    st.session_state.paper_project_compact_summaries = summary_map
    save_project_session_compact_memory_fn(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=user_uuid,
        compact_summary=result.summary,
    )
    persist_active_conversation_fn(
        user_uuid=user_uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        conversation_key=conversation_key,
    )
    if result.compacted:
        logger.info(
            "Auto compact applied: project_uid=%s session_uid=%s source_messages=%s tokens_before=%s tokens_after=%s llm=%s anchors=%s",
            project_uid,
            session_uid,
            result.source_message_count,
            result.source_token_estimate,
            result.compacted_token_estimate,
            result.used_llm,
            result.anchor_count,
        )
        mode = "LLM" if result.used_llm else "Heuristic"
        st.caption(
            "已执行自动压缩："
            f"{result.source_token_estimate} -> {result.compacted_token_estimate} tokens"
            f" | mode={mode} | anchors={result.anchor_count}"
        )
    return str(result.summary)
