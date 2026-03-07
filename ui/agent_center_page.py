def run_agent_center_page() -> None:
    import json
    import logging
    import os
    
    import streamlit as st
    import streamlit.components.v1 as components
    
    DEBUG_MODE = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}
    
    from agent.adapters import (
        count_session_messages_for_project,
        create_chat_model,
        create_leader_session,
        create_session_for_project,
        create_project_evidence_retriever,
        delete_session_for_project,
        ensure_default_session_for_project,
        extract_document_payload,
        list_project_files_for_user,
        list_session_messages_page_for_project,
        list_session_messages_for_project,
        list_sessions_for_project,
        list_user_files,
        list_user_projects,
        load_cached_extraction,
        read_user_api_key,
        read_user_base_url,
        read_user_model_name,
        read_user_policy_router_api_key,
        read_user_policy_router_base_url,
        read_user_policy_router_model_name,
        save_output,
        save_cached_extraction,
        save_session_messages_for_project,
        update_session_for_project,
    )
    from agent.agent_center_runner import execute_assistant_turn
    from agent.context_governance import (
        auto_compact_messages,
        build_context_usage_snapshot,
        extract_skill_context_texts_from_trace,
        inject_compact_summary,
        should_trigger_auto_compact,
    )
    from agent.memory.store import (
        get_project_session_compact_memory,
        save_project_session_compact_memory,
        search_project_memory_items,
    )
    from agent.memory.policy import (
        inject_long_term_memory,
    )
    from agent.logging_utils import configure_application_logging, logging_context
    from agent.metrics import record_query_metrics
    from agent.session_state import initialize_agent_center_session_state
    from agent.application.agent_center import (
        apply_turn_result,
        apply_auto_compact as apply_auto_compact_state,
        append_assistant_turn_message,
        append_skill_context_texts,
        build_hinted_prompt,
        build_turn_execution_context,
        build_routing_context,
        build_scope_cache_caption,
        build_session_maps,
        build_session_preview,
        clear_turn_lock,
        clear_project_runtime as clear_project_runtime_state,
        conversation_key as build_conversation_key,
        create_and_select_session,
        delete_and_select_next_session,
        drop_agent_session_cache as drop_agent_session_cache_state,
        drop_conversation_cache as drop_conversation_cache_state,
        ensure_compact_summary as ensure_compact_summary_state,
        ensure_conversation_messages as ensure_conversation_messages_state,
        get_history_paging_state,
        ensure_agent_runtime as ensure_agent_runtime_state,
        ensure_project_sessions,
        enqueue_user_turn,
        gate_prompt_and_enqueue,
        format_session_option,
        has_cached_agent_session as has_cached_agent_session_state,
        load_scope_docs_with_text,
        load_document_text as load_document_text_state,
        load_more_conversation_messages as load_more_conversation_messages_state,
        persist_active_conversation as persist_active_conversation_state,
        persist_turn_memory,
        prepare_scope_runtime,
        prepare_agent_session as prepare_agent_session_state,
        resolve_active_prompt,
        resolve_archive_target,
        resolve_current_session_uid,
        resolve_runtime_session_id,
        resolve_selected_doc_uid_for_logging,
        scope_signature as build_scope_signature,
        serialize_output_content,
        session_key as build_session_key,
        should_allow_delete_session,
        store_turn_metrics,
        update_selected_session_map,
        update_context_usage as update_context_usage_state,
        validate_runtime_prerequisites,
        with_language_hint,
    )
    from ui.project_workspace import (
        build_project_doc_count_map,
        inject_workspace_styles,
        render_project_context_hint,
        render_workspace_status_bar,
        select_project_sidebar,
        select_scope_documents_drawer,
    )
    from ui.agent_center_sidebar import (
        render_pinned_human_requests_panel,
        render_pinned_todo_panel,
    )
    from ui.theme import inject_global_theme
    from agent.ui_helpers import (
        _get_doc_metrics,
        _infer_output_type,
        _normalize_evidence_items,
        _render_chat_history,
        _render_context_usage,
        _render_output_archive,
        _render_workflow_metrics,
    )
    from utils.utils import (
        apply_user_runtime_tuning_env,
        detect_language,
        ensure_local_user,
        ensure_default_project_for_user,
        init_database,
    )
    
    configure_application_logging(debug_mode=DEBUG_MODE, default_level="INFO")
    logger = logging.getLogger("llm_app.agent_center")
    MODE_LEADER = "leader"
    
    
    st.set_page_config(page_title="Agent 中心", page_icon="🤖")
    st.title("🤖 Agent 中心")
    inject_global_theme()
    init_database("./database.sqlite")
    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"
    ensure_local_user(st.session_state["uuid"], db_name="./database.sqlite")
    ensure_default_project_for_user(
        st.session_state["uuid"],
        db_name="./database.sqlite",
        sync_existing_files=False,
    )
    
    if "files" not in st.session_state:
        st.session_state["files"] = []
    if "projects" not in st.session_state:
        st.session_state["projects"] = []
    
    
    def _load_files_from_db() -> None:
        raw_files = list_user_files(uuid=st.session_state["uuid"])
        st.session_state["files"] = []
        for file in raw_files:
            st.session_state["files"].append(
                {
                    "file_path": file["file_path"],
                    "file_name": file["file_name"],
                    "uid": file["uid"],
                    "created_at": file["created_at"],
                }
            )
        logger.debug("Loaded file list from DB: count=%s", len(st.session_state["files"]))

    
    def _load_projects_from_db() -> None:
        st.session_state["projects"] = list_user_projects(
            uuid=st.session_state["uuid"],
            include_archived=False,
        )
    
    
    def _drop_agent_session_cache(project_uid: str, session_uid: str) -> None:
        drop_agent_session_cache_state(
            session_state=st.session_state,
            build_session_key_fn=build_session_key,
            mode_leader=MODE_LEADER,
            project_uid=project_uid,
            session_uid=session_uid,
        )
    
    
    def _drop_conversation_cache(project_uid: str, session_uid: str) -> None:
        drop_conversation_cache_state(
            session_state=st.session_state,
            build_conversation_key_fn=build_conversation_key,
            project_uid=project_uid,
            session_uid=session_uid,
        )
    
    
    def _persist_active_conversation(
        *,
        user_uuid: str,
        project_uid: str,
        session_uid: str,
        conversation_key: str,
    ) -> None:
        persist_active_conversation_state(
            st=st,
            save_project_session_messages_fn=save_session_messages_for_project,
            list_project_session_messages_fn=list_session_messages_for_project,
            user_uuid=user_uuid,
            project_uid=project_uid,
            session_uid=session_uid,
            conversation_key=conversation_key,
        )
    
    
    def _ensure_conversation_messages(
        *,
        user_uuid: str,
        project_uid: str,
        project_name: str,
        session_uid: str,
        conversation_key: str,
        scope_docs_count: int,
    ) -> None:
        ensure_conversation_messages_state(
            st=st,
            list_project_session_messages_fn=list_session_messages_for_project,
            list_project_session_messages_page_fn=list_session_messages_page_for_project,
            count_project_session_messages_fn=count_session_messages_for_project,
            persist_active_conversation_fn=_persist_active_conversation,
            user_uuid=user_uuid,
            project_uid=project_uid,
            project_name=project_name,
            session_uid=session_uid,
            conversation_key=conversation_key,
            scope_docs_count=scope_docs_count,
        )


    def _scroll_chat_to_bottom() -> None:
        components.html(
            """
            <script>
            const root = window.parent.document;
            const selectors = ['section.main', '[data-testid="stAppViewContainer"]'];
            let target = null;
            for (const sel of selectors) {
              const el = root.querySelector(sel);
              if (el) { target = el; break; }
            }
            if (target) {
              target.scrollTo({ top: target.scrollHeight, behavior: 'auto' });
            } else {
              window.parent.scrollTo(0, document.body.scrollHeight);
            }
            </script>
            """,
            height=0,
        )


    def _inject_auto_load_more_on_scroll(conversation_key: str) -> None:
        escaped_key = conversation_key.replace("\\", "\\\\").replace("'", "\\'")
        components.html(
            f"""
            <script>
            const parentWin = window.parent;
            const parentDoc = parentWin.document;
            if (!parentWin.__agentHistoryAutoLoader) {{
              parentWin.__agentHistoryAutoLoader = {{
                key: '',
                lastClickAt: 0,
                intervalId: null,
              }};
            }}
            const state = parentWin.__agentHistoryAutoLoader;
            state.key = '{escaped_key}';
            const findScrollHost = () =>
              parentDoc.querySelector('section.main') ||
              parentDoc.querySelector('[data-testid="stAppViewContainer"]');
            const findLoadButton = () =>
              Array.from(parentDoc.querySelectorAll('button')).find(
                (btn) => (btn.innerText || '').trim() === '加载更早对话'
              );
            if (!state.intervalId) {{
              state.intervalId = parentWin.setInterval(() => {{
                const host = findScrollHost();
                const button = findLoadButton();
                if (!host || !button || button.disabled) {{
                  return;
                }}
                const now = Date.now();
                if (host.scrollTop <= 64 && now - state.lastClickAt > 1200) {{
                  state.lastClickAt = now;
                  button.click();
                }}
              }}, 280);
            }}
            </script>
            """,
            height=0,
        )


    def _disable_auto_load_more_on_scroll() -> None:
        components.html(
            """
            <script>
            const parentWin = window.parent;
            const state = parentWin.__agentHistoryAutoLoader;
            if (state && state.intervalId) {
              parentWin.clearInterval(state.intervalId);
              state.intervalId = null;
            }
            </script>
            """,
            height=0,
        )
    
    
    def _ensure_compact_summary(
        *,
        user_uuid: str,
        project_uid: str,
        session_uid: str,
        conversation_key: str,
    ) -> None:
        ensure_compact_summary_state(
            st=st,
            get_project_session_compact_memory_fn=get_project_session_compact_memory,
            user_uuid=user_uuid,
            project_uid=project_uid,
            session_uid=session_uid,
            conversation_key=conversation_key,
        )
    
    
    def _render_project_session_sidebar(
        *,
        user_uuid: str,
        project_uid: str,
        disabled: bool = False,
    ) -> dict[str, str]:
        sessions = ensure_project_sessions(
            list_sessions_fn=list_sessions_for_project,
            ensure_default_session_fn=ensure_default_session_for_project,
            project_uid=project_uid,
            user_uuid=user_uuid,
        )
        if not sessions:
            raise ValueError("会话初始化失败")

        by_uid, ordered_uids = build_session_maps(sessions)
        selected_map = st.session_state.get("agent_project_selected_sessions", {})
        current_uid = resolve_current_session_uid(
            selected_map=selected_map,
            project_uid=project_uid,
            by_uid=by_uid,
            ordered_uids=ordered_uids,
        )

        selector_key = f"agent_project_session_selector_{project_uid}"
        if str(st.session_state.get(selector_key) or "") != current_uid:
            st.session_state[selector_key] = current_uid
        selected_uid = st.selectbox(
            "当前会话",
            options=ordered_uids,
            format_func=lambda uid: format_session_option(by_uid.get(uid, {})),
            key=selector_key,
            disabled=disabled,
        )
        st.session_state.agent_project_selected_sessions = update_selected_session_map(
            selected_map=selected_map,
            project_uid=project_uid,
            selected_uid=selected_uid,
        )
        selected = by_uid[selected_uid]

        action_cols = st.columns(2)
        if action_cols[0].button(
            "新建会话",
            key=f"agent_project_session_new_{project_uid}",
            disabled=disabled,
            use_container_width=True,
        ):
            new_name = f"会话 {len(sessions) + 1}"
            st.session_state.agent_project_selected_sessions = create_and_select_session(
                create_session_fn=create_session_for_project,
                selected_map=st.session_state.get("agent_project_selected_sessions", {}),
                project_uid=project_uid,
                user_uuid=user_uuid,
                session_name=new_name,
            )
            st.rerun()
            return {"session_uid": "", "session_name": ""}

        can_delete = should_allow_delete_session(sessions)
        if action_cols[1].button(
            "删除会话",
            key=f"agent_project_session_delete_{project_uid}",
            disabled=disabled or not can_delete,
            use_container_width=True,
        ):
            selected_map, _next_uid = delete_and_select_next_session(
                delete_session_fn=delete_session_for_project,
                list_sessions_fn=list_sessions_for_project,
                drop_agent_session_cache_fn=_drop_agent_session_cache,
                drop_conversation_cache_fn=_drop_conversation_cache,
                selected_map=st.session_state.get("agent_project_selected_sessions", {}),
                project_uid=project_uid,
                user_uuid=user_uuid,
                selected_uid=selected_uid,
            )
            st.session_state.agent_project_selected_sessions = selected_map
            st.rerun()
            return {"session_uid": "", "session_name": ""}
    
        with st.expander("会话设置", expanded=False):
            rename_key = f"agent_project_session_rename_{project_uid}_{selected_uid}"
            if rename_key not in st.session_state:
                st.session_state[rename_key] = str(selected.get("session_name") or "")
            pin_key = f"agent_project_session_pin_{project_uid}_{selected_uid}"
            if pin_key not in st.session_state:
                st.session_state[pin_key] = bool(selected.get("is_pinned"))
            st.text_input("会话名称", key=rename_key, disabled=disabled)
            st.checkbox("置顶会话", key=pin_key, disabled=disabled)
            if st.button(
                "保存会话设置",
                key=f"agent_project_session_save_{project_uid}_{selected_uid}",
                disabled=disabled,
                use_container_width=True,
            ):
                update_session_for_project(
                    session_uid=selected_uid,
                    project_uid=project_uid,
                    uuid=user_uuid,
                    session_name=str(st.session_state.get(rename_key) or ""),
                    is_pinned=1 if st.session_state.get(pin_key) else 0,
                )
                st.rerun()
                return {"session_uid": "", "session_name": ""}
            st.caption(
                "创建时间："
                f"{str(selected.get('created_at') or '-')}"
                " ｜ 最近更新："
                f"{str(selected.get('updated_at') or '-')}"
            )

        preview = build_session_preview(selected)
        if preview:
            st.caption(f"最近一条：{preview}")
    
        return {
            "session_uid": selected_uid,
            "session_name": str(selected.get("session_name") or "未命名会话"),
        }
    
    
    def _has_cached_agent_session(project_uid: str, session_uid: str, scope_signature: str) -> bool:
        return has_cached_agent_session_state(
            session_state=st.session_state,
            build_session_key_fn=build_session_key,
            mode_leader=MODE_LEADER,
            project_uid=project_uid,
            session_uid=session_uid,
            scope_signature=scope_signature,
        )
    
    
    def _load_document_text(selected_uid: str, file_path: str) -> tuple[str | None, str]:
        def _extract_with_spinner(path: str):
            with st.spinner("正在解析文档内容..."):
                return extract_document_payload(path)

        text, source, error = load_document_text_state(
            session_state=st.session_state,
            logger=logger,
            selected_uid=selected_uid,
            file_path=file_path,
            load_cached_extraction_fn=load_cached_extraction,
            extract_document_fn=_extract_with_spinner,
            save_cached_extraction_fn=save_cached_extraction,
        )
        if error:
            st.error(error)
            return None, "error"
        return text, source
    
    
    def _clear_project_runtime(project_uid: str) -> None:
        clear_project_runtime_state(
            session_state=st.session_state,
            project_uid=project_uid,
            mode_leader=MODE_LEADER,
        )
    
    
    def _ensure_agent(
        project_uid: str,
        session_uid: str,
        project_name: str,
        scope_docs: list[dict],
        scope_signature: str,
    ) -> None:
        ensure_agent_runtime_state(
            session_state=st.session_state,
            logger=logger,
            project_uid=project_uid,
            session_uid=session_uid,
            project_name=project_name,
            scope_docs=scope_docs,
            scope_signature=scope_signature,
            mode_leader=MODE_LEADER,
            build_session_key_fn=build_session_key,
            clear_project_runtime_fn=_clear_project_runtime,
            normalize_evidence_items_fn=_normalize_evidence_items,
            get_user_api_key_fn=read_user_api_key,
            get_user_model_name_fn=read_user_model_name,
            get_user_base_url_fn=read_user_base_url,
            get_user_policy_router_model_name_fn=read_user_policy_router_model_name,
            get_user_policy_router_base_url_fn=read_user_policy_router_base_url,
            get_user_policy_router_api_key_fn=read_user_policy_router_api_key,
            create_chat_model_fn=create_chat_model,
            create_project_evidence_retriever_fn=create_project_evidence_retriever,
            create_leader_session_fn=create_leader_session,
        )
    
    
    def _prepare_agent_session(
        project_uid: str,
        session_uid: str,
        project_name: str,
        scope_docs: list[dict],
        scope_signature: str,
    ) -> None:
        def _run_with_spinner(run_fn):
            with st.spinner("正在构建项目级 RAG 索引（首次会自动下载模型）..."):
                run_fn()

        prepare_agent_session_state(
            logger=logger,
            has_cached_session_fn=_has_cached_agent_session,
            ensure_agent_runtime_fn=_ensure_agent,
            cached_caption_fn=lambda: st.caption("项目级 RAG 索引已存在，已复用。"),
            build_captioned_fn=_run_with_spinner,
            project_uid=project_uid,
            session_uid=session_uid,
            project_name=project_name,
            scope_docs=scope_docs,
            scope_signature=scope_signature,
        )
    
    
    def _update_context_usage(project_uid: str, conversation_key: str) -> None:
        update_context_usage_state(
            st=st,
            build_context_usage_snapshot_fn=build_context_usage_snapshot,
            project_uid=project_uid,
            conversation_key=conversation_key,
            extract_skill_context_texts_from_trace_fn=extract_skill_context_texts_from_trace,
        )
    
    
    def _apply_auto_compact(
        project_uid: str,
        session_uid: str,
        user_uuid: str,
        conversation_key: str,
    ) -> str:
        return apply_auto_compact_state(
            st=st,
            logger=logger,
            should_trigger_auto_compact_fn=should_trigger_auto_compact,
            auto_compact_messages_fn=auto_compact_messages,
            build_openai_compatible_chat_model_fn=create_chat_model,
            get_user_api_key_fn=read_user_api_key,
            get_user_model_name_fn=read_user_model_name,
            get_user_base_url_fn=read_user_base_url,
            save_project_session_compact_memory_fn=save_project_session_compact_memory,
            persist_active_conversation_fn=_persist_active_conversation,
            project_uid=project_uid,
            session_uid=session_uid,
            user_uuid=user_uuid,
            conversation_key=conversation_key,
        )
    
    
    def main():
        user_uuid = st.session_state.get("uuid", "local-user")
        applied_runtime_env = apply_user_runtime_tuning_env(user_uuid)
        if applied_runtime_env:
            logger.debug(
                "Applied user runtime env overrides: %s",
                sorted(applied_runtime_env.keys()),
            )
        turn_in_progress = bool(st.session_state.get("agent_turn_in_progress", False))
        pending_turn = st.session_state.get("agent_pending_turn")
        api_key = read_user_api_key()
        user_model = read_user_model_name()
        prerequisite_error = validate_runtime_prerequisites(
            api_key=api_key,
            model_name=user_model,
        )
        if prerequisite_error == "missing_api_key":
            st.warning("⚠️ 请先在“设置中心”页面配置您的 API Key")
            st.info('💡 请前往页面“设置中心（2_⚙️_设置中心）”完成配置后刷新。')
            logger.warning("Agent center blocked: missing API key")
            return
        if prerequisite_error == "missing_model_name":
            st.warning("⚠️ 请先在“设置中心”页面配置模型名称")
            st.info('💡 请前往页面“设置中心（2_⚙️_设置中心）”完成配置后刷新。')
            logger.warning("Agent center blocked: missing model name")
            return
    
        inject_workspace_styles()
        _load_projects_from_db()
        projects = st.session_state.get("projects", [])
        project_doc_count_map = build_project_doc_count_map(projects, user_uuid)
        with st.sidebar:
            st.markdown("## 项目工作台")
            st.caption("模型/API 配置统一在“设置中心”管理。")
            selected_project = select_project_sidebar(
                projects,
                project_doc_count_map,
                disabled=turn_in_progress,
            )
        if selected_project is None:
            st.write("### 暂无项目，请前往“项目中心”创建。")
            return
        selected_project_uid = str(selected_project["project_uid"])
        selected_project_name = str(selected_project["project_name"])
        with st.sidebar:
            st.markdown("### 项目会话")
            selected_session = _render_project_session_sidebar(
                user_uuid=user_uuid,
                project_uid=selected_project_uid,
                disabled=turn_in_progress,
            )
        selected_session_uid = str(selected_session.get("session_uid") or "")
        if not selected_session_uid:
            return
        selected_session_name = str(selected_session.get("session_name") or "未命名会话")
        conversation_key = build_conversation_key(selected_project_uid, selected_session_uid)
        st.session_state.agent_current_session_uid = selected_session_uid
        last_conversation_key = str(st.session_state.get("agent_last_conversation_key") or "")
        if last_conversation_key != conversation_key:
            st.session_state["agent_last_conversation_key"] = conversation_key
            st.session_state["agent_history_keep_position"] = False

        scoped_files = list_project_files_for_user(
            project_uid=selected_project_uid,
            uuid=st.session_state["uuid"],
            active_only=True,
        )
        with st.sidebar:
            scope_docs = select_scope_documents_drawer(
                scoped_files,
                selected_project_uid,
                disabled=turn_in_progress,
            )
            render_workspace_status_bar(
                project_name=selected_project_name,
                total_docs=project_doc_count_map.get(selected_project_uid, len(scoped_files)),
                selected_docs=len(scope_docs),
                turn_in_progress=turn_in_progress,
            )
        if not scope_docs:
            st.warning("当前项目还没有激活文档，请在文件中心或项目中心绑定文档。")
            return
    
        with logging_context(uid=user_uuid, project_uid=selected_project_uid):
            logger.debug(
                "Selected project: name=%s docs=%s",
                selected_project_name,
                len(scope_docs),
            )
    
        with logging_context(uid=user_uuid, project_uid=selected_project_uid):
            scope_runtime = prepare_scope_runtime(
                logger=logger,
                user_uuid=user_uuid,
                project_uid=selected_project_uid,
                project_name=selected_project_name,
                session_uid=selected_session_uid,
                conversation_key=conversation_key,
                scope_docs=scope_docs,
                load_scope_docs_with_text_fn=lambda *, scope_docs: load_scope_docs_with_text(
                    scope_docs=scope_docs,
                    load_document_text_fn=_load_document_text,
                ),
                build_scope_cache_caption_fn=build_scope_cache_caption,
                build_scope_signature_fn=build_scope_signature,
                has_cached_session_fn=_has_cached_agent_session,
                prepare_agent_session_fn=_prepare_agent_session,
                ensure_conversation_messages_fn=_ensure_conversation_messages,
                ensure_compact_summary_fn=_ensure_compact_summary,
                update_context_usage_fn=_update_context_usage,
            )
            if scope_runtime is None:
                return
            scope_docs_with_text = scope_runtime.scope_docs_with_text
            scope_signature = scope_runtime.scope_signature
            if scope_runtime.cache_caption:
                st.caption(scope_runtime.cache_caption)
    
        force_plan: bool | None = None
        force_team: bool | None = None
        with st.sidebar:
            with st.expander("执行策略", expanded=False):
                strategy_mode = st.radio(
                    "策略模式",
                    options=["自动", "手动"],
                    key=f"agent_strategy_mode_{selected_project_uid}_{selected_session_uid}",
                    disabled=turn_in_progress,
                    horizontal=True,
                )
                if strategy_mode == "手动":
                    force_plan = st.toggle(
                        "启用 Plan",
                        value=False,
                        key=f"agent_force_plan_{selected_project_uid}_{selected_session_uid}",
                        disabled=turn_in_progress,
                    )
                    force_team = st.toggle(
                        "启用 Team",
                        value=False,
                        key=f"agent_force_team_{selected_project_uid}_{selected_session_uid}",
                        disabled=turn_in_progress,
                    )
                    st.caption("手动模式下将覆盖自动策略判定。")
                else:
                    st.caption("自动模式：由策略路由器决定是否启用 Plan/Team。")
    
            st.markdown("### 会话信息")
            st.caption(f"当前会话：{selected_session_name}")
            _render_output_archive(selected_project_uid, disable_interaction=turn_in_progress)
            _render_workflow_metrics(conversation_key)
            _render_context_usage(conversation_key)
            render_pinned_todo_panel(project_uid=selected_project_uid, expanded=True)
            render_pinned_human_requests_panel(
                project_uid=selected_project_uid,
                chat_messages=st.session_state.get("agent_messages", []),
                expanded=False,
            )
            if turn_in_progress:
                st.info("正在生成回答，已临时锁定归档与文档切换，避免中断当前对话。")
    
        chat_messages = st.session_state.get("agent_messages", [])
        render_project_context_hint(selected_project_name, scope_docs)
        paging_state = get_history_paging_state(
            st=st,
            conversation_key=conversation_key,
        )
        if bool(paging_state.get("has_more_before")):
            loaded_count = int(paging_state.get("loaded_count", len(chat_messages)) or len(chat_messages))
            total_count = int(paging_state.get("total_count", loaded_count) or loaded_count)
            st.caption(f"会话记录：已加载 {loaded_count} / {total_count}")
            if st.button(
                "加载更早对话",
                key=f"agent_history_load_more_{conversation_key}",
                use_container_width=True,
                disabled=turn_in_progress,
            ):
                loaded = load_more_conversation_messages_state(
                    st=st,
                    list_project_session_messages_page_fn=list_session_messages_page_for_project,
                    user_uuid=user_uuid,
                    project_uid=selected_project_uid,
                    session_uid=selected_session_uid,
                    conversation_key=conversation_key,
                )
                st.session_state["agent_history_keep_position"] = loaded > 0
                st.rerun()
                return
            _inject_auto_load_more_on_scroll(conversation_key)
        else:
            _disable_auto_load_more_on_scroll()
        _render_chat_history(chat_messages)
        if not bool(st.session_state.get("agent_history_keep_position", False)):
            _scroll_chat_to_bottom()
        st.session_state["agent_history_keep_position"] = False

        prompt_input = st.chat_input("输入你的论文问题", disabled=turn_in_progress)
        prompt_gate = gate_prompt_and_enqueue(
            session_state=st.session_state,
            turn_in_progress=turn_in_progress,
            pending_turn=pending_turn,
            prompt_input=prompt_input,
            project_uid=selected_project_uid,
            session_uid=selected_session_uid,
            conversation_key=conversation_key,
            scope_signature=scope_signature,
            resolve_active_prompt_fn=resolve_active_prompt,
            clear_turn_lock_fn=clear_turn_lock,
            enqueue_user_turn_fn=enqueue_user_turn,
            persist_active_conversation_fn=_persist_active_conversation,
            user_uuid=user_uuid,
        )
        if prompt_gate.state == "idle":
            return
        if prompt_gate.state == "rerun":
            st.rerun()
            return
        prompt = str(prompt_gate.prompt or "")
    
        compact_summary = _apply_auto_compact(
            selected_project_uid,
            selected_session_uid,
            user_uuid,
            conversation_key,
        )
        turn_context = build_turn_execution_context(
            prompt=prompt,
            compact_summary=compact_summary,
            user_uuid=user_uuid,
            project_uid=selected_project_uid,
            session_state=st.session_state,
            build_routing_context_fn=build_routing_context,
            build_hinted_prompt_fn=lambda **kwargs: build_hinted_prompt(
                **kwargs,
                detect_language_fn=detect_language,
                with_language_hint_fn=with_language_hint,
                inject_compact_summary_fn=inject_compact_summary,
                search_project_memory_items_fn=search_project_memory_items,
                inject_long_term_memory_fn=inject_long_term_memory,
                memory_limit=4,
            ),
            resolve_runtime_session_id_fn=resolve_runtime_session_id,
            resolve_selected_doc_uid_for_logging_fn=resolve_selected_doc_uid_for_logging,
            scope_docs_with_text=scope_docs_with_text,
        )
        try:
            turn_result = execute_assistant_turn(
                prompt=prompt,
                hinted_prompt=turn_context.hinted_prompt,
                user_uuid=user_uuid,
                project_uid=selected_project_uid,
                selected_uid=turn_context.selected_doc_uid_for_logging,
                run_id=turn_context.run_id,
                session_id=turn_context.session_id,
                logger=logger,
                force_plan=force_plan,
                force_team=force_team,
                routing_context=turn_context.routing_context,
            )
        finally:
            clear_turn_lock(st.session_state)
        archive_payload = apply_turn_result(
            logger=logger,
            user_uuid=user_uuid,
            project_uid=selected_project_uid,
            session_uid=selected_session_uid,
            project_name=selected_project_name,
            conversation_key=conversation_key,
            prompt=prompt,
            turn_result=turn_result,
            session_state=st.session_state,
            extract_skill_context_texts_from_trace_fn=extract_skill_context_texts_from_trace,
            append_skill_context_texts_fn=append_skill_context_texts,
            get_doc_metrics_fn=_get_doc_metrics,
            record_query_metrics_fn=record_query_metrics,
            store_turn_metrics_fn=store_turn_metrics,
            append_assistant_turn_message_fn=append_assistant_turn_message,
            persist_turn_memory_fn=persist_turn_memory,
            infer_output_type_fn=_infer_output_type,
            serialize_output_content_fn=lambda **kwargs: serialize_output_content(
                **kwargs,
                json_dumps_fn=json.dumps,
            ),
            resolve_archive_target_fn=resolve_archive_target,
            scope_docs_with_text=scope_docs_with_text,
        )
        save_output(
            uuid=st.session_state.get("uuid", "local-user"),
            project_uid=selected_project_uid,
            session_uid=selected_session_uid,
            doc_uid=archive_payload.doc_uid,
            doc_name=archive_payload.doc_name,
            output_type=archive_payload.output_type,
            content=archive_payload.serialized_content,
        )
        logger.info(
            "Archived agent output: output_type=%s content_len=%s",
            archive_payload.output_type,
            len(archive_payload.serialized_content),
        )
    
        _persist_active_conversation(
            user_uuid=user_uuid,
            project_uid=selected_project_uid,
            session_uid=selected_session_uid,
            conversation_key=conversation_key,
        )
        _update_context_usage(selected_project_uid, conversation_key)
        st.rerun()
    
    
    initialize_agent_center_session_state()
    
    _load_files_from_db()
    
    if not st.session_state.files:
        st.write("### 暂无文档，请前往“文件中心”页面上传。")
    else:
        main()


if __name__ == "__main__":
    run_agent_center_page()
