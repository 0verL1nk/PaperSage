def run_agent_center_page() -> None:
    import json
    import logging
    import os
    from functools import partial

    import streamlit as st
    import streamlit.components.v1 as components
    from langgraph.checkpoint.sqlite import SqliteSaver

    DEBUG_MODE = os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}

    from agent.adapters import (
        apply_runtime_tuning_env_for_user,
        count_session_messages_for_project,
        create_chat_model,
        create_leader_session,
        create_project_evidence_retriever,
        create_session_for_project,
        delete_session_for_project,
        ensure_default_session_for_project,
        extract_document_payload,
        list_project_files_for_user,
        list_session_messages_for_project,
        list_session_messages_page_for_project,
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
        save_cached_extraction,
        save_output,
        save_session_messages_for_project,
        update_session_for_project,
    )
    from agent.application.agent_center import (
        append_assistant_turn_message,
        append_skill_context_texts,
        apply_turn_result,
        build_hinted_prompt,
        build_routing_context,
        build_runtime_deps_from_session_state,
        build_scope_cache_caption,
        build_session_maps,
        build_session_preview,
        build_turn_execution_context,
        clear_turn_lock,
        create_and_select_session,
        delete_and_select_next_session,
        enqueue_user_turn,
        ensure_project_sessions,
        format_session_option,
        gate_prompt_and_enqueue,
        get_history_paging_state,
        load_scope_docs_with_text,
        persist_turn_memory,
        prepare_scope_runtime,
        resolve_active_prompt,
        resolve_archive_target,
        resolve_current_session_uid,
        resolve_runtime_session_id,
        resolve_selected_doc_uid_for_logging,
        serialize_output_content,
        should_allow_delete_session,
        store_turn_metrics,
        update_selected_session_map,
        validate_runtime_prerequisites,
        with_language_hint,
    )
    from agent.application.agent_center import (
        clear_project_runtime as clear_project_runtime_state,
    )
    from agent.application.agent_center import (
        conversation_key as build_conversation_key,
    )
    from agent.application.agent_center import (
        drop_agent_session_cache as drop_agent_session_cache_state,
    )
    from agent.application.agent_center import (
        drop_conversation_cache as drop_conversation_cache_state,
    )
    from agent.application.agent_center import (
        ensure_agent_runtime as ensure_agent_runtime_state,
    )
    from agent.application.agent_center import (
        ensure_compact_summary as ensure_compact_summary_state,
    )
    from agent.application.agent_center import (
        ensure_conversation_messages as ensure_conversation_messages_state,
    )
    from agent.application.agent_center import (
        has_cached_agent_session as has_cached_agent_session_state,
    )
    from agent.application.agent_center import (
        load_document_text as load_document_text_state,
    )
    from agent.application.agent_center import (
        load_more_conversation_messages as load_more_conversation_messages_state,
    )
    from agent.application.agent_center import (
        persist_active_conversation as persist_active_conversation_state,
    )
    from agent.application.agent_center import (
        prepare_agent_session as prepare_agent_session_state,
    )
    from agent.application.agent_center import (
        scope_signature as build_scope_signature,
    )
    from agent.application.agent_center import (
        session_key as build_session_key,
    )
    from agent.application.agent_center import (
        update_context_usage as update_context_usage_state,
    )
    from agent.application.agent_center.facade import (
        AgentCenterTurnRequest,
        execute_agent_center_turn,
    )
    from agent.application.language import detect_language
    from agent.context_governance import (
        build_context_usage_snapshot,
        extract_skill_context_texts_from_trace,
    )
    from agent.logging_utils import configure_application_logging, logging_context
    from agent.memory.policy import (
        inject_long_term_memory,
    )
    from agent.memory.store import (
        get_project_session_compact_memory,
        search_project_memory_items,
    )
    from agent.metrics import record_query_metrics
    from agent.mindmap_renderer import render_mindmap_html_with_cli
    from agent.session_state import initialize_agent_center_session_state
    from agent.ui_helpers import (
        _get_doc_metrics,
        _infer_output_type,
        _normalize_evidence_items,
        _render_chat_history,
        _render_context_usage,
        _render_output_archive,
        _render_workflow_metrics,
    )
    from ui.agent_center.controller import (
        load_files_from_db,
        load_projects_from_db,
        render_chat_history_panel,
        render_project_session_sidebar,
        render_strategy_sidebar,
    )
    from ui.agent_center.state import (
        clear_project_runtime,
        ensure_agent_runtime,
        has_cached_agent_session,
        load_document_text,
        prepare_agent_session,
        update_context_usage,
    )
    from ui.agent_center_sidebar import (
        render_pinned_human_requests_panel,
        render_pinned_plan_panel,
        render_pinned_todo_panel,
    )
    from ui.agent_center_turn_view import build_status_event_line, render_turn_result
    from ui.page_bootstrap import bootstrap_page_context
    from ui.project_workspace import (
        build_project_doc_count_map,
        inject_workspace_styles,
        render_project_context_hint,
        render_workspace_status_bar,
        select_project_sidebar,
        select_scope_documents_drawer,
    )
    from ui.theme import inject_global_theme

    configure_application_logging(debug_mode=DEBUG_MODE, default_level="INFO")
    logger = logging.getLogger("llm_app.agent_center")
    MODE_LEADER = "leader"


    st.set_page_config(page_title="Agent 中心", page_icon="🤖")
    st.title("🤖 Agent 中心")
    inject_global_theme()
    bootstrap_page_context(
        session_state=st.session_state,
        db_name="./database.sqlite",
        ensure_default_project=True,
        sync_existing_files=False,
        state_defaults={
            "files": [],
            "projects": [],
        },
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


    has_cached_agent_session_fn = partial(
        has_cached_agent_session,
        session_state=st.session_state,
        build_session_key_fn=build_session_key,
        mode_leader=MODE_LEADER,
        has_cached_agent_session_state_fn=has_cached_agent_session_state,
    )

    clear_project_runtime_fn = partial(
        clear_project_runtime,
        session_state=st.session_state,
        mode_leader=MODE_LEADER,
        clear_project_runtime_state_fn=clear_project_runtime_state,
    )

    ensure_agent_runtime_fn = partial(
        ensure_agent_runtime,
        session_state=st.session_state,
        logger=logger,
        mode_leader=MODE_LEADER,
        build_session_key_fn=build_session_key,
        clear_project_runtime_fn=clear_project_runtime_fn,
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
        ensure_agent_runtime_state_fn=ensure_agent_runtime_state,
    )

    load_document_text_fn = partial(
        load_document_text,
        st=st,
        logger=logger,
        session_state=st.session_state,
        load_document_text_state_fn=load_document_text_state,
        load_cached_extraction_fn=load_cached_extraction,
        extract_document_payload_fn=extract_document_payload,
        save_cached_extraction_fn=save_cached_extraction,
    )

    prepare_agent_session_fn = partial(
        prepare_agent_session,
        st=st,
        logger=logger,
        has_cached_session_fn=has_cached_agent_session_fn,
        ensure_agent_runtime_fn=ensure_agent_runtime_fn,
        prepare_agent_session_state_fn=prepare_agent_session_state,
    )

    update_context_usage_fn = partial(
        update_context_usage,
        st=st,
        build_context_usage_snapshot_fn=build_context_usage_snapshot,
        extract_skill_context_texts_from_trace_fn=extract_skill_context_texts_from_trace,
        update_context_usage_state_fn=update_context_usage_state,
    )


    def main():
        user_uuid = st.session_state.get("uuid", "local-user")
        applied_runtime_env = apply_runtime_tuning_env_for_user(uuid=user_uuid)
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
        load_projects_from_db(
            session_state=st.session_state,
            user_uuid=user_uuid,
            list_user_projects_fn=list_user_projects,
        )
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
            selected_session = render_project_session_sidebar(
                st=st,
                user_uuid=user_uuid,
                project_uid=selected_project_uid,
                disabled=turn_in_progress,
                ensure_project_sessions_fn=ensure_project_sessions,
                list_sessions_fn=list_sessions_for_project,
                ensure_default_session_fn=ensure_default_session_for_project,
                build_session_maps_fn=build_session_maps,
                resolve_current_session_uid_fn=resolve_current_session_uid,
                format_session_option_fn=format_session_option,
                update_selected_session_map_fn=update_selected_session_map,
                create_and_select_session_fn=create_and_select_session,
                create_session_fn=create_session_for_project,
                delete_session_fn=delete_session_for_project,
                should_allow_delete_session_fn=should_allow_delete_session,
                delete_and_select_next_session_fn=delete_and_select_next_session,
                drop_agent_session_cache_fn=_drop_agent_session_cache,
                drop_conversation_cache_fn=_drop_conversation_cache,
                update_session_for_project_fn=update_session_for_project,
                build_session_preview_fn=build_session_preview,
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

            # Display current plan if exists
            current_plan = st.session_state.get("current_plan")
            if current_plan:
                st.markdown("### 📋 执行计划")
                with st.expander("查看计划详情", expanded=True):
                    goal = current_plan.get("goal", "")
                    description = current_plan.get("description", "")
                    if goal:
                        st.markdown(f"**目标:** {goal}")
                    if description:
                        st.markdown(f"**策略:**\n\n{description}")

            # Display current todos if exists
            current_todos = st.session_state.get("current_todos")
            if current_todos and isinstance(current_todos, list):
                st.markdown("### ✅ 任务列表")
                with st.expander("查看任务详情", expanded=True):
                    for idx, todo in enumerate(current_todos, 1):
                        if isinstance(todo, dict):
                            content = todo.get("content", "")
                            status = todo.get("status", "pending")
                            status_icon = {
                                "pending": "⬜",
                                "in_progress": "🟨",
                                "completed": "✅"
                            }.get(status, "⬜")
                            st.markdown(f"{idx}. {status_icon} {content}")

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
                    load_document_text_fn=load_document_text_fn,
                ),
                build_scope_cache_caption_fn=build_scope_cache_caption,
                build_scope_signature_fn=build_scope_signature,
                has_cached_session_fn=has_cached_agent_session_fn,
                prepare_agent_session_fn=prepare_agent_session_fn,
                ensure_conversation_messages_fn=_ensure_conversation_messages,
                ensure_compact_summary_fn=_ensure_compact_summary,
                update_context_usage_fn=update_context_usage_fn,
            )
            if scope_runtime is None:
                return
            scope_docs_with_text = scope_runtime.scope_docs_with_text
            scope_signature = scope_runtime.scope_signature
            if scope_runtime.cache_caption:
                st.caption(scope_runtime.cache_caption)

        chat_messages = st.session_state.get("agent_messages", [])
        render_strategy_sidebar(
            st=st,
            selected_project_uid=selected_project_uid,
            selected_session_uid=selected_session_uid,
            selected_session_name=selected_session_name,
            conversation_key=conversation_key,
            turn_in_progress=turn_in_progress,
            chat_messages=chat_messages,
            render_output_archive_fn=_render_output_archive,
            render_workflow_metrics_fn=_render_workflow_metrics,
            render_context_usage_fn=_render_context_usage,
            render_pinned_plan_panel_fn=render_pinned_plan_panel,
            render_pinned_todo_panel_fn=render_pinned_todo_panel,
            render_pinned_human_requests_panel_fn=render_pinned_human_requests_panel,
        )
        render_project_context_hint(selected_project_name, scope_docs)
        render_chat_history_panel(
            st=st,
            components=components,
            chat_messages=chat_messages,
            conversation_key=conversation_key,
            turn_in_progress=turn_in_progress,
            user_uuid=user_uuid,
            project_uid=selected_project_uid,
            session_uid=selected_session_uid,
            get_history_paging_state_fn=get_history_paging_state,
            load_more_conversation_messages_fn=lambda **kwargs: load_more_conversation_messages_state(
                **kwargs,
                list_project_session_messages_page_fn=list_session_messages_page_for_project,
            ),
            render_chat_history_fn=_render_chat_history,
        )

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

        # SummarizationMiddleware handles compression automatically
        turn_context = build_turn_execution_context(
            prompt=prompt,
            user_uuid=user_uuid,
            project_uid=selected_project_uid,
            session_state=st.session_state,
            build_routing_context_fn=build_routing_context,
            build_hinted_prompt_fn=lambda **kwargs: build_hinted_prompt(
                **kwargs,
                detect_language_fn=detect_language,
                with_language_hint_fn=with_language_hint,
                search_project_memory_items_fn=search_project_memory_items,
                inject_long_term_memory_fn=inject_long_term_memory,
                memory_limit=4,
            ),
            resolve_runtime_session_id_fn=resolve_runtime_session_id,
            resolve_selected_doc_uid_for_logging_fn=resolve_selected_doc_uid_for_logging,
            scope_docs_with_text=scope_docs_with_text,
        )

        try:
            with logging_context(
                uid=user_uuid,
                project_uid=selected_project_uid,
                doc_uid=turn_context.selected_doc_uid_for_logging,
                run_id=turn_context.run_id,
                session_id=turn_context.session_id,
            ):
                logger.info("Agent run started: prompt_len=%s", len(prompt))
                with st.chat_message("assistant"):
                    runtime_deps = build_runtime_deps_from_session_state(st.session_state)
                    event_count = [0]
                    with st.status("执行策略编排中...", expanded=True) as status:

                        def _on_event(item: dict[str, str]) -> None:
                            event_count[0] += 1
                            label, line = build_status_event_line(
                                event_index=event_count[0],
                                item=item,
                            )
                            status.update(label=label, state="running", expanded=True)
                            status.write(line)

                        turn_result = execute_agent_center_turn(
                            request=AgentCenterTurnRequest(
                                prompt=prompt,
                                hinted_prompt=turn_context.hinted_prompt,
                                routing_context=turn_context.routing_context,
                            ),
                            deps=runtime_deps,
                            on_event=_on_event,
                        )
                        status.update(label="执行完成", state="complete", expanded=False)

                    mindmap_data = turn_result["mindmap_data"]
                    mindmap_html = None
                    mindmap_render_error = None
                    if isinstance(mindmap_data, dict) and mindmap_data:
                        mindmap_html, mindmap_render_error = render_mindmap_html_with_cli(
                            mindmap_data,
                            title="思维导图",
                        )

                    render_turn_result(
                        answer=turn_result["answer"],
                        policy_decision=turn_result["policy_decision"],
                        team_execution=turn_result["team_execution"],
                        trace_payload=turn_result["trace_payload"],
                        evidence_items=turn_result["evidence_items"],
                        mindmap_data=mindmap_data,
                        mindmap_html=mindmap_html,
                        mindmap_render_error=mindmap_render_error,
                        method_compare_data=turn_result["method_compare_data"],
                        ask_human_requests=turn_result.get("ask_human_requests", []),
                        run_latency_ms=turn_result["run_latency_ms"],
                        phase_path=turn_result["phase_path"],
                    )

                # Store agent_plan data in session state
                agent_plan = turn_result.get("agent_plan")
                if agent_plan:
                    st.session_state["current_agent_plan"] = agent_plan
                    logger.info("Stored agent_plan to session_state: %s", agent_plan)

                # Store todos from turn_result in session state
                todos = turn_result.get("todos")
                if todos:
                    st.session_state["current_todos"] = todos
                    logger.info("Stored todos to session_state: count=%s", len(todos))

                logger.info(
                    "Agent run finished: latency_ms=%.2f trace_events=%s evidence_items=%s team_rounds=%s",
                    float(turn_result["run_latency_ms"]),
                    len(turn_result["trace_payload"]),
                    len(turn_result["evidence_items"]),
                    int(turn_result["team_execution"].get("rounds", 0)),
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

        # Sync messages from checkpointer to UI
        try:
            runtime_config = st.session_state.get("paper_agent_runtime_config", {})
            if runtime_config:
                checkpointer_db_path = os.getenv("CHECKPOINTER_DB_PATH", "./data/checkpoints.db")
                with SqliteSaver.from_conn_string(checkpointer_db_path) as checkpointer:
                    checkpoint = checkpointer.get(runtime_config)
                    if checkpoint and hasattr(checkpoint, "channel_values"):
                        messages = checkpoint.channel_values.get("messages", [])
                        if messages:
                            st.session_state["agent_messages"] = messages
        except Exception as e:
            logger.warning("Failed to sync messages from checkpointer: %s", e)

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
        update_context_usage_fn(selected_project_uid, conversation_key)
        st.rerun()


    initialize_agent_center_session_state()

    load_files_from_db(
        session_state=st.session_state,
        user_uuid=str(st.session_state.get("uuid") or "local-user"),
        list_user_files_fn=list_user_files,
        logger=logger,
    )

    if not st.session_state.files:
        st.write("### 暂无文档，请前往“文件中心”页面上传。")
    else:
        main()


if __name__ == "__main__":
    run_agent_center_page()
