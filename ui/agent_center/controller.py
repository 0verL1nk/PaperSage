from collections.abc import Callable, MutableMapping
from logging import Logger
from typing import Any


def load_files_from_db(
    *,
    session_state: MutableMapping[str, Any],
    user_uuid: str,
    list_user_files_fn: Callable[..., list[dict[str, Any]]],
    logger: Logger,
) -> None:
    raw_files = list_user_files_fn(uuid=user_uuid)
    session_state["files"] = []
    for file in raw_files:
        session_state["files"].append(
            {
                "file_path": file["file_path"],
                "file_name": file["file_name"],
                "uid": file["uid"],
                "created_at": file["created_at"],
            }
        )
    logger.debug("Loaded file list from DB: count=%s", len(session_state["files"]))


def load_projects_from_db(
    *,
    session_state: MutableMapping[str, Any],
    user_uuid: str,
    list_user_projects_fn: Callable[..., list[dict[str, Any]]],
) -> None:
    session_state["projects"] = list_user_projects_fn(
        uuid=user_uuid,
        include_archived=False,
    )


def scroll_chat_to_bottom(*, components: Any) -> None:
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


def inject_auto_load_more_on_scroll(*, components: Any, conversation_key: str) -> None:
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


def disable_auto_load_more_on_scroll(*, components: Any) -> None:
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


def render_project_session_sidebar(
    *,
    st: Any,
    user_uuid: str,
    project_uid: str,
    disabled: bool,
    ensure_project_sessions_fn: Callable[..., list[dict[str, Any]]],
    list_sessions_fn: Callable[..., list[dict[str, Any]]],
    ensure_default_session_fn: Callable[..., None],
    build_session_maps_fn: Callable[..., tuple[dict[str, dict[str, Any]], list[str]]],
    resolve_current_session_uid_fn: Callable[..., str],
    format_session_option_fn: Callable[[dict[str, Any]], str],
    update_selected_session_map_fn: Callable[..., dict[str, str]],
    create_and_select_session_fn: Callable[..., dict[str, str]],
    create_session_fn: Callable[..., dict[str, Any]],
    delete_session_fn: Callable[..., None],
    should_allow_delete_session_fn: Callable[[list[dict[str, Any]]], bool],
    delete_and_select_next_session_fn: Callable[..., tuple[dict[str, str], str]],
    drop_agent_session_cache_fn: Callable[[str, str], None],
    drop_conversation_cache_fn: Callable[[str, str], None],
    update_session_for_project_fn: Callable[..., None],
    build_session_preview_fn: Callable[[dict[str, Any]], str],
) -> dict[str, str]:
    sessions = ensure_project_sessions_fn(
        list_sessions_fn=list_sessions_fn,
        ensure_default_session_fn=ensure_default_session_fn,
        project_uid=project_uid,
        user_uuid=user_uuid,
    )
    if not sessions:
        raise ValueError("会话初始化失败")

    by_uid, ordered_uids = build_session_maps_fn(sessions)
    selected_map = st.session_state.get("agent_project_selected_sessions", {})
    current_uid = resolve_current_session_uid_fn(
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
        format_func=lambda uid: format_session_option_fn(by_uid.get(uid, {})),
        key=selector_key,
        disabled=disabled,
    )
    st.session_state.agent_project_selected_sessions = update_selected_session_map_fn(
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
        st.session_state.agent_project_selected_sessions = create_and_select_session_fn(
            create_session_fn=create_session_fn,
            selected_map=st.session_state.get("agent_project_selected_sessions", {}),
            project_uid=project_uid,
            user_uuid=user_uuid,
            session_name=new_name,
        )
        st.rerun()
        return {"session_uid": "", "session_name": ""}

    can_delete = should_allow_delete_session_fn(sessions)
    if action_cols[1].button(
        "删除会话",
        key=f"agent_project_session_delete_{project_uid}",
        disabled=disabled or not can_delete,
        use_container_width=True,
    ):
        selected_map, _next_uid = delete_and_select_next_session_fn(
            delete_session_fn=delete_session_fn,
            list_sessions_fn=list_sessions_fn,
            drop_agent_session_cache_fn=drop_agent_session_cache_fn,
            drop_conversation_cache_fn=drop_conversation_cache_fn,
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
            update_session_for_project_fn(
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

    preview = build_session_preview_fn(selected)
    if preview:
        st.caption(f"最近一条：{preview}")

    return {
        "session_uid": selected_uid,
        "session_name": str(selected.get("session_name") or "未命名会话"),
    }


def render_strategy_sidebar(
    *,
    st: Any,
    selected_project_uid: str,
    selected_session_uid: str,
    selected_session_name: str,
    conversation_key: str,
    turn_in_progress: bool,
    chat_messages: list[dict[str, Any]],
    render_output_archive_fn: Callable[..., None],
    render_workflow_metrics_fn: Callable[[str], None],
    render_context_usage_fn: Callable[[str], None],
    render_pinned_todo_panel_fn: Callable[..., None],
    render_pinned_human_requests_panel_fn: Callable[..., None],
) -> tuple[bool | None, bool | None]:
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
        render_output_archive_fn(selected_project_uid, disable_interaction=turn_in_progress)
        render_workflow_metrics_fn(conversation_key)
        render_context_usage_fn(conversation_key)
        render_pinned_todo_panel_fn(project_uid=selected_project_uid, expanded=True)
        render_pinned_human_requests_panel_fn(
            project_uid=selected_project_uid,
            chat_messages=chat_messages,
            expanded=False,
        )
        if turn_in_progress:
            st.info("正在生成回答，已临时锁定归档与文档切换，避免中断当前对话。")
    return force_plan, force_team


def render_chat_history_panel(
    *,
    st: Any,
    components: Any,
    chat_messages: list[dict[str, Any]],
    conversation_key: str,
    turn_in_progress: bool,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    get_history_paging_state_fn: Callable[..., dict[str, Any]],
    load_more_conversation_messages_fn: Callable[..., int],
    render_chat_history_fn: Callable[[list[dict[str, Any]]], None],
) -> None:
    paging_state = get_history_paging_state_fn(
        st=st,
        conversation_key=conversation_key,
    )
    if bool(paging_state.get("has_more_before")):
        loaded_count = int(
            paging_state.get("loaded_count", len(chat_messages)) or len(chat_messages)
        )
        total_count = int(paging_state.get("total_count", loaded_count) or loaded_count)
        st.caption(f"会话记录：已加载 {loaded_count} / {total_count}")
        if st.button(
            "加载更早对话",
            key=f"agent_history_load_more_{conversation_key}",
            use_container_width=True,
            disabled=turn_in_progress,
        ):
            loaded = load_more_conversation_messages_fn(
                st=st,
                user_uuid=user_uuid,
                project_uid=project_uid,
                session_uid=session_uid,
                conversation_key=conversation_key,
            )
            st.session_state["agent_history_keep_position"] = loaded > 0
            st.rerun()
            return
        inject_auto_load_more_on_scroll(
            components=components,
            conversation_key=conversation_key,
        )
    else:
        disable_auto_load_more_on_scroll(components=components)
    render_chat_history_fn(chat_messages)
    if not bool(st.session_state.get("agent_history_keep_position", False)):
        scroll_chat_to_bottom(components=components)
    st.session_state["agent_history_keep_position"] = False
