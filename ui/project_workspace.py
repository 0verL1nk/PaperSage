from typing import Any

import streamlit as st

from agent.adapters import list_project_files_for_user


def inject_workspace_styles() -> None:
    st.markdown(
        """
        <style>
        .workspace-status {
            background: linear-gradient(
                120deg,
                var(--workspace-grad-start) 0%,
                var(--workspace-grad-end) 100%
            );
            border: 1px solid var(--line-soft);
            border-radius: 12px;
            padding: 10px 14px;
            margin: 8px 0 12px 0;
        }
        .workspace-status .title {
            font-weight: 700;
            color: var(--workspace-title);
            margin-bottom: 6px;
        }
        .workspace-status .chips {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            font-size: 12px;
        }
        .workspace-status .chip {
            background: var(--workspace-chip-bg);
            border: 1px solid var(--workspace-chip-line);
            color: var(--ink-700);
            border-radius: 999px;
            padding: 2px 10px;
        }
        .workspace-hint {
            border-left: 4px solid var(--ink-500);
            background: var(--workspace-hint-bg);
            padding: 8px 12px;
            border-radius: 8px;
            margin: 8px 0 10px 0;
            color: var(--ink-700);
            font-size: 13px;
        }
        .project-card-title {
            font-weight: 700;
            color: var(--workspace-card-title);
            font-size: 15px;
            margin-bottom: 4px;
        }
        .project-card-sub {
            color: var(--workspace-card-sub);
            font-size: 12px;
            margin-bottom: 8px;
        }
        @media (max-width: 768px) {
            .workspace-status {
                padding: 9px 10px;
            }
            .workspace-status .chips {
                gap: 6px;
            }
            .workspace-status .chip {
                font-size: 11px;
                padding: 2px 8px;
            }
            .project-card-title {
                font-size: 14px;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def build_project_doc_count_map(
    projects: list[dict[str, Any]],
    user_uuid: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for project in projects:
        project_uid = str(project.get("project_uid") or "")
        if not project_uid:
            continue
        files = list_project_files_for_user(
            project_uid=project_uid,
            uuid=user_uuid,
            active_only=False,
        )
        counts[project_uid] = len(files)
    return counts


def select_project_card(
    projects: list[dict[str, Any]],
    doc_count_map: dict[str, int],
    *,
    disabled: bool = False,
) -> dict[str, Any] | None:
    if not projects:
        return None
    options = [
        str(item.get("project_uid") or "")
        for item in projects
        if item.get("project_uid")
    ]
    if not options:
        return None

    project_by_uid = {
        str(item["project_uid"]): item for item in projects if item.get("project_uid")
    }
    selected_key = "agent_project_selected_uid"
    if st.session_state.get(selected_key) not in options:
        st.session_state[selected_key] = options[0]
    selected_uid = str(st.session_state[selected_key])

    st.markdown("### 项目工作台")
    grid_cols = st.columns(2)
    for idx, project_uid in enumerate(options):
        project = project_by_uid[project_uid]
        project_name = str(project.get("project_name") or "未命名项目")
        updated_at = str(project.get("updated_at") or "")
        project_docs = int(doc_count_map.get(project_uid, 0))
        active = project_uid == selected_uid
        badge = "当前项目" if active else "切换到此项目"
        with grid_cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"<div class='project-card-title'>{project_name}</div>", unsafe_allow_html=True)
                st.markdown(
                    (
                        "<div class='project-card-sub'>"
                        f"文档数: {project_docs} | 更新时间: {updated_at or '-'}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    badge,
                    key=f"agent_project_pick_{project_uid}",
                    disabled=disabled,
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state[selected_key] = project_uid
                    selected_uid = project_uid
                    st.rerun()

    return project_by_uid.get(selected_uid)


def select_project_sidebar(
    projects: list[dict[str, Any]],
    doc_count_map: dict[str, int],
    *,
    disabled: bool = False,
) -> dict[str, Any] | None:
    if not projects:
        return None
    options = [
        str(item.get("project_uid") or "")
        for item in projects
        if item.get("project_uid")
    ]
    if not options:
        return None

    project_by_uid = {
        str(item["project_uid"]): item for item in projects if item.get("project_uid")
    }
    selected_key = "agent_project_selected_uid"
    current = str(st.session_state.get(selected_key) or "")
    if current not in options:
        current = options[0]
    sidebar_key = f"{selected_key}_sidebar"
    sidebar_current = str(st.session_state.get(sidebar_key) or current)
    if sidebar_current not in options:
        sidebar_current = current
    selected_uid = st.selectbox(
        "当前项目",
        options=options,
        index=options.index(sidebar_current),
        format_func=lambda uid: str(
            project_by_uid.get(uid, {}).get("project_name") or "未命名项目"
        ),
        key=sidebar_key,
        disabled=disabled,
    )
    st.session_state[selected_key] = selected_uid
    selected = project_by_uid.get(selected_uid)
    if selected is None:
        return None

    selected_name = str(selected.get("project_name") or "未命名项目")
    selected_docs = int(doc_count_map.get(selected_uid, 0))
    updated_at = str(selected.get("updated_at") or "-")
    st.markdown(
        (
            "<div class=\"workspace-hint\">"
            f"当前项目：<b>{selected_name}</b><br>"
            f"文档数：{selected_docs} ｜ 更新时间：{updated_at}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    return selected


def select_scope_documents_drawer(
    scoped_files: list[dict[str, Any]],
    project_uid: str,
    *,
    disabled: bool = False,
) -> list[dict[str, Any]]:
    if not scoped_files:
        return []
    scope_key = f"agent_project_scope_{project_uid}"
    scope_widget_key = f"{scope_key}_widget"
    all_uids = [str(item["uid"]) for item in scoped_files]
    if scope_key not in st.session_state:
        st.session_state[scope_key] = all_uids

    with st.expander("检索范围抽屉", expanded=False):
        query = st.text_input(
            "搜索范围文档",
            value="",
            key=f"agent_scope_search_{project_uid}",
            disabled=disabled,
            placeholder="按文档名过滤",
        )
        filtered = scoped_files
        if isinstance(query, str) and query.strip():
            q = query.strip().lower()
            filtered = [
                item
                for item in scoped_files
                if q in str(item.get("file_name") or "").lower()
            ]
        action_cols = st.columns(3)
        if action_cols[0].button("全选", key=f"agent_scope_all_{project_uid}", disabled=disabled):
            st.session_state[scope_key] = [str(item["uid"]) for item in filtered]
        if action_cols[1].button("清空", key=f"agent_scope_none_{project_uid}", disabled=disabled):
            st.session_state[scope_key] = []
        if action_cols[2].button("最近10篇", key=f"agent_scope_recent_{project_uid}", disabled=disabled):
            st.session_state[scope_key] = [str(item["uid"]) for item in filtered[:10]]
        options = [str(item["uid"]) for item in filtered]
        persisted_selected = st.session_state.get(scope_key, [])
        if not isinstance(persisted_selected, list):
            persisted_selected = []
        persisted_selected = [uid for uid in persisted_selected if uid in all_uids]
        visible_selected = [uid for uid in persisted_selected if uid in options]
        if not visible_selected and options:
            visible_selected = options
        st.session_state[scope_widget_key] = visible_selected
        label_map = {
            str(item["uid"]): f"{item['file_name']} ({str(item['uid'])[:8]})"
            for item in filtered
        }
        selected_visible_uids = st.multiselect(
            "选择纳入检索的文档",
            options=options,
            format_func=lambda uid: label_map.get(uid, uid),
            key=scope_widget_key,
            disabled=disabled,
        )
        hidden_selected = [uid for uid in persisted_selected if uid not in options]
        st.session_state[scope_key] = hidden_selected + [
            str(uid) for uid in selected_visible_uids if str(uid) in options
        ]
    selected_uid_set = {str(uid) for uid in st.session_state.get(scope_key, [])}
    return [item for item in scoped_files if str(item["uid"]) in selected_uid_set]


def render_workspace_status_bar(
    *,
    project_name: str,
    total_docs: int,
    selected_docs: int,
    turn_in_progress: bool,
) -> None:
    status_text = "生成中" if turn_in_progress else "空闲"
    scope_ratio = 0 if total_docs <= 0 else int((selected_docs / total_docs) * 100)
    st.markdown(
        f"""
        <div class="workspace-status">
            <div class="title">项目工作区：{project_name}</div>
            <div class="chips">
                <span class="chip">总文档 {total_docs}</span>
                <span class="chip">当前范围 {selected_docs}</span>
                <span class="chip">覆盖率 {scope_ratio}%</span>
                <span class="chip">状态 {status_text}</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_project_context_hint(
    project_name: str,
    scope_docs: list[dict[str, Any]],
) -> None:
    scope_names = [str(item.get("file_name") or "") for item in scope_docs]
    preview = "、".join(name for name in scope_names[:4] if name)
    if len(scope_names) > 4:
        preview = f"{preview} 等 {len(scope_names)} 篇"
    st.markdown(
        (
            "<div class=\"workspace-hint\">"
            f"当前项目：<b>{project_name}</b>｜检索范围：{preview or '未设置'}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
