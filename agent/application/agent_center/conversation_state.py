import os
from typing import Any

DEFAULT_HISTORY_PAGE_SIZE = 40
MIN_HISTORY_PAGE_SIZE = 1
MAX_HISTORY_PAGE_SIZE = 200


def _resolve_history_page_size(history_page_size: int | None = None) -> int:
    if isinstance(history_page_size, int) and history_page_size > 0:
        return max(MIN_HISTORY_PAGE_SIZE, min(history_page_size, MAX_HISTORY_PAGE_SIZE))
    raw = str(os.getenv("AGENT_HISTORY_PAGE_SIZE", "") or "").strip()
    try:
        parsed = int(raw)
    except Exception:
        parsed = DEFAULT_HISTORY_PAGE_SIZE
    return max(MIN_HISTORY_PAGE_SIZE, min(parsed, MAX_HISTORY_PAGE_SIZE))


def _ensure_message_paging_map(session_state: dict) -> dict:
    paging_map = session_state.get("paper_project_message_paging", {})
    if not isinstance(paging_map, dict):
        paging_map = {}
    session_state["paper_project_message_paging"] = paging_map
    return paging_map


def _refresh_bootstrap_scope_message(
    *,
    messages: list[dict] | Any,
    project_name: str,
    scope_docs_count: int,
) -> tuple[list[dict] | Any, bool]:
    if not isinstance(messages, list) or len(messages) != 1:
        return messages, False
    first = messages[0]
    if not isinstance(first, dict):
        return messages, False
    if str(first.get("role") or "").strip().lower() != "assistant":
        return messages, False
    content = str(first.get("content") or "")
    if "已加载项目《" not in content or "当前检索范围" not in content:
        return messages, False
    expected = (
        f"已加载项目《{project_name}》，当前检索范围 {scope_docs_count} 篇文档。"
        " 工作流将按问题自动路由。"
    )
    if content == expected:
        return messages, False
    updated = dict(first)
    updated["content"] = expected
    return [updated], True


def get_history_paging_state(
    *,
    st,
    conversation_key: str,
) -> dict:
    paging_map = _ensure_message_paging_map(st.session_state)
    raw = paging_map.get(conversation_key, {})
    if not isinstance(raw, dict):
        raw = {}
    total_count = int(raw.get("total_count", 0) or 0)
    loaded_start = int(raw.get("loaded_start", 0) or 0)
    loaded_count = int(raw.get("loaded_count", 0) or 0)
    page_size = int(raw.get("page_size", DEFAULT_HISTORY_PAGE_SIZE) or DEFAULT_HISTORY_PAGE_SIZE)
    has_more_before = loaded_start > 0
    return {
        "total_count": max(0, total_count),
        "loaded_start": max(0, loaded_start),
        "loaded_count": max(0, loaded_count),
        "page_size": max(MIN_HISTORY_PAGE_SIZE, min(page_size, MAX_HISTORY_PAGE_SIZE)),
        "has_more_before": has_more_before,
    }


def persist_active_conversation(
    *,
    st,
    save_project_session_messages_fn,
    list_project_session_messages_fn,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    conversation_key: str,
) -> None:
    messages = st.session_state.get("agent_messages", [])
    if not isinstance(messages, list):
        return
    paging = get_history_paging_state(st=st, conversation_key=conversation_key)
    merged_messages = messages
    if int(paging.get("loaded_start", 0)) > 0 and callable(list_project_session_messages_fn):
        persisted = list_project_session_messages_fn(
            session_uid=session_uid,
            project_uid=project_uid,
            uuid=user_uuid,
        )
        if isinstance(persisted, list):
            prefix_len = min(len(persisted), int(paging["loaded_start"]))
            merged_messages = list(persisted[:prefix_len]) + list(messages)

    save_project_session_messages_fn(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=user_uuid,
        messages=merged_messages,
    )
    messages_map = st.session_state.get("paper_project_messages", {})
    if not isinstance(messages_map, dict):
        messages_map = {}
    messages_map[conversation_key] = messages
    st.session_state.paper_project_messages = messages_map
    paging_map = _ensure_message_paging_map(st.session_state)
    total_count = len(merged_messages)
    loaded_count = len(messages)
    loaded_start = max(0, total_count - loaded_count)
    paging_map[conversation_key] = {
        "total_count": total_count,
        "loaded_start": loaded_start,
        "loaded_count": loaded_count,
        "page_size": int(paging.get("page_size", _resolve_history_page_size())),
    }
    st.session_state.paper_project_message_paging = paging_map


def ensure_conversation_messages(
    *,
    st,
    list_project_session_messages_fn,
    list_project_session_messages_page_fn=None,
    count_project_session_messages_fn=None,
    persist_active_conversation_fn,
    user_uuid: str,
    project_uid: str,
    project_name: str,
    session_uid: str,
    conversation_key: str,
    scope_docs_count: int,
    history_page_size: int | None = None,
) -> None:
    messages_map = st.session_state.get("paper_project_messages", {})
    if not isinstance(messages_map, dict):
        messages_map = {}
    paging_map = _ensure_message_paging_map(st.session_state)
    page_size = _resolve_history_page_size(history_page_size)

    cached = messages_map.get(conversation_key)
    if isinstance(cached, list):
        refreshed_cached, changed = _refresh_bootstrap_scope_message(
            messages=cached,
            project_name=project_name,
            scope_docs_count=scope_docs_count,
        )
        if changed and isinstance(refreshed_cached, list):
            messages_map[conversation_key] = refreshed_cached
            st.session_state.paper_project_messages = messages_map
            st.session_state.agent_messages = refreshed_cached
            persist_active_conversation_fn(
                user_uuid=user_uuid,
                project_uid=project_uid,
                session_uid=session_uid,
                conversation_key=conversation_key,
            )
        else:
            st.session_state.agent_messages = cached
        if conversation_key not in paging_map:
            cached_count = len(st.session_state.agent_messages)
            paging_map[conversation_key] = {
                "total_count": cached_count,
                "loaded_start": 0,
                "loaded_count": cached_count,
                "page_size": page_size,
            }
            st.session_state.paper_project_message_paging = paging_map
        return

    persisted: list[dict] = []
    total_count = 0
    if callable(count_project_session_messages_fn) and callable(list_project_session_messages_page_fn):
        total_count = int(
            count_project_session_messages_fn(
                session_uid=session_uid,
                project_uid=project_uid,
                uuid=user_uuid,
            )
            or 0
        )
        if total_count > 0:
            loaded_count = min(page_size, total_count)
            loaded_start = max(0, total_count - loaded_count)
            persisted = list_project_session_messages_page_fn(
                session_uid=session_uid,
                project_uid=project_uid,
                uuid=user_uuid,
                offset=loaded_start,
                limit=loaded_count,
            )
            paging_map[conversation_key] = {
                "total_count": total_count,
                "loaded_start": loaded_start,
                "loaded_count": len(persisted) if isinstance(persisted, list) else 0,
                "page_size": page_size,
            }
            st.session_state.paper_project_message_paging = paging_map
    else:
        persisted = list_project_session_messages_fn(
            session_uid=session_uid,
            project_uid=project_uid,
            uuid=user_uuid,
        )
        total_count = len(persisted) if isinstance(persisted, list) else 0
        paging_map[conversation_key] = {
            "total_count": total_count,
            "loaded_start": 0,
            "loaded_count": total_count,
            "page_size": page_size,
        }
        st.session_state.paper_project_message_paging = paging_map

    if persisted:
        refreshed_persisted, changed = _refresh_bootstrap_scope_message(
            messages=persisted,
            project_name=project_name,
            scope_docs_count=scope_docs_count,
        )
        final_messages = refreshed_persisted if changed and isinstance(refreshed_persisted, list) else persisted
        messages_map[conversation_key] = final_messages
        st.session_state.paper_project_messages = messages_map
        st.session_state.agent_messages = final_messages
        if changed:
            persist_active_conversation_fn(
                user_uuid=user_uuid,
                project_uid=project_uid,
                session_uid=session_uid,
                conversation_key=conversation_key,
            )
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
    paging_map[conversation_key] = {
        "total_count": len(bootstrap),
        "loaded_start": 0,
        "loaded_count": len(bootstrap),
        "page_size": page_size,
    }
    st.session_state.paper_project_message_paging = paging_map
    persist_active_conversation_fn(
        user_uuid=user_uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        conversation_key=conversation_key,
    )


def load_more_conversation_messages(
    *,
    st,
    list_project_session_messages_page_fn,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    conversation_key: str,
    page_size: int | None = None,
) -> int:
    if not callable(list_project_session_messages_page_fn):
        return 0
    paging = get_history_paging_state(st=st, conversation_key=conversation_key)
    loaded_start = int(paging.get("loaded_start", 0))
    if loaded_start <= 0:
        return 0
    step = _resolve_history_page_size(page_size or int(paging.get("page_size", 0) or 0))
    next_start = max(0, loaded_start - step)
    fetch_limit = loaded_start - next_start
    if fetch_limit <= 0:
        return 0
    chunk = list_project_session_messages_page_fn(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=user_uuid,
        offset=next_start,
        limit=fetch_limit,
    )
    if not isinstance(chunk, list) or not chunk:
        return 0
    current = st.session_state.get("agent_messages", [])
    if not isinstance(current, list):
        current = []
    merged = list(chunk) + list(current)
    st.session_state.agent_messages = merged
    messages_map = st.session_state.get("paper_project_messages", {})
    if not isinstance(messages_map, dict):
        messages_map = {}
    messages_map[conversation_key] = merged
    st.session_state.paper_project_messages = messages_map
    paging_map = _ensure_message_paging_map(st.session_state)
    total_count = int(paging.get("total_count", len(merged)) or len(merged))
    paging_map[conversation_key] = {
        "total_count": total_count,
        "loaded_start": next_start,
        "loaded_count": len(merged),
        "page_size": step,
    }
    st.session_state.paper_project_message_paging = paging_map
    return len(chunk)


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
    extract_skill_context_texts_from_trace_fn=None,
) -> None:
    compact_summaries = st.session_state.get("paper_project_compact_summaries", {})
    tool_specs_map = st.session_state.get("paper_project_tool_specs", {})
    tool_specs = tool_specs_map.get(project_uid, [])
    skill_texts_map = st.session_state.get("paper_project_skill_context_texts", {})
    skill_context_texts = skill_texts_map.get(conversation_key, [])
    if (
        (not isinstance(skill_context_texts, list) or not skill_context_texts)
        and callable(extract_skill_context_texts_from_trace_fn)
    ):
        derived_skill_texts: list[str] = []
        seen: set[str] = set()
        for message in st.session_state.get("agent_messages", []):
            if not isinstance(message, dict):
                continue
            trace_payload = message.get("acp_trace")
            if not isinstance(trace_payload, list):
                continue
            try:
                extracted = extract_skill_context_texts_from_trace_fn(trace_payload)
            except Exception:
                extracted = []
            if not isinstance(extracted, list):
                continue
            for item in extracted:
                value = str(item or "").strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                derived_skill_texts.append(value)
        if derived_skill_texts:
            skill_context_texts = derived_skill_texts
            if not isinstance(skill_texts_map, dict):
                skill_texts_map = {}
            skill_texts_map[conversation_key] = derived_skill_texts
            st.session_state.paper_project_skill_context_texts = skill_texts_map
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


