from typing import Any


def drop_agent_session_cache(
    *,
    session_state: dict[str, Any],
    build_session_key_fn,
    mode_leader: str,
    project_uid: str,
    session_uid: str,
) -> None:
    leader_sessions = session_state.get("paper_agent_sessions", {})
    if not isinstance(leader_sessions, dict):
        return
    leader_sessions.pop(build_session_key_fn(project_uid, session_uid, mode_leader), None)
    session_state["paper_agent_sessions"] = leader_sessions


def drop_conversation_cache(
    *,
    session_state: dict[str, Any],
    build_conversation_key_fn,
    project_uid: str,
    session_uid: str,
) -> None:
    key = build_conversation_key_fn(project_uid, session_uid)
    for map_name in (
        "paper_project_messages",
        "paper_project_metrics",
        "paper_project_compact_summaries",
        "paper_project_context_usage",
        "paper_project_skill_context_texts",
    ):
        source = session_state.get(map_name, {})
        if isinstance(source, dict):
            source.pop(key, None)
            session_state[map_name] = source


def ensure_project_sessions(
    *,
    list_sessions_fn,
    ensure_default_session_fn,
    project_uid: str,
    user_uuid: str,
) -> list[dict[str, Any]]:
    sessions = list_sessions_fn(project_uid=project_uid, uuid=user_uuid)
    if sessions:
        return sessions
    ensure_default_session_fn(project_uid=project_uid, uuid=user_uuid)
    return list_sessions_fn(project_uid=project_uid, uuid=user_uuid)


def build_session_maps(
    sessions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_uid = {str(item["session_uid"]): item for item in sessions}
    ordered_uids = [str(item["session_uid"]) for item in sessions]
    return by_uid, ordered_uids


def resolve_current_session_uid(
    *,
    selected_map: dict[str, str] | Any,
    project_uid: str,
    by_uid: dict[str, dict[str, Any]],
    ordered_uids: list[str],
) -> str:
    if not ordered_uids:
        return ""
    if not isinstance(selected_map, dict):
        selected_map = {}
    current_uid = str(selected_map.get(project_uid) or "")
    if current_uid not in by_uid:
        current_uid = ordered_uids[0]
    return current_uid


def normalize_selector_value(
    *,
    selector_value: Any,
    by_uid: dict[str, dict[str, Any]],
    fallback_uid: str,
) -> str:
    value = str(selector_value or "")
    if value not in by_uid:
        return fallback_uid
    return value


def format_session_option(item: dict[str, Any]) -> str:
    return (
        f"{str(item.get('session_name') or '未命名会话')} · "
        f"{int(item.get('message_count') or 0)} 条"
    )


def build_session_preview(
    selected_item: dict[str, Any],
    *,
    max_chars: int = 80,
) -> str:
    preview = str(selected_item.get("last_message") or "").replace("\n", " ").strip()
    if not preview:
        return ""
    return preview[:max_chars]


def should_allow_delete_session(sessions: list[dict[str, Any]]) -> bool:
    return len(sessions) > 1


def update_selected_session_map(
    *,
    selected_map: dict[str, str] | Any,
    project_uid: str,
    selected_uid: str,
) -> dict[str, str]:
    next_map: dict[str, str] = dict(selected_map) if isinstance(selected_map, dict) else {}
    next_map[project_uid] = selected_uid
    return next_map


def create_and_select_session(
    *,
    create_session_fn,
    selected_map: dict[str, str] | Any,
    project_uid: str,
    user_uuid: str,
    session_name: str,
) -> dict[str, str]:
    created = create_session_fn(
        project_uid=project_uid,
        uuid=user_uuid,
        session_name=session_name,
    )
    created_uid = str(created.get("session_uid") or "")
    return update_selected_session_map(
        selected_map=selected_map,
        project_uid=project_uid,
        selected_uid=created_uid,
    )


def delete_and_select_next_session(
    *,
    delete_session_fn,
    list_sessions_fn,
    drop_agent_session_cache_fn,
    drop_conversation_cache_fn,
    selected_map: dict[str, str] | Any,
    project_uid: str,
    user_uuid: str,
    selected_uid: str,
) -> tuple[dict[str, str], str]:
    delete_session_fn(
        session_uid=selected_uid,
        project_uid=project_uid,
        uuid=user_uuid,
    )
    drop_agent_session_cache_fn(project_uid, selected_uid)
    drop_conversation_cache_fn(project_uid, selected_uid)
    remaining = list_sessions_fn(project_uid=project_uid, uuid=user_uuid)
    next_uid = str(remaining[0]["session_uid"]) if remaining else ""
    next_map = update_selected_session_map(
        selected_map=selected_map,
        project_uid=project_uid,
        selected_uid=next_uid,
    )
    return next_map, next_uid
