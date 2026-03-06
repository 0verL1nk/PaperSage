from typing import Any


def has_cached_agent_session(
    *,
    session_state: dict[str, Any],
    build_session_key_fn,
    mode_leader: str,
    project_uid: str,
    session_uid: str,
    scope_signature: str,
) -> bool:
    leader_session_key = build_session_key_fn(project_uid, session_uid, mode_leader)
    leader_sessions = session_state.get("paper_agent_sessions", {})
    retrievers = session_state.get("paper_evidence_retrievers", {})
    signatures = session_state.get("paper_project_scope_signatures", {})
    current_signature = signatures.get(project_uid)
    return (
        leader_session_key in leader_sessions
        and project_uid in retrievers
        and current_signature == scope_signature
    )


def clear_project_runtime(
    *,
    session_state: dict[str, Any],
    project_uid: str,
    mode_leader: str,
) -> None:
    leader_sessions = session_state.get("paper_agent_sessions", {})
    retrievers = session_state.get("paper_evidence_retrievers", {})
    llm_map = session_state.get("paper_project_llms", {})
    policy_llm_map = session_state.get("paper_project_policy_llms", {})
    search_fn_map = session_state.get("paper_project_search_document_fns", {})
    if isinstance(leader_sessions, dict):
        leader_prefix = f"{mode_leader}:{project_uid}:"
        leader_keys = [key for key in leader_sessions if str(key).startswith(leader_prefix)]
        for key in leader_keys:
            leader_sessions.pop(key, None)
    retrievers.pop(project_uid, None)
    llm_map.pop(project_uid, None)
    policy_llm_map.pop(project_uid, None)
    search_fn_map.pop(project_uid, None)
    session_state["paper_agent_sessions"] = leader_sessions
    session_state["paper_evidence_retrievers"] = retrievers
    session_state["paper_project_llms"] = llm_map
    session_state["paper_project_policy_llms"] = policy_llm_map
    session_state["paper_project_search_document_fns"] = search_fn_map


def load_document_text(
    *,
    session_state: dict[str, Any],
    logger,
    selected_uid: str,
    file_path: str,
    load_cached_extraction_fn,
    extract_document_fn,
    save_cached_extraction_fn,
) -> tuple[str | None, str, str | None]:
    document_text_cache = session_state.get("document_text_cache", {})
    document_text = document_text_cache.get(selected_uid)
    if isinstance(document_text, str):
        logger.info("Document text cache hit")
        return document_text, "session_hit", None

    logger.info("Document text session cache miss: uid=%s", selected_uid)
    persisted_text: str | None = None
    try:
        cached_from_db = load_cached_extraction_fn(selected_uid)
        if isinstance(cached_from_db, str) and cached_from_db.strip():
            persisted_text = cached_from_db
            logger.info(
                "Document text restored from DB cache: uid=%s text_len=%s",
                selected_uid,
                len(persisted_text),
            )
    except Exception as exc:
        logger.warning("Failed to load persisted document extraction: %s", exc)

    if persisted_text is None:
        logger.info("Document text DB cache miss, extracting: path=%s", file_path)
        document_result = extract_document_fn(file_path)
        if document_result.get("result") != 1:
            reason = str(document_result.get("text") or "unknown")
            logger.error("Document extraction failed: reason=%s", reason)
            return None, "error", f"文档加载失败：{reason}"
        document_text = document_result.get("text")
        if not isinstance(document_text, str):
            logger.error("Document extraction returned non-string text")
            return None, "error", "文档内容解析失败，无法建立 RAG 索引。"
        try:
            save_cached_extraction_fn(
                uid=selected_uid,
                file_path=file_path,
                content=document_text,
            )
            logger.info(
                "Document extraction persisted to DB cache: uid=%s text_len=%s",
                selected_uid,
                len(document_text),
            )
        except Exception as exc:
            logger.warning("Failed to persist document extraction: %s", exc)
        source = "extracted"
    else:
        document_text = persisted_text
        source = "db_restore"

    document_text_cache[selected_uid] = document_text
    session_state["document_text_cache"] = document_text_cache
    logger.info("Document text cached in session: text_len=%s", len(document_text))
    return document_text, source, None
