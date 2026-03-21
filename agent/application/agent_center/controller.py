from typing import Any


def validate_runtime_prerequisites(*, api_key: str, model_name: str) -> str | None:
    if not api_key:
        return "missing_api_key"
    if not model_name:
        return "missing_model_name"
    return None


def load_scope_docs_with_text(
    *,
    scope_docs: list[dict[str, Any]],
    load_document_text_fn,
) -> tuple[list[dict[str, Any]], dict[str, int], str | None]:
    scope_docs_with_text: list[dict[str, Any]] = []
    cache_stats = {"session_hit": 0, "db_restore": 0, "extracted": 0}
    for scope_doc in scope_docs:
        scope_uid = str(scope_doc["uid"])
        text, source = load_document_text_fn(scope_uid, scope_doc["file_path"])
        if text is None:
            return [], cache_stats, scope_uid
        if source in cache_stats:
            cache_stats[source] += 1
        enriched = dict(scope_doc)
        enriched["text"] = text
        scope_docs_with_text.append(enriched)
    return scope_docs_with_text, cache_stats, None


def build_scope_cache_caption(cache_stats: dict[str, int]) -> str:
    db_restore = int(cache_stats.get("db_restore", 0))
    session_hit = int(cache_stats.get("session_hit", 0))
    if db_restore > 0:
        return f"文档内容已从数据库缓存恢复：{db_restore} 篇。"
    if session_hit > 0:
        return f"文档内容已命中会话缓存：{session_hit} 篇。"
    return ""


def build_turn_context(
    *,
    prompt: str,
    user_uuid: str,
    project_uid: str,
    detect_language_fn,
    search_project_memory_items_fn,
    memory_limit: int = 4,
) -> dict[str, Any]:
    base_prompt = str(prompt or "").strip()
    if not base_prompt:
        return {}

    long_term_memories = search_project_memory_items_fn(
        uuid=user_uuid,
        project_uid=project_uid,
        query=base_prompt,
        limit=memory_limit,
    )
    context: dict[str, Any] = {}

    response_language = _normalize_language_code(detect_language_fn(base_prompt))
    if response_language:
        context["response_language"] = response_language

    memory_items = _normalize_memory_items(long_term_memories, max_chars=1600)
    if memory_items:
        context["memory_items"] = memory_items
    return context


def _collapse_inline_text(text: Any, *, limit: int) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ""
    if len(compact) <= max(1, int(limit)):
        return compact
    clipped = max(1, int(limit)) - 3
    return f"{compact[:clipped]}..."


def _normalize_language_code(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "en":
        return "en"
    if normalized == "zh":
        return "zh"
    return ""


def _normalize_memory_items(
    memory_items: list[dict[str, Any]],
    *,
    max_chars: int,
) -> list[dict[str, str]]:
    if not isinstance(memory_items, list) or not memory_items:
        return []
    normalized_items: list[dict[str, str]] = []
    current_len = 0
    for item in memory_items:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type") or "episodic").strip().lower() or "episodic"
        content = _collapse_inline_text(item.get("content"), limit=220)
        if not content:
            continue
        candidate = f"{memory_type}:{content}"
        added_len = len(candidate) if not normalized_items else len(candidate) + 1
        if current_len + added_len > max_chars:
            break
        normalized_items.append({"memory_type": memory_type, "content": content})
        current_len += added_len
    return normalized_items


def resolve_runtime_session_id(runtime_config: dict[str, Any] | Any) -> str:
    if isinstance(runtime_config, dict):
        return str(runtime_config.get("configurable", {}).get("thread_id") or "-")
    return "-"


def resolve_selected_doc_uid_for_logging(scope_docs_with_text: list[dict[str, Any]]) -> str:
    if not scope_docs_with_text:
        return ""
    return str(scope_docs_with_text[0].get("uid") or "")


def resolve_archive_target(
    *,
    scope_docs_with_text: list[dict[str, Any]],
    project_name: str,
) -> tuple[str | None, str]:
    if len(scope_docs_with_text) == 1:
        doc = scope_docs_with_text[0]
        return str(doc.get("uid") or ""), str(doc.get("file_name") or project_name)
    return None, project_name


def serialize_output_content(
    *,
    answer: str,
    mindmap_data: dict[str, Any] | None,
    json_dumps_fn,
) -> str:
    if mindmap_data:
        return str(json_dumps_fn(mindmap_data, ensure_ascii=False))
    return answer
