from typing import Any

PROMPT_LAYOUT_VERSION = "CTXv1"
PROMPT_INVENTORY_TAG = "I"
PROMPT_SUMMARY_TAG = "S"
PROMPT_MEMORY_TAG = "M"
PROMPT_LANGUAGE_TAG = "L"
PROMPT_QUERY_TAG = "Q"
DEFAULT_SYSTEM_PROMPT_ID = "paper_qa"
DEFAULT_COLLAB_PROMPT_ID = "a2a"


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


def build_hinted_prompt(
    *,
    prompt: str,
    compact_summary: str,
    user_uuid: str,
    project_uid: str,
    detect_language_fn,
    with_language_hint_fn,
    inject_compact_summary_fn,
    search_project_memory_items_fn,
    inject_long_term_memory_fn,
    tool_specs: list[dict[str, Any]] | None = None,
    skill_names: list[str] | None = None,
    system_prompt_id: str = DEFAULT_SYSTEM_PROMPT_ID,
    collaborator_prompt_id: str = DEFAULT_COLLAB_PROMPT_ID,
    memory_limit: int = 4,
) -> str:
    base_prompt = str(prompt or "").strip()
    if not base_prompt:
        return ""

    prompt_with_language = with_language_hint_fn(base_prompt, detect_language_fn)
    resolved_prompt, language_note = _split_prompt_and_language_note(
        base_prompt=base_prompt,
        prompt_with_language=str(prompt_with_language or base_prompt),
    )

    long_term_memories = search_project_memory_items_fn(
        uuid=user_uuid,
        project_uid=project_uid,
        query=base_prompt,
        limit=memory_limit,
    )
    compact_summary_text = _collapse_inline_text(compact_summary, limit=1400)
    memory_text = _serialize_memory_items(long_term_memories, max_chars=1600)
    language_text = _normalize_language_note(language_note)
    inventory_line = _build_inventory_line(
        system_prompt_id=system_prompt_id,
        collaborator_prompt_id=collaborator_prompt_id,
        tool_specs=tool_specs,
        skill_names=skill_names,
    )
    return "\n".join(
        [
            PROMPT_LAYOUT_VERSION,
            inventory_line,
            f"{PROMPT_SUMMARY_TAG}:{compact_summary_text or '-'}",
            f"{PROMPT_MEMORY_TAG}:{memory_text or '-'}",
            f"{PROMPT_LANGUAGE_TAG}:{language_text or '-'}",
            f"{PROMPT_QUERY_TAG}:{resolved_prompt}",
        ]
    )


def _split_prompt_and_language_note(
    *,
    base_prompt: str,
    prompt_with_language: str,
) -> tuple[str, str]:
    normalized_base = str(base_prompt or "")
    normalized_prompt = str(prompt_with_language or "").strip()
    if not normalized_prompt:
        return normalized_base, ""
    if normalized_prompt.startswith(normalized_base):
        language_note = normalized_prompt[len(normalized_base) :].strip()
        return normalized_base, language_note
    return normalized_prompt, ""


def _collapse_inline_text(text: Any, *, limit: int) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return ""
    if len(compact) <= max(1, int(limit)):
        return compact
    clipped = max(1, int(limit)) - 3
    return f"{compact[:clipped]}..."


def _normalize_language_note(note: str) -> str:
    compact = _collapse_inline_text(note, limit=140)
    while compact.startswith(":"):
        compact = compact[1:].lstrip()
    lowered = compact.lower()
    if "english" in lowered:
        return "en"
    if "中文" in compact or "chinese" in lowered:
        return "zh"
    return compact


def _serialize_memory_items(
    memory_items: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    if not isinstance(memory_items, list) or not memory_items:
        return ""
    lines: list[str] = []
    current_len = 0
    for item in memory_items:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type") or "episodic").strip().lower() or "episodic"
        content = _collapse_inline_text(item.get("content"), limit=220)
        if not content:
            continue
        candidate = f"{memory_type}:{content}"
        added_len = len(candidate) if not lines else len(candidate) + 3
        if current_len + added_len > max_chars:
            break
        lines.append(candidate)
        current_len += added_len
    return " | ".join(lines)


def _build_inventory_line(
    *,
    system_prompt_id: str,
    collaborator_prompt_id: str,
    tool_specs: list[dict[str, Any]] | None,
    skill_names: list[str] | None,
) -> str:
    system_id = _collapse_inline_text(system_prompt_id, limit=32) or DEFAULT_SYSTEM_PROMPT_ID
    collab_id = _collapse_inline_text(collaborator_prompt_id, limit=32) or DEFAULT_COLLAB_PROMPT_ID
    tools = _serialize_tool_inventory(tool_specs, max_chars=320) or "-"
    skills = _serialize_skill_names(skill_names, max_chars=180) or "-"
    return f"{PROMPT_INVENTORY_TAG}:sys={system_id},col={collab_id},tools={tools},skills={skills}"


def _serialize_tool_names(
    tool_specs: list[dict[str, Any]] | None,
    *,
    max_chars: int,
) -> str:
    if not isinstance(tool_specs, list):
        return ""
    names = sorted(
        {
            str(item.get("name") or "").strip()
            for item in tool_specs
            if isinstance(item, dict)
        }
    )
    names = [name for name in names if name]
    if not names:
        return ""
    text = "|".join(names)
    return _collapse_inline_text(text, limit=max_chars)


def _serialize_tool_inventory(
    tool_specs: list[dict[str, Any]] | None,
    *,
    max_chars: int,
) -> str:
    if not isinstance(tool_specs, list):
        return ""
    records: list[str] = []
    for item in tool_specs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        description = " ".join(str(item.get("description") or "").split()).strip()
        fields = " ".join(str(item.get("args_schema") or "").split()).strip()
        pieces = [name]
        if description:
            pieces.append(description)
        if fields and fields not in {"{}", "[]"}:
            pieces.append(fields)
        records.append(":".join(pieces))
    if not records:
        return ""
    return _collapse_inline_text(" | ".join(sorted(records)), limit=max_chars)


def _serialize_skill_names(
    skill_names: list[str] | None,
    *,
    max_chars: int,
) -> str:
    if not isinstance(skill_names, list):
        return ""
    names = sorted({str(item or "").strip() for item in skill_names if str(item or "").strip()})
    if not names:
        return ""
    return _collapse_inline_text("|".join(names), limit=max_chars)


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
