from typing import Any

from agent.adapters.openviking_runtime import get_openviking_adapter
from agent.adapters.viking_adapter import VikingAdapterError
from agent.domain.openviking_contracts import OpenVikingSearchRequest
from agent.memory.policy import classify_turn_memory_type, ttl_for_memory_type

_MEMORY_NAMESPACE = "memory"
_MEMORY_TYPE_DEFAULT = "episodic"
_MEMORY_MAX_CONTENT_LEN = 1200
_MEMORY_MAX_TITLE_LEN = 80
_MEMORY_MAX_PROMPT_LEN = 500
_MEMORY_MAX_ANSWER_LEN = 800


def persist_turn_memory(
    *,
    user_uuid: str,
    project_uid: str,
    session_uid: str,
    prompt: str,
    answer: str,
) -> None:
    prompt_text = str(prompt or "").strip()
    answer_text = str(answer or "").strip()
    if not prompt_text or not answer_text:
        return
    memory_content = f"Q: {prompt_text}\nA: {answer_text}"
    if len(memory_content) > _MEMORY_MAX_CONTENT_LEN:
        memory_content = memory_content[:_MEMORY_MAX_CONTENT_LEN] + "..."
    memory_type = classify_turn_memory_type(prompt_text, answer_text)
    expires_at = ttl_for_memory_type(memory_type)
    metadata: dict[str, Any] = {
        "namespace": _MEMORY_NAMESPACE,
        "user_uuid": user_uuid,
        "project_uid": project_uid,
        "session_uid": session_uid,
        "memory_type": memory_type,
        "title": prompt_text[:_MEMORY_MAX_TITLE_LEN],
        "source_prompt": prompt_text[:_MEMORY_MAX_PROMPT_LEN],
        "source_answer": answer_text[:_MEMORY_MAX_ANSWER_LEN],
        "expires_at": expires_at,
    }
    try:
        adapter = get_openviking_adapter()
        _ = adapter.add_resource(
            project_uid=project_uid,
            content=memory_content,
            metadata=metadata,
        )
    except (ImportError, ModuleNotFoundError, OSError, VikingAdapterError) as exc:
        raise RuntimeError(
            f"failed to persist OpenViking memory for project {project_uid}"
        ) from exc


def query_turn_memory(
    *,
    uuid: str,
    project_uid: str,
    query: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    adapter = get_openviking_adapter()
    hits = adapter.search(
        OpenVikingSearchRequest(
            project_uid=project_uid,
            query=normalized_query,
            top_k=max(1, int(limit)),
        )
    )
    result: list[dict[str, Any]] = []
    for hit in hits:
        metadata = hit.metadata if isinstance(hit.metadata, dict) else {}
        if str(metadata.get("namespace") or "").strip() != _MEMORY_NAMESPACE:
            continue
        owner_uuid = str(metadata.get("user_uuid") or "").strip()
        if owner_uuid and owner_uuid != uuid:
            continue
        result.append(
            {
                "memory_type": str(metadata.get("memory_type") or _MEMORY_TYPE_DEFAULT),
                "content": hit.content,
                "score": hit.score,
            }
        )
    return result
