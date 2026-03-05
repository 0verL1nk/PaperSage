import datetime
import re
from typing import Any

from .repository import list_project_memory_items, touch_memory_items


def _memory_term_set(text: str) -> set[str]:
    value = str(text or "").lower()
    terms = re.findall(r"[a-z0-9\u4e00-\u9fff]+", value)
    return {item for item in terms if len(item.strip()) >= 2}


def _memory_recency_score(updated_at: str) -> float:
    if not isinstance(updated_at, str) or not updated_at.strip():
        return 0.0
    try:
        updated_dt = datetime.datetime.strptime(updated_at.strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return 0.0
    delta_hours = max(0.0, (datetime.datetime.now() - updated_dt).total_seconds() / 3600.0)
    return 1.0 / (1.0 + delta_hours / 24.0)


def search_project_memory_items(
    *,
    uuid: str,
    project_uid: str,
    query: str,
    limit: int = 5,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    base_items = list_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        limit=max(20, int(limit) * 10),
        db_name=db_name,
    )
    if not base_items:
        return []

    query_terms = _memory_term_set(query)
    query_text = str(query or "").strip().lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in base_items:
        content = str(item.get("content") or "")
        title = str(item.get("title") or "")
        content_lower = content.lower()
        title_lower = title.lower()
        item_terms = _memory_term_set(f"{title}\n{content}")
        overlap = len(query_terms & item_terms) if query_terms else 0
        partial_hits = 0
        for term in query_terms:
            if term in content_lower or term in title_lower:
                partial_hits += 1
        text_bonus = 1.0 if query_text and query_text in content_lower else 0.0
        recency = _memory_recency_score(str(item.get("updated_at") or ""))
        score = overlap * 3.0 + partial_hits * 1.5 + text_bonus + recency
        if query_terms and overlap <= 0 and partial_hits <= 0 and text_bonus <= 0:
            continue
        enriched = dict(item)
        enriched["score"] = round(score, 4)
        scored.append((score, enriched))

    if not scored and not query_terms:
        return base_items[: max(1, int(limit))]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_items = [item for _, item in scored[: max(1, int(limit))]]
    touch_memory_items(
        memory_uids=[str(item.get("memory_uid") or "") for item in top_items],
        db_name=db_name,
    )
    return top_items
