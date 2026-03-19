import datetime
import hashlib
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
    memory_type: str | None = None,
    status: str | None = "active",
    limit: int = 5,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    base_items = list_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        memory_type=memory_type,
        status=status,
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
        canonical = str(item.get("canonical_text") or "")
        content_lower = content.lower()
        title_lower = title.lower()
        canonical_lower = canonical.lower()
        item_terms = _memory_term_set(f"{title}\n{canonical}\n{content}")
        overlap = len(query_terms & item_terms) if query_terms else 0
        partial_hits = 0
        for term in query_terms:
            if term in content_lower or term in title_lower or term in canonical_lower:
                partial_hits += 1
        text_bonus = 1.0 if query_text and (
            query_text in content_lower or query_text in canonical_lower
        ) else 0.0
        recency = _memory_recency_score(str(item.get("updated_at") or ""))
        score = overlap * 3.0 + partial_hits * 1.5 + text_bonus + recency
        if query_terms and overlap <= 0 and partial_hits <= 0 and text_bonus <= 0:
            continue
        enriched = dict(item)
        enriched["score"] = round(score, 4)
        scored.append((score, enriched))

    if not scored and not query_terms:
        return base_items[: max(1, int(limit))]
    if not scored and isinstance(memory_type, str) and memory_type.strip().lower() == "knowledge_memory":
        top_items = base_items[: max(1, int(limit))]
        touch_memory_items(
            memory_uids=[str(item.get("memory_uid") or "") for item in top_items],
            db_name=db_name,
        )
        return top_items

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_items = [item for _, item in scored[: max(1, int(limit))]]
    touch_memory_items(
        memory_uids=[str(item.get("memory_uid") or "") for item in top_items],
        db_name=db_name,
    )
    return top_items


def _stable_memory_dedup_key(*, memory_type: str, canonical_text: str) -> str:
    normalized_type = str(memory_type or "").strip().lower()
    normalized_text = " ".join(str(canonical_text or "").strip().lower().split())
    digest = hashlib.sha1(f"{normalized_type}:{normalized_text}".encode("utf-8")).hexdigest()
    return f"{normalized_type}:{digest[:16]}"


def write_memory_from_leader(
    *,
    uuid: str,
    project_uid: str,
    session_uid: str,
    items: list[dict[str, Any]],
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    from .reconcile import apply_memory_candidates

    candidates: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type") or "").strip().lower()
        content = " ".join(str(item.get("content") or "").split()).strip()
        canonical_text = " ".join(
            str(item.get("canonical_text") or content).split()
        ).strip()
        if not memory_type or not content or not canonical_text:
            continue
        dedup_key = str(item.get("dedup_key") or "").strip()
        if not dedup_key:
            dedup_key = _stable_memory_dedup_key(
                memory_type=memory_type,
                canonical_text=canonical_text,
            )
        title = " ".join(str(item.get("title") or canonical_text).split()).strip()[:80]
        evidence = item.get("evidence")
        normalized_evidence = evidence if isinstance(evidence, list) else []
        if not normalized_evidence:
            normalized_evidence = [{"source": "leader_tool", "quote": content}]
        candidates.append(
            {
                "action": str(item.get("action") or "ADD").strip().upper() or "ADD",
                "memory_type": memory_type,
                "title": title or canonical_text[:80],
                "content": content,
                "canonical_text": canonical_text,
                "dedup_key": dedup_key,
                "confidence": float(item.get("confidence") or 0.9),
                "source_episode_uid": str(item.get("source_episode_uid") or ""),
                "evidence": normalized_evidence,
            }
        )
    if not candidates:
        return []
    return apply_memory_candidates(
        uuid=uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        candidates=candidates,
        db_name=db_name,
    )
