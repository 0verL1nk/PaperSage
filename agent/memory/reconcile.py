from typing import Any

from .store import list_project_memory_items, update_memory_item_status, upsert_project_memory_item


def apply_memory_candidates(
    *,
    uuid: str,
    project_uid: str,
    session_uid: str,
    candidates: list[dict[str, Any]],
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    active_items = list_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        status="active",
        limit=200,
        db_name=db_name,
    )
    for candidate in candidates:
        requested_action = str(candidate.get("action") or "ADD").strip().upper() or "ADD"
        dedup_key = str(candidate.get("dedup_key") or "").strip()
        canonical_text = str(candidate.get("canonical_text") or "").strip()
        existing = next(
            (
                item
                for item in active_items
                if str(item.get("dedup_key") or "").strip() == dedup_key
                and str(item.get("memory_type") or "").strip()
                == str(candidate.get("memory_type") or "").strip()
            ),
            None,
        )
        if requested_action == "DELETE" and existing:
            update_memory_item_status(
                memory_uid=str(existing.get("memory_uid") or ""),
                status="deleted",
                db_name=db_name,
            )
            results.append(
                {
                    "action": "DELETE",
                    "memory_uid": str(existing.get("memory_uid") or ""),
                    "dedup_key": dedup_key,
                }
            )
            active_items = [
                item
                for item in active_items
                if str(item.get("memory_uid") or "") != str(existing.get("memory_uid") or "")
            ]
            continue

        if existing and str(existing.get("canonical_text") or "").strip() == canonical_text:
            results.append(
                {
                    "action": "NONE",
                    "memory_uid": str(existing.get("memory_uid") or ""),
                    "dedup_key": dedup_key,
                }
            )
            continue

        if requested_action == "UPDATE" and existing:
            memory_uid = upsert_project_memory_item(
                uuid=uuid,
                project_uid=project_uid,
                session_uid=session_uid,
                memory_type=str(candidate.get("memory_type") or ""),
                title=str(candidate.get("title") or ""),
                content=str(candidate.get("content") or ""),
                canonical_text=canonical_text,
                dedup_key=dedup_key,
                status="active",
                confidence=float(candidate.get("confidence") or 0.0),
                source_episode_uid=str(candidate.get("source_episode_uid") or ""),
                evidence=list(candidate.get("evidence") or []),
                db_name=db_name,
            )
            results.append(
                {
                    "action": "UPDATE",
                    "memory_uid": memory_uid,
                    "dedup_key": dedup_key,
                }
            )
            for item in active_items:
                if str(item.get("memory_uid") or "") == memory_uid:
                    item.update(candidate)
                    item["status"] = "active"
            continue

        if existing:
            new_uid = upsert_project_memory_item(
                uuid=uuid,
                project_uid=project_uid,
                session_uid=session_uid,
                memory_type=str(candidate.get("memory_type") or ""),
                title=str(candidate.get("title") or ""),
                content=str(candidate.get("content") or ""),
                canonical_text=canonical_text,
                dedup_key=dedup_key,
                status="active",
                confidence=float(candidate.get("confidence") or 0.0),
                source_episode_uid=str(candidate.get("source_episode_uid") or ""),
                evidence=list(candidate.get("evidence") or []),
                allow_update=False,
                db_name=db_name,
            )
            update_memory_item_status(
                memory_uid=str(existing.get("memory_uid") or ""),
                status="superseded",
                superseded_by=new_uid,
                db_name=db_name,
            )
            results.append(
                {
                    "action": "SUPERSEDE",
                    "memory_uid": new_uid,
                    "previous_memory_uid": str(existing.get("memory_uid") or ""),
                    "dedup_key": dedup_key,
                }
            )
            active_items = [
                item
                for item in active_items
                if str(item.get("memory_uid") or "") != str(existing.get("memory_uid") or "")
            ]
            created_item = dict(candidate)
            created_item["memory_uid"] = new_uid
            created_item["status"] = "active"
            active_items.append(created_item)
            continue

        memory_uid = upsert_project_memory_item(
            uuid=uuid,
            project_uid=project_uid,
            session_uid=session_uid,
            memory_type=str(candidate.get("memory_type") or ""),
            title=str(candidate.get("title") or ""),
            content=str(candidate.get("content") or ""),
            canonical_text=canonical_text,
            dedup_key=dedup_key,
            status="active",
            confidence=float(candidate.get("confidence") or 0.0),
            source_episode_uid=str(candidate.get("source_episode_uid") or ""),
            evidence=list(candidate.get("evidence") or []),
            db_name=db_name,
        )
        results.append(
            {
                "action": "ADD",
                "memory_uid": memory_uid,
                "dedup_key": dedup_key,
            }
        )
        created_item = dict(candidate)
        created_item["memory_uid"] = memory_uid
        created_item["status"] = "active"
        active_items.append(created_item)
    return results
