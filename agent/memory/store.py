from typing import Any

from .repository import (
    ensure_memory_tables,
    get_project_memory_episode,
    get_project_session_compact_memory,
    list_project_memory_episodes,
    list_project_memory_items,
    save_project_memory_episode,
    save_project_session_compact_memory,
    touch_memory_items,
    update_memory_item_status,
    upsert_project_memory_item,
)
from .service import search_project_memory_items

__all__ = [
    "ensure_memory_tables",
    "save_project_memory_episode",
    "get_project_memory_episode",
    "list_project_memory_episodes",
    "get_project_session_compact_memory",
    "save_project_session_compact_memory",
    "upsert_project_memory_item",
    "update_memory_item_status",
    "list_project_memory_items",
    "touch_memory_items",
    "search_project_memory_items",
]


def ensure_memory_layer_ready(db_name: str = "./database.sqlite") -> None:
    ensure_memory_tables(db_name=db_name)


def query_long_term_memory(
    *,
    uuid: str,
    project_uid: str,
    query: str,
    limit: int = 5,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    return search_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        query=query,
        limit=limit,
        db_name=db_name,
    )
