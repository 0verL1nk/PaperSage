from pathlib import Path

from agent.memory.reconcile import apply_memory_candidates
from agent.memory.store import list_project_memory_items, upsert_project_memory_item
from utils.utils import ensure_local_user, init_database


def _prepare_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    return db_path


def test_apply_memory_candidates_updates_existing_active_item(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path)
    upsert_project_memory_item(
        uuid="local-user",
        project_uid="project-1",
        session_uid="session-1",
        memory_type="knowledge_memory",
        title="结论",
        content="旧结论",
        canonical_text="旧结论",
        dedup_key="knowledge:conclusion",
        status="active",
        db_name=str(db_path),
    )

    results = apply_memory_candidates(
        uuid="local-user",
        project_uid="project-1",
        session_uid="session-1",
        candidates=[
            {
                "action": "UPDATE",
                "memory_type": "knowledge_memory",
                "title": "结论",
                "content": "新结论",
                "canonical_text": "新结论",
                "dedup_key": "knowledge:conclusion",
                "confidence": 0.8,
                "source_episode_uid": "ep-1",
                "evidence": [{"episode_uid": "ep-1", "quote": "新结论"}],
            }
        ],
        db_name=str(db_path),
    )

    items = list_project_memory_items(
        uuid="local-user",
        project_uid="project-1",
        status="active",
        limit=10,
        db_name=str(db_path),
    )
    assert results[0]["action"] == "UPDATE"
    assert len(items) == 1
    assert items[0]["canonical_text"] == "新结论"


def test_apply_memory_candidates_deletes_existing_active_item(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path)
    upsert_project_memory_item(
        uuid="local-user",
        project_uid="project-1",
        session_uid="session-1",
        memory_type="user_memory",
        title="语言偏好",
        content="默认用中文回答",
        canonical_text="默认用中文回答",
        dedup_key="user:response_preferences",
        status="active",
        db_name=str(db_path),
    )

    results = apply_memory_candidates(
        uuid="local-user",
        project_uid="project-1",
        session_uid="session-1",
        candidates=[
            {
                "action": "DELETE",
                "memory_type": "user_memory",
                "title": "语言偏好",
                "content": "",
                "canonical_text": "默认用中文回答",
                "dedup_key": "user:response_preferences",
                "confidence": 0.9,
                "source_episode_uid": "ep-2",
                "evidence": [{"episode_uid": "ep-2", "quote": "删除该偏好"}],
            }
        ],
        db_name=str(db_path),
    )

    active_items = list_project_memory_items(
        uuid="local-user",
        project_uid="project-1",
        status="active",
        limit=10,
        db_name=str(db_path),
    )
    deleted_items = list_project_memory_items(
        uuid="local-user",
        project_uid="project-1",
        status="deleted",
        limit=10,
        db_name=str(db_path),
    )
    assert results[0]["action"] == "DELETE"
    assert active_items == []
    assert len(deleted_items) == 1
