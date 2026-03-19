from pathlib import Path

from agent.memory.service import write_memory_from_leader
from agent.memory.store import list_project_memory_items
from utils.utils import ensure_local_user, init_database


def test_write_memory_from_leader_deduplicates_active_memory(tmp_path: Path) -> None:
    db_path = tmp_path / "leader-memory.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    first = write_memory_from_leader(
        uuid="local-user",
        project_uid="project-1",
        session_uid="session-1",
        items=[
            {
                "memory_type": "user_memory",
                "content": "user prefers concise answers",
                "canonical_text": "user prefers concise answers",
            }
        ],
        db_name=str(db_path),
    )
    second = write_memory_from_leader(
        uuid="local-user",
        project_uid="project-1",
        session_uid="session-1",
        items=[
            {
                "memory_type": "user_memory",
                "content": "user prefers concise answers",
                "canonical_text": "user prefers concise answers",
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

    assert first[0]["action"] == "ADD"
    assert second[0]["action"] == "NONE"
    assert len(active_items) == 1
    assert active_items[0]["canonical_text"] == "user prefers concise answers"
