from pathlib import Path

from agent.adapters.sqlite.project_repository import ensure_default_project_session
from agent.memory.store import save_project_memory_episode
from utils.tasks import task_memory_writer
from utils.utils import ensure_local_user, init_database


def _prepare_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    return db_path


def test_task_memory_writer_reads_episode_and_bounded_context(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path)
    project_uid = "project-1"
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    earlier_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="第一个问题",
        answer="第一个回答",
        db_name=str(db_path),
    )
    current_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="第二个问题",
        answer="第二个回答",
        db_name=str(db_path),
    )

    ok, payload = task_memory_writer(
        "task-1",
        current_uid,
        "local-user",
        db_name=str(db_path),
        context_limit=2,
    )

    assert ok is True
    assert payload["episode"]["episode_uid"] == current_uid
    assert payload["episode"]["prompt"] == "第二个问题"
    assert [item["episode_uid"] for item in payload["recent_episodes"]] == [current_uid, earlier_uid]
