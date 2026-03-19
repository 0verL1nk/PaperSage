from pathlib import Path

from agent.adapters.sqlite.project_repository import ensure_default_project_session
from agent.memory.store import list_project_memory_items, save_project_memory_episode
from utils.tasks import task_memory_writer
from utils.utils import ensure_local_user, init_database


def _prepare_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    return db_path


def _fake_candidates(*, episode, recent_episodes, active_memories, user_uuid):
    del recent_episodes, active_memories, user_uuid
    prompt = str(episode["prompt"])
    answer = str(episode["answer"])
    canonical = answer.replace("收到，后续默认用", "").replace("回答。", "").strip("。")
    return [
        {
            "action": "ADD",
            "memory_type": "user_memory",
            "title": prompt,
            "content": f"{prompt}\n{answer}",
            "canonical_text": canonical or answer,
            "dedup_key": "user:response_preferences",
            "confidence": 0.9,
            "source_episode_uid": str(episode["episode_uid"]),
            "evidence": [{"episode_uid": str(episode["episode_uid"]), "quote": prompt}],
        }
    ]


def test_task_memory_writer_reads_episode_and_bounded_context(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.tasks.extract_memory_candidates", _fake_candidates)
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
    assert isinstance(payload["candidates"], list)


def test_task_memory_writer_is_idempotent_for_same_episode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.tasks.extract_memory_candidates", _fake_candidates)
    db_path = _prepare_db(tmp_path)
    project_uid = "project-1"
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    episode_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="请记住，以后默认用中文回答",
        answer="收到，后续默认用中文回答。",
        db_name=str(db_path),
    )

    ok_first, _ = task_memory_writer("task-1", episode_uid, "local-user", db_name=str(db_path))
    ok_second, _ = task_memory_writer("task-2", episode_uid, "local-user", db_name=str(db_path))

    assert ok_first is True
    assert ok_second is True
    active_items = list_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        status="active",
        limit=10,
        db_name=str(db_path),
    )
    assert len(active_items) == 1
    assert active_items[0]["memory_type"] == "user_memory"


def test_task_memory_writer_supersedes_old_user_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.tasks.extract_memory_candidates", _fake_candidates)
    db_path = _prepare_db(tmp_path)
    project_uid = "project-1"
    session_uid = ensure_default_project_session(
        project_uid=project_uid,
        uuid="local-user",
        db_name=str(db_path),
    )
    first_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="请记住，以后默认用中文回答",
        answer="收到，后续默认用中文回答。",
        db_name=str(db_path),
    )
    second_uid = save_project_memory_episode(
        uuid="local-user",
        project_uid=project_uid,
        session_uid=session_uid,
        prompt="请记住，以后默认用英文回答",
        answer="收到，后续默认用英文回答。",
        db_name=str(db_path),
    )

    assert task_memory_writer("task-1", first_uid, "local-user", db_name=str(db_path))[0] is True
    assert task_memory_writer("task-2", second_uid, "local-user", db_name=str(db_path))[0] is True

    active_items = list_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        status="active",
        limit=10,
        db_name=str(db_path),
    )
    superseded_items = list_project_memory_items(
        uuid="local-user",
        project_uid=project_uid,
        status="superseded",
        limit=10,
        db_name=str(db_path),
    )

    assert len(active_items) == 1
    assert "英文" in active_items[0]["canonical_text"]
    assert len(superseded_items) == 1
    assert "中文" in superseded_items[0]["canonical_text"]
