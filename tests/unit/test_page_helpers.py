import json
from pathlib import Path

import streamlit as st

from utils import page_helpers
from utils.task_queue import TaskStatus
from utils.utils import ensure_local_user, init_database, save_api_key, save_model_name


def _prepare_user(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    st.session_state["uuid"] = "local-user"


def test_start_async_task_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_user(tmp_path)
    save_api_key("local-user", "key", db_name=str(tmp_path / "database.sqlite"))
    save_model_name(
        "local-user", "gpt-4o-mini", db_name=str(tmp_path / "database.sqlite")
    )

    captured = {}

    def fake_create_task(task_id: str, uid: str, content_type: str) -> None:
        captured["task_id"] = task_id
        captured["uid"] = uid
        captured["content_type"] = content_type

    def fake_enqueue_task(task_func, task_id: str, *args):
        captured["args"] = args
        return {"mode": "queued", "job_id": "job-1"}

    def fake_update_task_status(task_id: str, status: TaskStatus, job_id=None):
        captured["status"] = status
        captured["job_id"] = job_id

    monkeypatch.setattr(page_helpers, "create_task", fake_create_task)
    monkeypatch.setattr(page_helpers, "enqueue_task", fake_enqueue_task)
    monkeypatch.setattr(
        "utils.task_queue.update_task_status", fake_update_task_status, raising=True
    )

    task_id = page_helpers.start_async_task("uid-1", "file_summary", lambda: None, "a")

    assert task_id == captured["task_id"]
    assert captured["uid"] == "uid-1"
    assert captured["content_type"] == "file_summary"
    assert captured["job_id"] == "job-1"


def test_start_async_task_sync_mode_does_not_mark_queued(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_user(tmp_path)
    save_api_key("local-user", "key", db_name=str(tmp_path / "database.sqlite"))
    save_model_name(
        "local-user", "gpt-4o-mini", db_name=str(tmp_path / "database.sqlite")
    )

    called = {"updated": False}

    monkeypatch.setattr(page_helpers, "create_task", lambda *args: None)
    monkeypatch.setattr(
        page_helpers,
        "enqueue_task",
        lambda *args: {"mode": "sync", "job_id": None},
    )

    def fake_update_task_status(*_args, **_kwargs):
        called["updated"] = True

    monkeypatch.setattr(
        "utils.task_queue.update_task_status", fake_update_task_status, raising=True
    )

    task_id = page_helpers.start_async_task("uid-1", "file_summary", lambda: None)

    assert isinstance(task_id, str)
    assert not called["updated"]


def test_start_async_task_returns_none_when_not_configured(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_user(tmp_path)

    monkeypatch.setattr(page_helpers, "check_api_key_configured", lambda: (False, "x"))

    task_id = page_helpers.start_async_task("uid-1", "file_summary", lambda: None)
    assert task_id is None


def test_start_async_task_returns_none_when_queue_fails(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_user(tmp_path)
    save_api_key("local-user", "key", db_name=str(tmp_path / "database.sqlite"))
    save_model_name(
        "local-user", "gpt-4o-mini", db_name=str(tmp_path / "database.sqlite")
    )

    monkeypatch.setattr(page_helpers, "create_task", lambda *args: None)
    monkeypatch.setattr(page_helpers, "enqueue_task", lambda *args: None)

    task_id = page_helpers.start_async_task("uid-1", "file_summary", lambda: None)
    assert task_id is None


def test_check_task_and_content_returns_existing_summary(monkeypatch) -> None:
    monkeypatch.setattr(page_helpers, "get_task_status_by_uid", lambda uid, ctype: None)
    monkeypatch.setattr("utils.utils.get_content_by_uid", lambda uid, ctype: "summary")

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_summary"
    )

    assert content == {"summary": "summary"}
    assert status is None
    assert task_id is None


def test_check_task_and_content_tracks_running_task(monkeypatch) -> None:
    monkeypatch.setattr("utils.utils.get_content_by_uid", lambda uid, ctype: None)
    monkeypatch.setattr(
        page_helpers,
        "get_task_status_by_uid",
        lambda uid, ctype: {
            "status": "started",
            "task_id": "task-1",
            "job_id": "job-1",
        },
    )
    monkeypatch.setattr(page_helpers, "get_job_status", lambda job_id: "started")

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_summary"
    )

    assert content is None
    assert status == TaskStatus.STARTED.value
    assert task_id == "task-1"


def test_check_task_and_content_reads_finished_json(monkeypatch) -> None:
    payload = json.dumps({"name": "root", "children": []})

    states = {"count": 0}

    def fake_content(uid, ctype):
        states["count"] += 1
        if states["count"] == 1:
            return None
        return payload

    monkeypatch.setattr("utils.utils.get_content_by_uid", fake_content)
    monkeypatch.setattr(
        page_helpers,
        "get_task_status_by_uid",
        lambda uid, ctype: {"status": "finished", "task_id": "task-2", "job_id": None},
    )

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_mindmap"
    )

    assert content == {"name": "root", "children": []}
    assert status is None
    assert task_id is None


def test_check_task_and_content_parses_existing_mindmap_json(monkeypatch) -> None:
    payload = json.dumps({"name": "root", "children": []})
    monkeypatch.setattr("utils.utils.get_content_by_uid", lambda uid, ctype: payload)
    monkeypatch.setattr(page_helpers, "get_task_status_by_uid", lambda uid, ctype: None)

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_mindmap"
    )

    assert content == {"name": "root", "children": []}
    assert status is None
    assert task_id is None


def test_check_task_and_content_updates_status_from_rq(monkeypatch) -> None:
    monkeypatch.setattr("utils.utils.get_content_by_uid", lambda uid, ctype: None)
    monkeypatch.setattr(
        page_helpers,
        "get_task_status_by_uid",
        lambda uid, ctype: {"status": "queued", "task_id": "task-3", "job_id": "job-3"},
    )
    monkeypatch.setattr(page_helpers, "get_job_status", lambda job_id: "failed")

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_summary"
    )

    assert content is None
    assert status == TaskStatus.FAILED.value
    assert task_id == "task-3"


def test_check_task_and_content_marks_stale_queue_failed_without_workers(
    monkeypatch,
) -> None:
    monkeypatch.setattr("utils.utils.get_content_by_uid", lambda uid, ctype: None)
    monkeypatch.setattr(
        page_helpers,
        "get_task_status_by_uid",
        lambda uid, ctype: {"status": "queued", "task_id": "task-4", "job_id": "job-4"},
    )
    monkeypatch.setattr(page_helpers, "get_job_status", lambda job_id: None)
    monkeypatch.setattr(page_helpers, "has_active_rq_workers", lambda: False)

    called = {"updated": False}

    def fake_update_task_status(*_args, **_kwargs):
        called["updated"] = True

    monkeypatch.setattr(
        "utils.task_queue.update_task_status", fake_update_task_status, raising=True
    )

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_summary"
    )

    assert content is None
    assert status is None
    assert task_id is None
    assert called["updated"]


def test_check_task_and_content_auto_start_branch(monkeypatch) -> None:
    monkeypatch.setattr("utils.utils.get_content_by_uid", lambda uid, ctype: None)
    monkeypatch.setattr(page_helpers, "get_task_status_by_uid", lambda uid, ctype: None)

    content, status, task_id = page_helpers.check_task_and_content(
        "uid-1", "file_summary", auto_start=True
    )

    assert content is None
    assert status is None
    assert task_id is None


def test_display_task_status_calls_streamlit_methods(monkeypatch) -> None:
    called = {"info": 0, "error": 0, "success": 0, "rerun": 0}

    monkeypatch.setattr(
        st, "info", lambda msg: called.__setitem__("info", called["info"] + 1)
    )
    monkeypatch.setattr(
        st, "error", lambda msg: called.__setitem__("error", called["error"] + 1)
    )
    monkeypatch.setattr(
        st, "success", lambda msg: called.__setitem__("success", called["success"] + 1)
    )
    monkeypatch.setattr(
        st, "rerun", lambda: called.__setitem__("rerun", called["rerun"] + 1)
    )
    monkeypatch.setattr("time.sleep", lambda n: None)

    page_helpers.display_task_status(TaskStatus.STARTED.value)
    page_helpers.display_task_status(TaskStatus.FAILED.value, error_message="x")
    page_helpers.display_task_status(TaskStatus.FINISHED.value)

    assert called["info"] >= 1
    assert called["error"] >= 1
    assert called["success"] >= 1
    assert called["rerun"] >= 1
