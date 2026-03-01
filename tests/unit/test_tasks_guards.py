from pathlib import Path

from utils import tasks as task_module
from utils.utils import ensure_local_user, init_database, save_api_key


def _prepare_db(tmp_path: Path) -> None:
    db_path = tmp_path / "database.sqlite"
    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))
    save_api_key("local-user", "test-key", db_name=str(db_path))


def test_task_text_extraction_requires_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_db(tmp_path)

    monkeypatch.setattr(
        task_module, "extract_files", lambda p: {"result": 1, "text": "doc"}
    )
    monkeypatch.setattr(task_module, "update_task_status", lambda *args, **kwargs: None)

    ok, message = task_module.task_text_extraction(
        "task-1", "fake.pdf", "uid-1", "local-user"
    )

    assert not ok
    assert message == "请先在设置中配置模型名称"


def test_task_file_summary_requires_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_db(tmp_path)

    monkeypatch.setattr(
        task_module, "extract_files", lambda p: {"result": 1, "text": "doc"}
    )
    monkeypatch.setattr(task_module, "update_task_status", lambda *args, **kwargs: None)

    ok, message = task_module.task_file_summary(
        "task-2", "fake.pdf", "uid-1", "local-user"
    )

    assert not ok
    assert message == "请先在设置中配置模型名称"


def test_task_generate_mindmap_requires_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_db(tmp_path)

    monkeypatch.setattr(
        task_module, "extract_files", lambda p: {"result": 1, "text": "doc"}
    )
    monkeypatch.setattr(task_module, "update_task_status", lambda *args, **kwargs: None)

    ok, payload = task_module.task_generate_mindmap(
        "task-3", "fake.pdf", "uid-1", "local-user"
    )

    assert not ok
    assert payload is None
