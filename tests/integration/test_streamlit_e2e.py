import sqlite3
from pathlib import Path

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare_legacy_files_db_without_uuid(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            uid TEXT NOT NULL,
            md5 TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def _prepare_minimal_user_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            uuid TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            api_key TEXT DEFAULT NULL,
            model_name TEXT DEFAULT NULL
        )
    """
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO users (uuid, username, password, api_key, model_name)
        VALUES ('local-user', 'local', 'local', 'test-key', NULL)
    """
    )
    conn.commit()
    conn.close()


def test_main_app_boots_without_login(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    at = AppTest.from_file(str(REPO_ROOT / "main.py"), default_timeout=8)
    at.run()

    assert len(at.exception) == 0
    assert any(item.value == "🤖 Agent 中心" for item in at.title)


def test_main_app_boots_with_legacy_files_schema(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_legacy_files_db_without_uuid(tmp_path / "database.sqlite")

    at = AppTest.from_file(str(REPO_ROOT / "main.py"))
    at.run()

    assert len(at.exception) == 0


def test_main_app_renders_file_list_without_pandas(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_minimal_user_db(tmp_path / "database.sqlite")

    conn = sqlite3.connect(tmp_path / "database.sqlite")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            uid TEXT NOT NULL,
            md5 TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uuid TEXT NOT NULL,
            created_at TEXT
        )
    """
    )
    cursor.execute(
        """
        INSERT INTO files (original_filename, uid, md5, file_path, uuid, created_at)
        VALUES ('paper.pdf', 'uid-1', 'md5-1', '/tmp/paper.pdf', 'local-user', '2026-01-01 00:00:00')
    """
    )
    conn.commit()
    conn.close()

    at = AppTest.from_file(str(REPO_ROOT / "main.py"))
    at.run()

    assert len(at.exception) == 0


def test_main_app_settings_include_model_and_base_url(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_minimal_user_db(tmp_path / "database.sqlite")

    at = AppTest.from_file(str(REPO_ROOT / "main.py"), default_timeout=8)
    at.run()

    labels = [item.label for item in at.text_input]
    assert "API Key:" in labels
    assert "模型名称:" in labels
    assert "OpenAI Compatible Base URL:" in labels


def test_settings_page_boots_with_expected_fields(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_minimal_user_db(tmp_path / "database.sqlite")

    at = AppTest.from_file(str(REPO_ROOT / "pages/2_⚙️_设置中心.py"))
    at.run()

    assert len(at.exception) == 0
    labels = [item.label for item in at.text_input]
    assert "API Key" in labels
    assert "模型名称" in labels
    assert "OpenAI Compatible Base URL" in labels
