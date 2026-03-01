import sqlite3
from pathlib import Path

from utils.utils import (
    ensure_local_user,
    get_model_name,
    get_user_files,
    init_database,
    save_api_key,
    save_model_name,
)
from utils.page_helpers import check_api_key_configured


def _create_legacy_db_without_files_uuid(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE users (
            uuid TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            api_key TEXT DEFAULT NULL
        )
    """
    )
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
    cursor.execute(
        """
        INSERT INTO files (original_filename, uid, md5, file_path)
        VALUES ('paper.pdf', 'uid-1', 'md5-1', '/tmp/paper.pdf')
    """
    )
    conn.commit()
    conn.close()


def test_init_database_migrates_legacy_files_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    _create_legacy_db_without_files_uuid(db_path)

    init_database(str(db_path))
    ensure_local_user(db_name=str(db_path))

    files = get_user_files("local-user", db_name=str(db_path))
    assert len(files) == 1
    assert files[0]["file_name"] == "paper.pdf"
    assert files[0]["uid"] == "uid-1"


def test_init_database_adds_created_at_for_legacy_files(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    _create_legacy_db_without_files_uuid(db_path)

    init_database(str(db_path))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(files)")
    column_names = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "uuid" in column_names
    assert "created_at" in column_names


def test_check_api_key_configured_requires_model_name(
    tmp_path: Path, monkeypatch
) -> None:
    db_path = tmp_path / "database.sqlite"
    monkeypatch.chdir(tmp_path)

    init_database(str(db_path))
    ensure_local_user("local-user", db_name=str(db_path))

    import streamlit as st

    st.session_state["uuid"] = "local-user"

    is_ok, error = check_api_key_configured()
    assert not is_ok
    assert error == "请先在侧边栏设置中配置您的 API Key"

    save_api_key("local-user", "test-api-key", db_name=str(db_path))
    is_ok, error = check_api_key_configured()
    assert not is_ok
    assert error == "请先在侧边栏设置中配置模型名称"

    save_model_name("local-user", "gpt-4o-mini", db_name=str(db_path))
    assert get_model_name("local-user", db_name=str(db_path)) == "gpt-4o-mini"
    is_ok, error = check_api_key_configured()
    assert is_ok
    assert error is None
