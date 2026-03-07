import sqlite3
from pathlib import Path

from streamlit.testing.v1 import AppTest
from utils.utils import add_file_to_project, ensure_default_project_for_user


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


def _prepare_configured_user_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            uuid TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            api_key TEXT DEFAULT NULL,
            model_name TEXT DEFAULT NULL,
            base_url TEXT DEFAULT NULL
        )
    """
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO users (uuid, username, password, api_key, model_name, base_url)
        VALUES ('local-user', 'local', 'local', 'test-key', 'gpt-4o-mini', 'https://example.com/v1')
    """
    )
    conn.commit()
    conn.close()


def _bind_file_to_default_project(db_path: Path, file_uid: str) -> None:
    project_uid = ensure_default_project_for_user(
        "local-user",
        db_name=str(db_path),
        sync_existing_files=False,
    )
    add_file_to_project(
        project_uid=project_uid,
        file_uid=file_uid,
        uuid="local-user",
        db_name=str(db_path),
    )


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
    assert "API Key:" not in labels
    assert "模型名称:" not in labels
    assert "OpenAI Compatible Base URL:" not in labels


def test_settings_page_boots_with_expected_fields(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_minimal_user_db(tmp_path / "database.sqlite")

    at = AppTest.from_file(str(REPO_ROOT / "pages/2_settings.py"))
    at.run()

    assert len(at.exception) == 0
    labels = [item.label for item in at.text_input]
    assert "API Key" in labels
    assert "模型名称" in labels
    assert "OpenAI Compatible Base URL" in labels


def test_settings_page_save_persists_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_minimal_user_db(tmp_path / "database.sqlite")

    at = AppTest.from_file(str(REPO_ROOT / "pages/2_settings.py"), default_timeout=8)
    at.run()

    at.text_input[0].set_value("new-key")
    at.text_input[1].set_value("new-model")
    at.text_input[2].set_value("https://new.example.com/v1")
    at.button[0].click().run()

    assert any(item.value == "设置已保存" for item in at.success)

    conn = sqlite3.connect(tmp_path / "database.sqlite")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT api_key, model_name, base_url FROM users WHERE uuid = ?",
        ("local-user",),
    )
    result = cursor.fetchone()
    conn.close()

    assert result == ("new-key", "new-model", "https://new.example.com/v1")


def test_main_app_shows_no_files_message_for_configured_user(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_configured_user_db(tmp_path / "database.sqlite")

    at = AppTest.from_file(str(REPO_ROOT / "main.py"), default_timeout=8)
    at.run()

    assert len(at.exception) == 0
    assert any("暂无文档" in item.value for item in at.markdown)


def test_file_center_page_renders_existing_file_list(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _prepare_configured_user_db(tmp_path / "database.sqlite")

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
        VALUES ('existing-paper.pdf', 'uid-9', 'md5-9', '/tmp/existing-paper.pdf', 'local-user', '2026-02-01 10:00:00')
    """
    )
    conn.commit()
    conn.close()
    _bind_file_to_default_project(tmp_path / "database.sqlite", "uid-9")

    at = AppTest.from_file(str(REPO_ROOT / "pages/1_file_center.py"), default_timeout=8)
    at.run()

    assert len(at.exception) == 0
    markdown_values = [item.value for item in at.markdown]
    assert "### 已上传文档" in markdown_values
    assert "existing-paper.pdf" in markdown_values
    assert "2026-02-01 10:00:00" in markdown_values


def test_main_app_warns_when_model_missing_with_uploaded_file(
    monkeypatch, tmp_path: Path
) -> None:
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
        VALUES ('paper.pdf', 'uid-2', 'md5-2', '/tmp/paper.pdf', 'local-user', '2026-01-01 00:00:00')
    """
    )
    conn.commit()
    conn.close()

    at = AppTest.from_file(str(REPO_ROOT / "main.py"), default_timeout=8)
    at.run()

    warning_values = [item.value for item in at.warning]
    assert any("设置中心" in value and "模型名称" in value for value in warning_values)


def test_main_app_uses_persisted_extraction_cache(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    import agent.adapters.rag as rag_adapter_module
    import agent.rag.hybrid as rag_hybrid_module

    fake_retriever_builder = (
        lambda document_text, doc_uid="", doc_name="", project_uid="", **kwargs: (
            lambda query: {"evidences": [], "trace": {"mode": "test"}}
        )
    )
    monkeypatch.setattr(
        rag_hybrid_module,
        "build_local_evidence_retriever_with_settings",
        fake_retriever_builder,
    )
    monkeypatch.setattr(
        rag_adapter_module,
        "build_local_evidence_retriever_with_settings",
        fake_retriever_builder,
    )
    _prepare_configured_user_db(tmp_path / "database.sqlite")

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
        CREATE TABLE IF NOT EXISTS contents (
            uid TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            file_extraction TEXT,
            file_mindmap TEXT,
            file_summary TEXT
        )
    """
    )
    cursor.execute(
        """
        INSERT INTO files (original_filename, uid, md5, file_path, uuid, created_at)
        VALUES ('paper.pdf', 'uid-cache-1', 'md5-cache-1', '/tmp/missing-paper.pdf', 'local-user', '2026-03-02 22:20:00')
    """
    )
    cursor.execute(
        """
        INSERT INTO contents (uid, file_path, file_extraction)
        VALUES ('uid-cache-1', '/tmp/missing-paper.pdf', 'cached extracted text')
    """
    )
    conn.commit()
    conn.close()
    _bind_file_to_default_project(tmp_path / "database.sqlite", "uid-cache-1")

    at = AppTest.from_file(str(REPO_ROOT / "main.py"), default_timeout=8)
    at.run()

    assert len(at.exception) == 0
    caption_values = [item.value for item in at.caption]
    assert any("数据库缓存恢复" in value for value in caption_values)
