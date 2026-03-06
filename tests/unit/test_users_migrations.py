import sqlite3

from utils.utils import _try_add_users_column, ensure_users_runtime_tuning_columns


class _DuplicateCursor:
    def execute(self, _sql: str):
        raise sqlite3.OperationalError("duplicate column name: agent_policy_async_enabled")


def test_try_add_users_column_ignores_duplicate_error():
    assert _try_add_users_column(_DuplicateCursor(), "ALTER TABLE users ADD COLUMN x TEXT") is False


def test_ensure_users_runtime_tuning_columns_is_idempotent(tmp_path):
    db_path = tmp_path / "migration.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE users (
            uuid TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            api_key TEXT DEFAULT NULL,
            model_name TEXT DEFAULT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    ensure_users_runtime_tuning_columns(str(db_path))
    ensure_users_runtime_tuning_columns(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "agent_policy_async_enabled" in columns
    assert "agent_policy_async_refresh_seconds" in columns
    assert "agent_policy_async_min_confidence" in columns
    assert "agent_policy_async_max_staleness_seconds" in columns
    assert "rag_index_batch_size" in columns
    assert "agent_document_text_cache_max_chars" in columns
    assert "local_rag_project_max_chars" in columns
    assert "local_rag_project_max_chunks" in columns
