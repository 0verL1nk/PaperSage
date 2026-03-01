import sqlite3
from pathlib import Path

from utils.utils import save_content_to_database


def _create_strict_contents_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE contents (
            uid VARCHAR PRIMARY KEY,
            file_path VARCHAR NOT NULL,
            file_extraction TEXT,
            file_mindmap JSON,
            file_summary JSON,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def test_save_content_to_database_supports_strict_contents_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "strict.sqlite"
    _create_strict_contents_schema(db_path)

    save_content_to_database(
        uid="uid-1",
        file_path="/tmp/a.pdf",
        content="hello",
        content_type="file_extraction",
        db_name=str(db_path),
    )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT uid, file_path, file_extraction, created_at, updated_at FROM contents WHERE uid = ?",
        ("uid-1",),
    )
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    assert result[0] == "uid-1"
    assert result[2] == "hello"
    assert result[3]
    assert result[4]
