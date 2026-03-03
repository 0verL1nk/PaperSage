import datetime
import sqlite3
from typing import Any


def ensure_agent_outputs_table(db_name: str = "./database.sqlite") -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL,
            project_uid TEXT,
            session_uid TEXT,
            doc_uid TEXT,
            doc_name TEXT,
            output_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )
    cursor.execute("PRAGMA table_info(agent_outputs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "project_uid" not in columns:
        cursor.execute("ALTER TABLE agent_outputs ADD COLUMN project_uid TEXT")
    if "session_uid" not in columns:
        cursor.execute("ALTER TABLE agent_outputs ADD COLUMN session_uid TEXT")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_outputs_uuid_doc ON agent_outputs(uuid, doc_uid, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_outputs_uuid_project ON agent_outputs(uuid, project_uid, created_at)"
    )
    conn.commit()
    conn.close()


def save_agent_output(
    *,
    uuid: str,
    output_type: str,
    content: str,
    project_uid: str | None = None,
    session_uid: str | None = None,
    doc_uid: str | None = None,
    doc_name: str | None = None,
    db_name: str = "./database.sqlite",
) -> None:
    ensure_agent_outputs_table(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO agent_outputs (
            uuid, project_uid, session_uid, doc_uid, doc_name, output_type, content, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            uuid,
            project_uid,
            session_uid,
            doc_uid,
            doc_name,
            output_type,
            content,
            created_at,
        ),
    )
    conn.commit()
    conn.close()


def list_agent_outputs(
    *,
    uuid: str,
    project_uid: str | None = None,
    doc_uid: str | None = None,
    limit: int = 30,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_agent_outputs_table(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if project_uid:
        cursor.execute(
            """
            SELECT id, uuid, project_uid, session_uid, doc_uid, doc_name, output_type, content, created_at
            FROM agent_outputs
            WHERE uuid = ? AND project_uid = ?
            ORDER BY id DESC
            LIMIT ?
        """,
            (uuid, project_uid, limit),
        )
    elif doc_uid:
        cursor.execute(
            """
            SELECT id, uuid, project_uid, session_uid, doc_uid, doc_name, output_type, content, created_at
            FROM agent_outputs
            WHERE uuid = ? AND doc_uid = ?
            ORDER BY id DESC
            LIMIT ?
        """,
            (uuid, doc_uid, limit),
        )
    else:
        cursor.execute(
            """
            SELECT id, uuid, project_uid, session_uid, doc_uid, doc_name, output_type, content, created_at
            FROM agent_outputs
            WHERE uuid = ?
            ORDER BY id DESC
            LIMIT ?
        """,
            (uuid, limit),
        )

    rows = cursor.fetchall()
    conn.close()

    outputs: list[dict[str, Any]] = []
    for row in rows:
        outputs.append(
            {
                "id": row[0],
                "uuid": row[1],
                "project_uid": row[2],
                "session_uid": row[3],
                "doc_uid": row[4],
                "doc_name": row[5],
                "output_type": row[6],
                "content": row[7],
                "created_at": row[8],
            }
        )
    return outputs
