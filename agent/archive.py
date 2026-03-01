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
            doc_uid TEXT,
            doc_name TEXT,
            output_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_outputs_uuid_doc ON agent_outputs(uuid, doc_uid, created_at)"
    )
    conn.commit()
    conn.close()


def save_agent_output(
    *,
    uuid: str,
    output_type: str,
    content: str,
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
        INSERT INTO agent_outputs (uuid, doc_uid, doc_name, output_type, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (uuid, doc_uid, doc_name, output_type, content, created_at),
    )
    conn.commit()
    conn.close()


def list_agent_outputs(
    *,
    uuid: str,
    doc_uid: str | None = None,
    limit: int = 30,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_agent_outputs_table(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if doc_uid:
        cursor.execute(
            """
            SELECT id, uuid, doc_uid, doc_name, output_type, content, created_at
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
            SELECT id, uuid, doc_uid, doc_name, output_type, content, created_at
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
                "doc_uid": row[2],
                "doc_name": row[3],
                "output_type": row[4],
                "content": row[5],
                "created_at": row[6],
            }
        )
    return outputs
