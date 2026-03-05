import datetime
import json
import sqlite3
from typing import Any
from uuid import uuid4


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_memory_tables(db_name: str = "./database.sqlite") -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS session_compact_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uid TEXT NOT NULL,
            project_uid TEXT NOT NULL,
            uuid TEXT NOT NULL,
            compact_summary TEXT DEFAULT '',
            anchors_json TEXT DEFAULT '[]',
            updated_at TEXT NOT NULL,
            UNIQUE(session_uid, project_uid, uuid)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_uid TEXT UNIQUE NOT NULL,
            uuid TEXT NOT NULL,
            project_uid TEXT NOT NULL,
            session_uid TEXT,
            memory_type TEXT NOT NULL,
            title TEXT DEFAULT '',
            content TEXT NOT NULL,
            source_prompt TEXT DEFAULT '',
            source_answer TEXT DEFAULT '',
            access_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT,
            expires_at TEXT DEFAULT ''
        )
    """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_compact_scope ON session_compact_memory(session_uid, project_uid, uuid)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(uuid, project_uid, memory_type, updated_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_items_session ON memory_items(session_uid, updated_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_items_expire ON memory_items(expires_at)"
    )
    cursor.execute("PRAGMA table_info(memory_items)")
    memory_columns = {row[1] for row in cursor.fetchall()}
    if "expires_at" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN expires_at TEXT DEFAULT ''")
    conn.commit()
    conn.close()


def get_project_session_compact_memory(
    session_uid: str,
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    ensure_memory_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT compact_summary, anchors_json, updated_at
        FROM session_compact_memory
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
        LIMIT 1
    """,
        (session_uid, project_uid, uuid),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"compact_summary": "", "anchors": [], "updated_at": ""}

    compact_summary = str(row[0] or "")
    anchors_raw = row[1]
    anchors: list[Any] = []
    if isinstance(anchors_raw, str) and anchors_raw.strip():
        try:
            parsed = json.loads(anchors_raw)
            if isinstance(parsed, list):
                anchors = parsed
        except Exception:
            anchors = []
    return {
        "compact_summary": compact_summary,
        "anchors": anchors,
        "updated_at": str(row[2] or ""),
    }


def save_project_session_compact_memory(
    session_uid: str,
    project_uid: str,
    uuid: str,
    compact_summary: str,
    anchors: list[dict[str, Any]] | None = None,
    db_name: str = "./database.sqlite",
) -> None:
    ensure_memory_tables(db_name)
    now = _now_str()
    anchors_payload = anchors if isinstance(anchors, list) else []
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO session_compact_memory (
            session_uid, project_uid, uuid, compact_summary, anchors_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_uid, project_uid, uuid) DO UPDATE SET
            compact_summary = excluded.compact_summary,
            anchors_json = excluded.anchors_json,
            updated_at = excluded.updated_at
    """,
        (
            session_uid,
            project_uid,
            uuid,
            str(compact_summary or ""),
            json.dumps(anchors_payload, ensure_ascii=False),
            now,
        ),
    )
    conn.commit()
    conn.close()


def upsert_project_memory_item(
    *,
    uuid: str,
    project_uid: str,
    session_uid: str | None,
    memory_type: str,
    content: str,
    title: str = "",
    source_prompt: str = "",
    source_answer: str = "",
    expires_at: str = "",
    db_name: str = "./database.sqlite",
) -> str:
    ensure_memory_tables(db_name)
    normalized_content = str(content or "").strip()
    if not normalized_content:
        return ""
    normalized_type = str(memory_type or "episodic").strip().lower() or "episodic"
    now = _now_str()
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT memory_uid
        FROM memory_items
        WHERE uuid = ? AND project_uid = ? AND memory_type = ? AND content = ?
        LIMIT 1
    """,
        (uuid, project_uid, normalized_type, normalized_content),
    )
    row = cursor.fetchone()
    if row:
        memory_uid = str(row[0] or "")
        cursor.execute(
            """
            UPDATE memory_items
            SET session_uid = ?, title = ?, source_prompt = ?, source_answer = ?, updated_at = ?, expires_at = ?
            WHERE memory_uid = ?
        """,
            (
                str(session_uid or ""),
                str(title or ""),
                str(source_prompt or ""),
                str(source_answer or ""),
                now,
                str(expires_at or ""),
                memory_uid,
            ),
        )
        conn.commit()
        conn.close()
        return memory_uid

    memory_uid = uuid4().hex
    cursor.execute(
        """
        INSERT INTO memory_items (
            memory_uid, uuid, project_uid, session_uid, memory_type, title, content,
            source_prompt, source_answer, access_count, created_at, updated_at, last_accessed_at, expires_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, '', ?)
    """,
        (
            memory_uid,
            uuid,
            project_uid,
            str(session_uid or ""),
            normalized_type,
            str(title or ""),
            normalized_content,
            str(source_prompt or ""),
            str(source_answer or ""),
            now,
            now,
            str(expires_at or ""),
        ),
    )
    conn.commit()
    conn.close()
    return memory_uid


def list_project_memory_items(
    *,
    uuid: str,
    project_uid: str,
    memory_type: str | None = None,
    limit: int = 100,
    include_expired: bool = False,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_memory_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    resolved_limit = max(1, int(limit))
    now = _now_str()
    expire_condition = ""
    params: list[Any] = [uuid, project_uid]
    if not include_expired:
        expire_condition = "AND (expires_at = '' OR expires_at > ?)"
        params.append(now)
    if isinstance(memory_type, str) and memory_type.strip():
        params.append(memory_type.strip().lower())
        params.append(resolved_limit)
        cursor.execute(
            """
            SELECT memory_uid, session_uid, memory_type, title, content, source_prompt,
                   source_answer, access_count, created_at, updated_at, last_accessed_at, expires_at
            FROM memory_items
            WHERE uuid = ? AND project_uid = ? AND memory_type = ?
            {expire_condition}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
        """.format(expire_condition=expire_condition),
            tuple(params),
        )
    else:
        params.append(resolved_limit)
        cursor.execute(
            """
            SELECT memory_uid, session_uid, memory_type, title, content, source_prompt,
                   source_answer, access_count, created_at, updated_at, last_accessed_at, expires_at
            FROM memory_items
            WHERE uuid = ? AND project_uid = ?
            {expire_condition}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
        """.format(expire_condition=expire_condition),
            tuple(params),
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "memory_uid": str(row[0] or ""),
            "session_uid": str(row[1] or ""),
            "memory_type": str(row[2] or ""),
            "title": str(row[3] or ""),
            "content": str(row[4] or ""),
            "source_prompt": str(row[5] or ""),
            "source_answer": str(row[6] or ""),
            "access_count": int(row[7] or 0),
            "created_at": str(row[8] or ""),
            "updated_at": str(row[9] or ""),
            "last_accessed_at": str(row[10] or ""),
            "expires_at": str(row[11] or ""),
        }
        for row in rows
    ]


def touch_memory_items(
    *,
    memory_uids: list[str],
    db_name: str = "./database.sqlite",
) -> None:
    if not memory_uids:
        return
    ensure_memory_tables(db_name)
    now = _now_str()
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    for memory_uid in memory_uids:
        normalized_uid = str(memory_uid or "").strip()
        if not normalized_uid:
            continue
        cursor.execute(
            """
            UPDATE memory_items
            SET access_count = access_count + 1, last_accessed_at = ?
            WHERE memory_uid = ?
        """,
            (now, normalized_uid),
        )
    conn.commit()
    conn.close()
