import datetime
import json
import sqlite3
from typing import Any
from uuid import uuid4


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_dumps(value: Any, *, fallback: Any) -> str:
    payload = value if value is not None else fallback
    return json.dumps(payload, ensure_ascii=False)


def _ensure_memory_item_columns(cursor: sqlite3.Cursor) -> None:
    cursor.execute("PRAGMA table_info(memory_items)")
    memory_columns = {row[1] for row in cursor.fetchall()}
    alter_statements = {
        "expires_at": "ALTER TABLE memory_items ADD COLUMN expires_at TEXT DEFAULT ''",
        "canonical_text": "ALTER TABLE memory_items ADD COLUMN canonical_text TEXT DEFAULT ''",
        "dedup_key": "ALTER TABLE memory_items ADD COLUMN dedup_key TEXT DEFAULT ''",
        "status": "ALTER TABLE memory_items ADD COLUMN status TEXT DEFAULT 'active'",
        "confidence": "ALTER TABLE memory_items ADD COLUMN confidence REAL DEFAULT 0",
        "source_episode_uid": "ALTER TABLE memory_items ADD COLUMN source_episode_uid TEXT DEFAULT ''",
        "evidence_json": "ALTER TABLE memory_items ADD COLUMN evidence_json TEXT DEFAULT '[]'",
        "superseded_by": "ALTER TABLE memory_items ADD COLUMN superseded_by TEXT DEFAULT ''",
    }
    for column, statement in alter_statements.items():
        if column not in memory_columns:
            cursor.execute(statement)


def _parse_json_list(value: Any) -> list[Any]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _row_to_memory_item(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
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
        "canonical_text": str(row[12] or ""),
        "dedup_key": str(row[13] or ""),
        "status": str(row[14] or "active"),
        "confidence": float(row[15] or 0.0),
        "source_episode_uid": str(row[16] or ""),
        "evidence": _parse_json_list(row[17]),
        "superseded_by": str(row[18] or ""),
    }


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
        CREATE TABLE IF NOT EXISTS memory_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_uid TEXT UNIQUE NOT NULL,
            uuid TEXT NOT NULL,
            project_uid TEXT NOT NULL,
            session_uid TEXT NOT NULL,
            prompt TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
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
            expires_at TEXT DEFAULT '',
            canonical_text TEXT DEFAULT '',
            dedup_key TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            confidence REAL DEFAULT 0,
            source_episode_uid TEXT DEFAULT '',
            evidence_json TEXT DEFAULT '[]',
            superseded_by TEXT DEFAULT ''
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_item_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_uid TEXT NOT NULL,
            episode_uid TEXT NOT NULL,
            evidence_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    )
    _ensure_memory_item_columns(cursor)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_compact_scope ON session_compact_memory(session_uid, project_uid, uuid)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_episodes_scope ON memory_episodes(uuid, project_uid, session_uid, created_at DESC)"
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
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_items_active_lookup ON memory_items(uuid, project_uid, status, updated_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_items_dedup_key ON memory_items(dedup_key)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_items_episode ON memory_items(source_episode_uid, updated_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_item_evidence_memory ON memory_item_evidence(memory_uid, episode_uid)"
    )
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
    return {
        "compact_summary": compact_summary,
        "anchors": _parse_json_list(row[1]),
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
            _json_dumps(anchors, fallback=[]),
            now,
        ),
    )
    conn.commit()
    conn.close()


def save_project_memory_episode(
    *,
    uuid: str,
    project_uid: str,
    session_uid: str,
    prompt: str,
    answer: str,
    db_name: str = "./database.sqlite",
) -> str:
    ensure_memory_tables(db_name)
    prompt_text = str(prompt or "").strip()
    answer_text = str(answer or "").strip()
    if not prompt_text or not answer_text:
        return ""
    now = _now_str()
    episode_uid = uuid4().hex
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO memory_episodes (
            episode_uid, uuid, project_uid, session_uid, prompt, answer, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            episode_uid,
            uuid,
            project_uid,
            session_uid,
            prompt_text,
            answer_text,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return episode_uid


def get_project_memory_episode(
    *,
    episode_uid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    ensure_memory_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT episode_uid, uuid, project_uid, session_uid, prompt, answer, created_at
        FROM memory_episodes
        WHERE episode_uid = ?
        LIMIT 1
    """,
        (str(episode_uid or "").strip(),),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "episode_uid": str(row[0] or ""),
        "uuid": str(row[1] or ""),
        "project_uid": str(row[2] or ""),
        "session_uid": str(row[3] or ""),
        "prompt": str(row[4] or ""),
        "answer": str(row[5] or ""),
        "created_at": str(row[6] or ""),
    }


def list_project_memory_episodes(
    *,
    uuid: str,
    project_uid: str,
    session_uid: str | None = None,
    limit: int = 5,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_memory_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    params: list[Any] = [uuid, project_uid]
    clauses = ["uuid = ?", "project_uid = ?"]
    if isinstance(session_uid, str) and session_uid.strip():
        clauses.append("session_uid = ?")
        params.append(session_uid.strip())
    params.append(max(1, int(limit)))
    cursor.execute(
        f"""
        SELECT episode_uid, uuid, project_uid, session_uid, prompt, answer, created_at
        FROM memory_episodes
        WHERE {" AND ".join(clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """,
        tuple(params),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "episode_uid": str(row[0] or ""),
            "uuid": str(row[1] or ""),
            "project_uid": str(row[2] or ""),
            "session_uid": str(row[3] or ""),
            "prompt": str(row[4] or ""),
            "answer": str(row[5] or ""),
            "created_at": str(row[6] or ""),
        }
        for row in rows
    ]


def _replace_memory_item_evidence(
    *,
    cursor: sqlite3.Cursor,
    memory_uid: str,
    source_episode_uid: str,
    evidence: list[dict[str, Any]],
    now: str,
) -> None:
    cursor.execute("DELETE FROM memory_item_evidence WHERE memory_uid = ?", (memory_uid,))
    if not evidence:
        return
    normalized_episode_uid = str(source_episode_uid or "").strip()
    for item in evidence:
        cursor.execute(
            """
            INSERT INTO memory_item_evidence (memory_uid, episode_uid, evidence_json, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (
                memory_uid,
                str(item.get("episode_uid") or normalized_episode_uid),
                _json_dumps(item, fallback={}),
                now,
            ),
        )


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
    canonical_text: str = "",
    dedup_key: str = "",
    status: str = "active",
    confidence: float = 0.0,
    source_episode_uid: str = "",
    evidence: list[dict[str, Any]] | None = None,
    db_name: str = "./database.sqlite",
) -> str:
    ensure_memory_tables(db_name)
    normalized_content = str(content or "").strip()
    if not normalized_content:
        return ""
    normalized_type = str(memory_type or "episodic").strip().lower() or "episodic"
    normalized_canonical = str(canonical_text or "").strip() or normalized_content
    normalized_dedup_key = str(dedup_key or "").strip()
    normalized_status = str(status or "active").strip().lower() or "active"
    normalized_confidence = float(confidence or 0.0)
    evidence_payload = evidence if isinstance(evidence, list) else []
    now = _now_str()
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if normalized_dedup_key:
        cursor.execute(
            """
            SELECT memory_uid
            FROM memory_items
            WHERE uuid = ? AND project_uid = ? AND dedup_key = ?
            LIMIT 1
        """,
            (uuid, project_uid, normalized_dedup_key),
        )
    else:
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
            SET session_uid = ?, title = ?, content = ?, source_prompt = ?, source_answer = ?,
                updated_at = ?, expires_at = ?, canonical_text = ?, dedup_key = ?, status = ?,
                confidence = ?, source_episode_uid = ?, evidence_json = ?
            WHERE memory_uid = ?
        """,
            (
                str(session_uid or ""),
                str(title or ""),
                normalized_content,
                str(source_prompt or ""),
                str(source_answer or ""),
                now,
                str(expires_at or ""),
                normalized_canonical,
                normalized_dedup_key,
                normalized_status,
                normalized_confidence,
                str(source_episode_uid or ""),
                _json_dumps(evidence_payload, fallback=[]),
                memory_uid,
            ),
        )
        _replace_memory_item_evidence(
            cursor=cursor,
            memory_uid=memory_uid,
            source_episode_uid=str(source_episode_uid or ""),
            evidence=evidence_payload,
            now=now,
        )
        conn.commit()
        conn.close()
        return memory_uid

    memory_uid = uuid4().hex
    cursor.execute(
        """
        INSERT INTO memory_items (
            memory_uid, uuid, project_uid, session_uid, memory_type, title, content,
            source_prompt, source_answer, access_count, created_at, updated_at, last_accessed_at,
            expires_at, canonical_text, dedup_key, status, confidence, source_episode_uid,
            evidence_json, superseded_by
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, '')
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
            normalized_canonical,
            normalized_dedup_key,
            normalized_status,
            normalized_confidence,
            str(source_episode_uid or ""),
            _json_dumps(evidence_payload, fallback=[]),
        ),
    )
    _replace_memory_item_evidence(
        cursor=cursor,
        memory_uid=memory_uid,
        source_episode_uid=str(source_episode_uid or ""),
        evidence=evidence_payload,
        now=now,
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
    status: str | None = None,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_memory_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    resolved_limit = max(1, int(limit))
    now = _now_str()
    clauses = ["uuid = ?", "project_uid = ?"]
    params: list[Any] = [uuid, project_uid]
    if not include_expired:
        clauses.append("(expires_at = '' OR expires_at > ?)")
        params.append(now)
    if isinstance(memory_type, str) and memory_type.strip():
        clauses.append("memory_type = ?")
        params.append(memory_type.strip().lower())
    if isinstance(status, str) and status.strip():
        clauses.append("status = ?")
        params.append(status.strip().lower())
    params.append(resolved_limit)
    cursor.execute(
        f"""
        SELECT memory_uid, session_uid, memory_type, title, content, source_prompt,
               source_answer, access_count, created_at, updated_at, last_accessed_at, expires_at,
               canonical_text, dedup_key, status, confidence, source_episode_uid, evidence_json,
               superseded_by
        FROM memory_items
        WHERE {" AND ".join(clauses)}
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
    """,
        tuple(params),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_memory_item(row) for row in rows]


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
