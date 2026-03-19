import datetime
import json
import sqlite3
import uuid as uuid_lib
from typing import Any

from utils.schemas import FileRecord


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _gen_uuid() -> str:
    return str(uuid_lib.uuid4())


def _files_table_columns(db_name: str) -> set[str]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(files)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()
    return columns


def ensure_files_table_columns(db_name: str = "./database.sqlite") -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(files)")
    columns = cursor.fetchall()
    column_names = {row[1] for row in columns}

    if "uuid" not in column_names:
        cursor.execute("ALTER TABLE files ADD COLUMN uuid TEXT DEFAULT 'local-user'")
        cursor.execute(
            "UPDATE files SET uuid = 'local-user' WHERE uuid IS NULL OR uuid = ''"
        )

    if "created_at" not in column_names:
        cursor.execute("ALTER TABLE files ADD COLUMN created_at TEXT DEFAULT ''")

    conn.commit()
    conn.close()


def ensure_projects_tables(db_name: str = "./database.sqlite") -> None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_uid TEXT UNIQUE NOT NULL,
            uuid TEXT NOT NULL,
            project_name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archived INTEGER DEFAULT 0
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS project_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_uid TEXT NOT NULL,
            file_uid TEXT NOT NULL,
            uuid TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            added_at TEXT NOT NULL,
            UNIQUE(project_uid, file_uid)
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS project_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uid TEXT UNIQUE NOT NULL,
            project_uid TEXT NOT NULL,
            uuid TEXT NOT NULL,
            session_name TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_pinned INTEGER DEFAULT 0
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS project_session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uid TEXT NOT NULL,
            project_uid TEXT NOT NULL,
            uuid TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            message_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )
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
        CREATE TABLE IF NOT EXISTS memory_item_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_uid TEXT NOT NULL,
            episode_uid TEXT NOT NULL,
            evidence_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """
    )
    cursor.execute("PRAGMA table_info(memory_items)")
    memory_columns = {row[1] for row in cursor.fetchall()}
    if "expires_at" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN expires_at TEXT DEFAULT ''")
    if "canonical_text" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN canonical_text TEXT DEFAULT ''")
    if "dedup_key" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN dedup_key TEXT DEFAULT ''")
    if "status" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN status TEXT DEFAULT 'active'")
    if "confidence" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN confidence REAL DEFAULT 0")
    if "source_episode_uid" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN source_episode_uid TEXT DEFAULT ''")
    if "evidence_json" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN evidence_json TEXT DEFAULT '[]'")
    if "superseded_by" not in memory_columns:
        cursor.execute("ALTER TABLE memory_items ADD COLUMN superseded_by TEXT DEFAULT ''")

    cursor.execute("PRAGMA table_info(project_sessions)")
    session_columns = {row[1] for row in cursor.fetchall()}
    if "thread_id" not in session_columns:
        cursor.execute("ALTER TABLE project_sessions ADD COLUMN thread_id TEXT DEFAULT ''")

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_projects_uuid_updated_at ON projects(uuid, updated_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_files_project_uid ON project_files(project_uid)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_files_uuid ON project_files(uuid)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_sessions_project_uid ON project_sessions(project_uid, updated_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_messages_session ON project_session_messages(session_uid, id ASC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_project_messages_project ON project_session_messages(project_uid, uuid, id ASC)"
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
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_episodes_scope ON memory_episodes(uuid, project_uid, session_uid, created_at DESC)"
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

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_outputs'"
    )
    has_agent_outputs = cursor.fetchone() is not None
    if has_agent_outputs:
        cursor.execute("PRAGMA table_info(agent_outputs)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "project_uid" not in existing_columns:
            cursor.execute("ALTER TABLE agent_outputs ADD COLUMN project_uid TEXT")
        if "session_uid" not in existing_columns:
            cursor.execute("ALTER TABLE agent_outputs ADD COLUMN session_uid TEXT")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_outputs_uuid_project ON agent_outputs(uuid, project_uid, created_at)"
        )

    conn.commit()
    conn.close()


def create_project_session(
    project_uid: str,
    uuid: str,
    session_name: str = "",
    is_pinned: int = 0,
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    ensure_projects_tables(db_name)
    session_uid = _gen_uuid()
    now = _now_str()
    normalized_name = session_name.strip() or "新会话"

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO project_sessions (
            session_uid, project_uid, uuid, session_name, created_at, updated_at, is_pinned
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            session_uid,
            project_uid,
            uuid,
            normalized_name,
            now,
            now,
            1 if int(is_pinned) else 0,
        ),
    )
    cursor.execute(
        """
        UPDATE projects
        SET updated_at = ?
        WHERE project_uid = ? AND uuid = ?
    """,
        (now, project_uid, uuid),
    )
    conn.commit()
    conn.close()

    return {
        "session_uid": session_uid,
        "project_uid": project_uid,
        "uuid": uuid,
        "session_name": normalized_name,
        "created_at": now,
        "updated_at": now,
        "is_pinned": 1 if int(is_pinned) else 0,
        "message_count": 0,
        "last_message": "",
    }


def update_project_session(
    session_uid: str,
    project_uid: str,
    uuid: str,
    session_name: str | None = None,
    is_pinned: int | None = None,
    db_name: str = "./database.sqlite",
) -> bool:
    ensure_projects_tables(db_name)
    updates: list[str] = []
    values: list[Any] = []
    if session_name is not None:
        updates.append("session_name = ?")
        values.append(session_name.strip() or "未命名会话")
    if is_pinned is not None:
        updates.append("is_pinned = ?")
        values.append(1 if int(is_pinned) else 0)
    updates.append("updated_at = ?")
    now = _now_str()
    values.append(now)
    values.extend([session_uid, project_uid, uuid])

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE project_sessions
        SET {", ".join(updates)}
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        tuple(values),
    )
    affected = cursor.rowcount > 0
    if affected:
        cursor.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE project_uid = ? AND uuid = ?
        """,
            (now, project_uid, uuid),
        )
    conn.commit()
    conn.close()
    return affected


def list_project_sessions(
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            s.session_uid,
            s.project_uid,
            s.uuid,
            s.session_name,
            s.created_at,
            s.updated_at,
            s.is_pinned,
            (
                SELECT COUNT(*)
                FROM project_session_messages m
                WHERE m.session_uid = s.session_uid
                  AND m.project_uid = s.project_uid
                  AND m.uuid = s.uuid
            ) AS message_count,
            (
                SELECT m.content
                FROM project_session_messages m
                WHERE m.session_uid = s.session_uid
                  AND m.project_uid = s.project_uid
                  AND m.uuid = s.uuid
                ORDER BY m.id DESC
                LIMIT 1
            ) AS last_message
        FROM project_sessions s
        WHERE s.project_uid = ? AND s.uuid = ?
        ORDER BY s.is_pinned DESC, s.updated_at DESC, s.id DESC
    """,
        (project_uid, uuid),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "session_uid": row[0],
            "project_uid": row[1],
            "uuid": row[2],
            "session_name": row[3] or "",
            "created_at": row[4] or "",
            "updated_at": row[5] or "",
            "is_pinned": int(row[6] or 0),
            "message_count": int(row[7] or 0),
            "last_message": row[8] or "",
        }
        for row in rows
    ]


def ensure_default_project_session(
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> str:
    sessions = list_project_sessions(project_uid=project_uid, uuid=uuid, db_name=db_name)
    if sessions:
        return str(sessions[0]["session_uid"])
    created = create_project_session(
        project_uid=project_uid,
        uuid=uuid,
        session_name="默认会话",
        db_name=db_name,
    )
    return str(created["session_uid"])


def delete_project_session(
    session_uid: str,
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> bool:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM project_session_messages
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (session_uid, project_uid, uuid),
    )
    cursor.execute(
        """
        DELETE FROM session_compact_memory
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (session_uid, project_uid, uuid),
    )
    cursor.execute(
        """
        DELETE FROM project_sessions
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (session_uid, project_uid, uuid),
    )
    affected = cursor.rowcount > 0
    if affected:
        cursor.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE project_uid = ? AND uuid = ?
        """,
            (_now_str(), project_uid, uuid),
        )
    conn.commit()
    conn.close()
    return affected


def list_project_session_messages(
    session_uid: str,
    project_uid: str,
    uuid: str,
    limit: int | None = None,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if isinstance(limit, int) and limit > 0:
        cursor.execute(
            """
            SELECT role, content, message_json
            FROM (
                SELECT id, role, content, message_json
                FROM project_session_messages
                WHERE session_uid = ? AND project_uid = ? AND uuid = ?
                ORDER BY id DESC
                LIMIT ?
            ) latest
            ORDER BY id ASC
        """,
            (session_uid, project_uid, uuid, limit),
        )
    else:
        cursor.execute(
            """
            SELECT role, content, message_json
            FROM project_session_messages
            WHERE session_uid = ? AND project_uid = ? AND uuid = ?
            ORDER BY id ASC
        """,
            (session_uid, project_uid, uuid),
        )
    rows = cursor.fetchall()
    conn.close()

    messages: list[dict[str, Any]] = []
    for role, content, raw_json in rows:
        fallback = {"role": role or "assistant", "content": content or ""}
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                messages.append(parsed)
                continue
        except Exception:
            pass
        messages.append(fallback)
    return messages


def count_project_session_messages(
    session_uid: str,
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> int:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(1)
        FROM project_session_messages
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (session_uid, project_uid, uuid),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return 0
    try:
        return max(0, int(row[0]))
    except Exception:
        return 0


def list_project_session_messages_page(
    session_uid: str,
    project_uid: str,
    uuid: str,
    *,
    offset: int = 0,
    limit: int = 50,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_projects_tables(db_name)
    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(int(limit), 500))
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content, message_json
        FROM project_session_messages
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
        ORDER BY id ASC
        LIMIT ? OFFSET ?
    """,
        (session_uid, project_uid, uuid, safe_limit, safe_offset),
    )
    rows = cursor.fetchall()
    conn.close()

    messages: list[dict[str, Any]] = []
    for role, content, raw_json in rows:
        fallback = {"role": role or "assistant", "content": content or ""}
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                messages.append(parsed)
                continue
        except Exception:
            pass
        messages.append(fallback)
    return messages


def save_project_session_messages(
    session_uid: str,
    project_uid: str,
    uuid: str,
    messages: list[dict[str, Any]],
    db_name: str = "./database.sqlite",
) -> None:
    ensure_projects_tables(db_name)
    now = _now_str()
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM project_session_messages
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (session_uid, project_uid, uuid),
    )
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "assistant")
        content = str(message.get("content") or "")
        serialized = json.dumps(message, ensure_ascii=False)
        cursor.execute(
            """
            INSERT INTO project_session_messages (
                session_uid, project_uid, uuid, role, content, message_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (session_uid, project_uid, uuid, role, content, serialized, now),
        )

    cursor.execute(
        """
        UPDATE project_sessions
        SET updated_at = ?
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (now, session_uid, project_uid, uuid),
    )
    cursor.execute(
        """
        UPDATE projects
        SET updated_at = ?
        WHERE project_uid = ? AND uuid = ?
    """,
        (now, project_uid, uuid),
    )
    conn.commit()
    conn.close()


def create_project(
    uuid: str,
    project_name: str,
    description: str = "",
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    ensure_projects_tables(db_name)
    project_uid = _gen_uuid()
    now = _now_str()
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO projects (project_uid, uuid, project_name, description, created_at, updated_at, archived)
        VALUES (?, ?, ?, ?, ?, ?, 0)
    """,
        (project_uid, uuid, project_name.strip() or "未命名项目", description, now, now),
    )
    conn.commit()
    conn.close()
    return {
        "project_uid": project_uid,
        "uuid": uuid,
        "project_name": project_name.strip() or "未命名项目",
        "description": description,
        "created_at": now,
        "updated_at": now,
        "archived": 0,
    }


def list_projects(
    uuid: str,
    include_archived: bool = False,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if include_archived:
        cursor.execute(
            """
            SELECT project_uid, uuid, project_name, description, created_at, updated_at, archived
            FROM projects
            WHERE uuid = ?
            ORDER BY updated_at DESC, id DESC
        """,
            (uuid,),
        )
    else:
        cursor.execute(
            """
            SELECT project_uid, uuid, project_name, description, created_at, updated_at, archived
            FROM projects
            WHERE uuid = ? AND archived = 0
            ORDER BY updated_at DESC, id DESC
        """,
            (uuid,),
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "project_uid": row[0],
            "uuid": row[1],
            "project_name": row[2],
            "description": row[3] or "",
            "created_at": row[4],
            "updated_at": row[5],
            "archived": int(row[6] or 0),
        }
        for row in rows
    ]


def get_project_by_uid(
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, Any] | None:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT project_uid, uuid, project_name, description, created_at, updated_at, archived
        FROM projects
        WHERE project_uid = ? AND uuid = ?
        LIMIT 1
    """,
        (project_uid, uuid),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "project_uid": row[0],
        "uuid": row[1],
        "project_name": row[2],
        "description": row[3] or "",
        "created_at": row[4],
        "updated_at": row[5],
        "archived": int(row[6] or 0),
    }


def update_project(
    project_uid: str,
    uuid: str,
    project_name: str | None = None,
    description: str | None = None,
    archived: int | None = None,
    db_name: str = "./database.sqlite",
) -> bool:
    ensure_projects_tables(db_name)
    updates: list[str] = []
    values: list[Any] = []
    if project_name is not None:
        updates.append("project_name = ?")
        values.append(project_name.strip() or "未命名项目")
    if description is not None:
        updates.append("description = ?")
        values.append(description)
    if archived is not None:
        updates.append("archived = ?")
        values.append(1 if int(archived) else 0)
    updates.append("updated_at = ?")
    values.append(_now_str())
    values.extend([project_uid, uuid])
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE projects
        SET {", ".join(updates)}
        WHERE project_uid = ? AND uuid = ?
    """,
        tuple(values),
    )
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected


def add_file_to_project(
    project_uid: str,
    file_uid: str,
    uuid: str,
    is_active: int = 1,
    db_name: str = "./database.sqlite",
) -> bool:
    ensure_projects_tables(db_name)
    now = _now_str()
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO project_files (project_uid, file_uid, uuid, is_active, added_at)
        VALUES (?, ?, ?, ?, ?)
    """,
        (project_uid, file_uid, uuid, 1 if int(is_active) else 0, now),
    )
    cursor.execute(
        """
        UPDATE project_files
        SET is_active = ?, added_at = ?
        WHERE project_uid = ? AND file_uid = ? AND uuid = ?
    """,
        (1 if int(is_active) else 0, now, project_uid, file_uid, uuid),
    )
    cursor.execute(
        """
        UPDATE projects
        SET updated_at = ?
        WHERE project_uid = ? AND uuid = ?
    """,
        (now, project_uid, uuid),
    )
    affected = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return affected


def remove_file_from_project(
    project_uid: str,
    file_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> bool:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM project_files
        WHERE project_uid = ? AND file_uid = ? AND uuid = ?
    """,
        (project_uid, file_uid, uuid),
    )
    affected = cursor.rowcount > 0
    if affected:
        cursor.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE project_uid = ? AND uuid = ?
        """,
            (_now_str(), project_uid, uuid),
        )
    conn.commit()
    conn.close()
    return affected


def set_project_file_active(
    project_uid: str,
    file_uid: str,
    uuid: str,
    is_active: int,
    db_name: str = "./database.sqlite",
) -> bool:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    now = _now_str()
    cursor.execute(
        """
        UPDATE project_files
        SET is_active = ?, added_at = ?
        WHERE project_uid = ? AND file_uid = ? AND uuid = ?
    """,
        (1 if int(is_active) else 0, now, project_uid, file_uid, uuid),
    )
    affected = cursor.rowcount > 0
    if affected:
        cursor.execute(
            """
            UPDATE projects
            SET updated_at = ?
            WHERE project_uid = ? AND uuid = ?
        """,
            (now, project_uid, uuid),
        )
    conn.commit()
    conn.close()
    return affected


def list_project_files(
    project_uid: str,
    uuid: str,
    active_only: bool = True,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    ensure_projects_tables(db_name)
    ensure_files_table_columns(db_name)
    file_columns = _files_table_columns(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    active_filter = "AND pf.is_active = 1" if active_only else ""
    if "uuid" in file_columns and "user_uuid" in file_columns:
        join_owner_filter = "AND (f.uuid = pf.uuid OR f.user_uuid = pf.uuid)"
    elif "uuid" in file_columns:
        join_owner_filter = "AND f.uuid = pf.uuid"
    elif "user_uuid" in file_columns:
        join_owner_filter = "AND f.user_uuid = pf.uuid"
    else:
        join_owner_filter = ""
    cursor.execute(
        f"""
        SELECT
            pf.project_uid,
            pf.file_uid,
            pf.is_active,
            pf.added_at,
            f.file_path,
            COALESCE(f.original_filename, '') AS file_name,
            COALESCE(f.created_at, '') AS created_at
        FROM project_files pf
        JOIN files f ON f.uid = pf.file_uid
        {join_owner_filter}
        WHERE pf.project_uid = ? AND pf.uuid = ?
        {active_filter}
        ORDER BY pf.added_at DESC, pf.id DESC, f.rowid DESC
    """,
        (project_uid, uuid),
    )
    rows = cursor.fetchall()
    conn.close()
    normalized: list[dict[str, Any]] = []
    seen_file_uids: set[str] = set()
    for row in rows:
        file_uid = str(row[1] or "").strip()
        if file_uid and file_uid in seen_file_uids:
            continue
        if file_uid:
            seen_file_uids.add(file_uid)
        record = FileRecord.model_validate(
            {
                "file_path": row[4] or "",
                "file_name": row[5] or "",
                "uid": file_uid,
                "created_at": row[6] or "",
            }
        ).model_dump()
        record["project_uid"] = row[0]
        record["is_active"] = int(row[2] or 0)
        record["added_at"] = row[3] or ""
        normalized.append(record)
    return normalized


def get_file_project_counts(
    uuid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, int]:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT file_uid, COUNT(*)
        FROM project_files
        WHERE uuid = ?
        GROUP BY file_uid
    """,
        (uuid,),
    )
    rows = cursor.fetchall()
    conn.close()
    return {str(file_uid): int(count) for file_uid, count in rows}


def _list_user_file_uids_for_sync(uuid: str, db_name: str) -> list[str]:
    ensure_files_table_columns(db_name)
    file_columns = _files_table_columns(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if "uuid" in file_columns and "user_uuid" in file_columns:
        cursor.execute(
            "SELECT uid FROM files WHERE uuid = ? OR user_uuid = ? ORDER BY rowid DESC",
            (uuid, uuid),
        )
    elif "uuid" in file_columns:
        cursor.execute(
            "SELECT uid FROM files WHERE uuid = ? ORDER BY rowid DESC",
            (uuid,),
        )
    elif "user_uuid" in file_columns:
        cursor.execute(
            "SELECT uid FROM files WHERE user_uuid = ? ORDER BY rowid DESC",
            (uuid,),
        )
    else:
        cursor.execute("SELECT uid FROM files ORDER BY rowid DESC")

    rows = cursor.fetchall()
    conn.close()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def ensure_default_project_for_user(
    uuid: str,
    db_name: str = "./database.sqlite",
    sync_existing_files: bool = True,
) -> str:
    ensure_projects_tables(db_name)
    projects = list_projects(uuid=uuid, include_archived=True, db_name=db_name)
    if projects:
        default_project_uid = str(projects[0]["project_uid"])
    else:
        created = create_project(
            uuid=uuid,
            project_name="默认项目",
            description="系统自动创建",
            db_name=db_name,
        )
        default_project_uid = str(created["project_uid"])

    if sync_existing_files:
        for file_uid in _list_user_file_uids_for_sync(uuid, db_name=db_name):
            add_file_to_project(
                project_uid=default_project_uid,
                file_uid=file_uid,
                uuid=uuid,
                is_active=1,
                db_name=db_name,
            )
    return default_project_uid


def get_or_create_thread_id(
    project_uid: str,
    session_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> str:
    ensure_projects_tables(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT thread_id
        FROM project_sessions
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (session_uid, project_uid, uuid),
    )
    row = cursor.fetchone()
    if row and row[0]:
        conn.close()
        return str(row[0])

    thread_id = f"thread-{_gen_uuid()}"
    cursor.execute(
        """
        UPDATE project_sessions
        SET thread_id = ?
        WHERE session_uid = ? AND project_uid = ? AND uuid = ?
    """,
        (thread_id, session_uid, project_uid, uuid),
    )
    conn.commit()
    conn.close()
    return thread_id
