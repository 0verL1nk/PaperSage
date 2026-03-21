import datetime
import hashlib
import json
import logging
import os
import random
import sqlite3
import string
import uuid
from pathlib import Path
from typing import Any, Iterator, Tuple

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.application.language import detect_language as _detect_language
from agent.application.runtime_tuning import apply_runtime_tuning_env
from agent.llm_provider import build_openai_compatible_chat_model
from agent.logging_utils import configure_application_logging
from agent.memory.store import (
    get_project_session_compact_memory as _memory_store_get_project_session_compact_memory,
)
from agent.memory.store import (
    list_project_memory_items as _memory_store_list_project_memory_items,
)
from agent.memory.store import (
    save_project_session_compact_memory as _memory_store_save_project_session_compact_memory,
)
from agent.memory.store import (
    search_project_memory_items as _memory_store_search_project_memory_items,
)
from agent.memory.store import (
    touch_memory_items as _memory_store_touch_memory_items,
)
from agent.memory.store import (
    upsert_project_memory_item as _memory_store_upsert_project_memory_item,
)
from agent.settings import load_agent_settings

from .schemas import FileRecord


def get_user_api_key(uuid: str | None = None) -> str:
    """
    获取指定用户的 API key（从数据库获取，确保隔离）
    如果没有提供 uuid，尝试从 session_state 获取 uuid
    如果没有用户 API key，返回空字符串
    """
    # 如果没有提供 uuid，尝试从 session_state 获取
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return ""
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return ""
        uuid = candidate_uuid

    # 始终从数据库获取，确保每个用户只看到自己的 API key
    api_key = get_api_key(uuid)
    return api_key if api_key else ""


def get_openai_client():
    """
    获取当前用户的 OpenAI client（每次调用时创建，确保使用正确的 API key）
    """
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")
    settings = load_agent_settings()
    configured_base_url = get_user_base_url()
    resolved_base_url = (
        configured_base_url
        if configured_base_url
        else settings.openai_compatible_base_url
    )
    return OpenAI(api_key=api_key, base_url=resolved_base_url)


def init_database(db_name: str):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # 创建表格存储文件信息（如果不存在）
    # 保存的文件名以随机uid重新命名
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_filename TEXT NOT NULL,
        uid TEXT NOT NULL,
        md5 TEXT NOT NULL,
        file_path TEXT NOT NULL,
        uuid TEXT NOT NULL,
        created_at TEXT
    )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contents (
            uid TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            file_extraction TEXT,
            file_mindmap TEXT,
            file_summary TEXT
        )
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                uuid TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                api_key TEXT DEFAULT NULL,
                model_name TEXT DEFAULT NULL
            )
            """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """)
    # 创建索引提高查询性能
    cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tokens_expires_at ON tokens(expires_at)
            """)
    conn.commit()
    conn.close()

    ensure_files_table_columns(db_name)
    ensure_users_model_name_column(db_name)
    ensure_users_base_url_column(db_name)
    ensure_users_policy_router_model_name_column(db_name)
    ensure_users_policy_router_base_url_column(db_name)
    ensure_users_policy_router_api_key_column(db_name)
    ensure_users_runtime_tuning_columns(db_name)
    ensure_projects_tables(db_name)

    # 初始化任务状态表
    from .task_queue import init_task_table

    init_task_table(db_name)


def _files_table_columns(db_name: str) -> set[str]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(files)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()
    return columns


def get_user_files(
    uuid_value: str, db_name="./database.sqlite"
) -> list[dict[str, Any]]:
    ensure_files_table_columns(db_name)
    column_names = _files_table_columns(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if "uuid" in column_names and "user_uuid" in column_names:
        cursor.execute(
            "SELECT * FROM files WHERE uuid = ? OR user_uuid = ? ORDER BY rowid DESC",
            (uuid_value, uuid_value),
        )
    elif "uuid" in column_names:
        cursor.execute(
            "SELECT * FROM files WHERE uuid = ? ORDER BY rowid DESC",
            (uuid_value,),
        )
    elif "user_uuid" in column_names:
        cursor.execute(
            "SELECT * FROM files WHERE user_uuid = ? ORDER BY rowid DESC",
            (uuid_value,),
        )
    else:
        cursor.execute("SELECT * FROM files ORDER BY rowid DESC")

    rows = cursor.fetchall()
    selected_columns = (
        [item[0] for item in cursor.description] if cursor.description else []
    )
    conn.close()

    normalized_rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for row in rows:
        row_map = {
            selected_columns[idx]: row[idx] for idx in range(len(selected_columns))
        }
        file_name_value = (
            row_map.get("original_filename") or row_map.get("filename") or ""
        )
        file_uid_value = row_map.get("uid") or row_map.get("id") or ""
        created_at_value = row_map.get("created_at") or row_map.get("updated_at") or ""
        file_path_value = row_map.get("file_path") or ""
        record = FileRecord.model_validate(
            {
                "file_path": file_path_value,
                "file_name": file_name_value,
                "uid": file_uid_value,
                "created_at": created_at_value,
            }
        )
        normalized_record = record.model_dump()
        normalized_uid = str(normalized_record.get("uid") or "").strip()
        if normalized_uid and normalized_uid in seen_uids:
            continue
        if normalized_uid:
            seen_uids.add(normalized_uid)
        normalized_rows.append(normalized_record)
    return normalized_rows


def ensure_files_table_columns(db_name="./database.sqlite"):
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


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _project_repo_module():
    # Lazy import avoids circular import during module initialization.
    from agent.adapters.sqlite import project_repository

    return project_repository


def ensure_projects_tables(db_name: str = "./database.sqlite") -> None:
    repo = _project_repo_module()
    repo.ensure_projects_tables(db_name=db_name)


def create_project_session(
    project_uid: str,
    uuid: str,
    session_name: str = "",
    is_pinned: int = 0,
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    repo = _project_repo_module()
    return repo.create_project_session(
        project_uid=project_uid,
        uuid=uuid,
        session_name=session_name,
        is_pinned=is_pinned,
        db_name=db_name,
    )


def update_project_session(
    session_uid: str,
    project_uid: str,
    uuid: str,
    session_name: str | None = None,
    is_pinned: int | None = None,
    db_name: str = "./database.sqlite",
) -> bool:
    repo = _project_repo_module()
    return repo.update_project_session(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        session_name=session_name,
        is_pinned=is_pinned,
        db_name=db_name,
    )


def list_project_sessions(
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    repo = _project_repo_module()
    return repo.list_project_sessions(project_uid=project_uid, uuid=uuid, db_name=db_name)


def ensure_default_project_session(
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> str:
    repo = _project_repo_module()
    return repo.ensure_default_project_session(
        project_uid=project_uid,
        uuid=uuid,
        db_name=db_name,
    )


def delete_project_session(
    session_uid: str,
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> bool:
    repo = _project_repo_module()
    return repo.delete_project_session(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        db_name=db_name,
    )


def list_project_session_messages(
    session_uid: str,
    project_uid: str,
    uuid: str,
    limit: int | None = None,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    repo = _project_repo_module()
    return repo.list_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        limit=limit,
        db_name=db_name,
    )


def count_project_session_messages(
    session_uid: str,
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> int:
    repo = _project_repo_module()
    return repo.count_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        db_name=db_name,
    )


def list_project_session_messages_page(
    session_uid: str,
    project_uid: str,
    uuid: str,
    *,
    offset: int = 0,
    limit: int = 50,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    repo = _project_repo_module()
    return repo.list_project_session_messages_page(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        offset=offset,
        limit=limit,
        db_name=db_name,
    )


def save_project_session_messages(
    session_uid: str,
    project_uid: str,
    uuid: str,
    messages: list[dict[str, Any]],
    db_name: str = "./database.sqlite",
) -> None:
    repo = _project_repo_module()
    repo.save_project_session_messages(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        messages=messages,
        db_name=db_name,
    )


def get_project_session_compact_memory(
    session_uid: str,
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    return _memory_store_get_project_session_compact_memory(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        db_name=db_name,
    )


def save_project_session_compact_memory(
    session_uid: str,
    project_uid: str,
    uuid: str,
    compact_summary: str,
    anchors: list[dict[str, Any]] | None = None,
    db_name: str = "./database.sqlite",
) -> None:
    _memory_store_save_project_session_compact_memory(
        session_uid=session_uid,
        project_uid=project_uid,
        uuid=uuid,
        compact_summary=compact_summary,
        anchors=anchors,
        db_name=db_name,
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
    db_name: str = "./database.sqlite",
) -> str:
    return _memory_store_upsert_project_memory_item(
        uuid=uuid,
        project_uid=project_uid,
        session_uid=session_uid,
        memory_type=memory_type,
        content=content,
        title=title,
        source_prompt=source_prompt,
        source_answer=source_answer,
        expires_at=expires_at,
        db_name=db_name,
    )


def list_project_memory_items(
    *,
    uuid: str,
    project_uid: str,
    memory_type: str | None = None,
    limit: int = 100,
    include_expired: bool = False,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    return _memory_store_list_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        memory_type=memory_type,
        limit=limit,
        include_expired=include_expired,
        db_name=db_name,
    )


def touch_memory_items(
    *,
    memory_uids: list[str],
    db_name: str = "./database.sqlite",
) -> None:
    _memory_store_touch_memory_items(memory_uids=memory_uids, db_name=db_name)


def search_project_memory_items(
    *,
    uuid: str,
    project_uid: str,
    query: str,
    limit: int = 5,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    return _memory_store_search_project_memory_items(
        uuid=uuid,
        project_uid=project_uid,
        query=query,
        limit=limit,
        db_name=db_name,
    )


def create_project(
    uuid: str,
    project_name: str,
    description: str = "",
    db_name: str = "./database.sqlite",
) -> dict[str, Any]:
    repo = _project_repo_module()
    return repo.create_project(
        uuid=uuid,
        project_name=project_name,
        description=description,
        db_name=db_name,
    )


def list_projects(
    uuid: str,
    include_archived: bool = False,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    repo = _project_repo_module()
    return repo.list_projects(
        uuid=uuid,
        include_archived=include_archived,
        db_name=db_name,
    )


def get_project_by_uid(
    project_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, Any] | None:
    repo = _project_repo_module()
    return repo.get_project_by_uid(
        project_uid=project_uid,
        uuid=uuid,
        db_name=db_name,
    )


def update_project(
    project_uid: str,
    uuid: str,
    project_name: str | None = None,
    description: str | None = None,
    archived: int | None = None,
    db_name: str = "./database.sqlite",
) -> bool:
    repo = _project_repo_module()
    return repo.update_project(
        project_uid=project_uid,
        uuid=uuid,
        project_name=project_name,
        description=description,
        archived=archived,
        db_name=db_name,
    )


def add_file_to_project(
    project_uid: str,
    file_uid: str,
    uuid: str,
    is_active: int = 1,
    db_name: str = "./database.sqlite",
) -> bool:
    repo = _project_repo_module()
    return repo.add_file_to_project(
        project_uid=project_uid,
        file_uid=file_uid,
        uuid=uuid,
        is_active=is_active,
        db_name=db_name,
    )


def remove_file_from_project(
    project_uid: str,
    file_uid: str,
    uuid: str,
    db_name: str = "./database.sqlite",
) -> bool:
    repo = _project_repo_module()
    return repo.remove_file_from_project(
        project_uid=project_uid,
        file_uid=file_uid,
        uuid=uuid,
        db_name=db_name,
    )


def set_project_file_active(
    project_uid: str,
    file_uid: str,
    uuid: str,
    is_active: int,
    db_name: str = "./database.sqlite",
) -> bool:
    repo = _project_repo_module()
    return repo.set_project_file_active(
        project_uid=project_uid,
        file_uid=file_uid,
        uuid=uuid,
        is_active=is_active,
        db_name=db_name,
    )


def list_project_files(
    project_uid: str,
    uuid: str,
    active_only: bool = True,
    db_name: str = "./database.sqlite",
) -> list[dict[str, Any]]:
    repo = _project_repo_module()
    return repo.list_project_files(
        project_uid=project_uid,
        uuid=uuid,
        active_only=active_only,
        db_name=db_name,
    )


def get_file_project_counts(
    uuid: str,
    db_name: str = "./database.sqlite",
) -> dict[str, int]:
    repo = _project_repo_module()
    return repo.get_file_project_counts(uuid=uuid, db_name=db_name)


def ensure_default_project_for_user(
    uuid: str,
    db_name: str = "./database.sqlite",
    sync_existing_files: bool = True,
) -> str:
    repo = _project_repo_module()
    return repo.ensure_default_project_for_user(
        uuid=uuid,
        db_name=db_name,
        sync_existing_files=sync_existing_files,
    )


def gen_random_str(length: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def gen_uuid() -> str:
    return str(uuid.uuid4())


def save_token(user_id: str, db_name="./database.sqlite") -> str:
    """
    保存 token 到数据库，有效期1天
    """
    token = gen_random_str(32)
    current_time = int(datetime.datetime.now().timestamp())
    expires_at = current_time + 60 * 60 * 24  # 1天后过期

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # 如果 token 已存在则更新，否则插入
    cursor.execute(
        """
        INSERT OR REPLACE INTO tokens (token, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
    """,
        (token, user_id, current_time, expires_at),
    )
    conn.commit()
    conn.close()

    # 清理过期 token（异步清理，避免影响性能）
    _cleanup_expired_tokens(db_name)

    return token


# 若成功,返回true,uuid,'',依次为result,token,error
def login(
    username: str, password: str, db_name="./database.sqlite"
) -> Tuple[bool, str, str]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # 校验用户名是否存在
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if (not user) or hashlib.sha256(password.encode("utf-8")).hexdigest() != user[2]:
        return False, "", "账号密码错误"
    return True, save_token(user[0], db_name), ""

    # 若成功,返回true,uuid,'',依次为result,token,error


def register(
    username: str, password: str, db_name="./database.sqlite"
) -> Tuple[bool, str, str]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone():
        conn.close()
        return False, "", "用户名已存在"
    uid = gen_uuid()
    cursor.execute(
        """
           INSERT INTO users (uuid, username, password)
           VALUES (?, ?, ?)
           """,
        (uid, username, hashlib.sha256(password.encode("utf-8")).hexdigest()),
    )
    conn.commit()
    conn.close()
    return True, save_token(uid, db_name), ""


def ensure_local_user(
    user_uuid: str = "local-user",
    username: str = "local",
    db_name: str = "./database.sqlite",
):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    table_info = cursor.fetchall()
    column_names = [row[1] for row in table_info]

    cursor.execute("SELECT 1 FROM users WHERE uuid = ?", (user_uuid,))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        ensure_default_project_for_user(
            user_uuid, db_name=db_name, sync_existing_files=False
        )
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    values: dict[str, str] = {
        "uuid": user_uuid,
        "username": username,
        "password": "local",
        "email": f"{username}@local.dev",
        "preferred_model": "local-model",
        "created_at": now,
        "updated_at": now,
    }

    required_columns: list[str] = []
    insert_values: list[str] = []
    for column in table_info:
        name = column[1]
        not_null = column[3] == 1
        default_value = column[4]
        if name not in column_names:
            continue
        if name in values:
            required_columns.append(name)
            insert_values.append(values[name])
            continue
        if not_null and default_value is None and name != "id":
            required_columns.append(name)
            insert_values.append("")

    placeholders = ", ".join(["?"] * len(required_columns))
    columns_sql = ", ".join(required_columns)
    cursor.execute(
        f"INSERT INTO users ({columns_sql}) VALUES ({placeholders})",
        tuple(insert_values),
    )
    conn.commit()
    conn.close()
    ensure_default_project_for_user(user_uuid, db_name=db_name, sync_existing_files=False)


def is_token_expired(token, db_name="./database.sqlite"):
    """
    检查 Token 是否过期
    """
    current_time = int(datetime.datetime.now().timestamp())

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT expires_at FROM tokens WHERE token = ?
    """,
        (token,),
    )
    result = cursor.fetchone()
    conn.close()

    if not result:
        return True  # Token 不存在，认为已过期

    expires_at = result[0]
    if current_time >= expires_at:
        # Token 已过期，删除它
        _delete_token(token, db_name)
        return True

    return False  # Token 未过期


def print_contents(content):
    for key, value in content.items():
        st.write("### " + key + "\n")
        for i in value:
            st.write("- " + i + "\n")


def save_content_to_database(
    uid: str,
    file_path: str,
    content: str,
    content_type: str,
    db_name="./database.sqlite",
):
    """保存内容到数据库，如果记录已存在则更新对应字段"""
    allowed_content_types = {"file_extraction", "file_mindmap", "file_summary"}
    if content_type not in allowed_content_types:
        raise ValueError(f"Unsupported content_type: {content_type}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(db_name, timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA busy_timeout = 5000")

        cursor.execute("PRAGMA table_info(contents)")
        column_names = {row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT 1 FROM contents WHERE uid = ?", (uid,))
        exists = cursor.fetchone() is not None

        if exists:
            if "updated_at" in column_names:
                cursor.execute(
                    f"""
                    UPDATE contents
                    SET {content_type} = ?, updated_at = ?
                    WHERE uid = ?
                """,
                    (content, current_time, uid),
                )
            else:
                cursor.execute(
                    f"""
                    UPDATE contents
                    SET {content_type} = ?
                    WHERE uid = ?
                """,
                    (content, uid),
                )
            return

        insert_columns = ["uid", "file_path", content_type]
        insert_values: list[Any] = [uid, file_path, content]

        if "created_at" in column_names:
            insert_columns.append("created_at")
            insert_values.append(current_time)
        if "updated_at" in column_names:
            insert_columns.append("updated_at")
            insert_values.append(current_time)

        columns_sql = ", ".join(insert_columns)
        placeholders = ", ".join(["?"] * len(insert_columns))
        cursor.execute(
            f"INSERT INTO contents ({columns_sql}) VALUES ({placeholders})",
            tuple(insert_values),
        )


def get_uid_by_md5(md5_value: str, db_name="./database.sqlite"):
    column_names = _files_table_columns(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if "uid" in column_names:
        cursor.execute(
            "SELECT uid FROM files WHERE md5=? ORDER BY rowid DESC LIMIT 1",
            (md5_value,),
        )
    elif "id" in column_names:
        cursor.execute(
            "SELECT id FROM files WHERE md5=? ORDER BY rowid DESC LIMIT 1",
            (md5_value,),
        )
    else:
        conn.close()
        return None
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None


def get_uuid_by_token(token: str, db_name="./database.sqlite") -> str | None:
    """
    通过 token 获取用户 UUID
    """
    # 先检查 token 是否过期
    if is_token_expired(token, db_name):
        return None

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_id FROM tokens WHERE token = ?
    """,
        (token,),
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    return None


def _delete_token(token: str, db_name="./database.sqlite"):
    """
    删除指定的 token（内部函数）
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tokens WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def _cleanup_expired_tokens(db_name="./database.sqlite"):
    """
    清理过期的 token（内部函数）
    定期清理可以保持数据库整洁
    """
    current_time = int(datetime.datetime.now().timestamp())
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tokens WHERE expires_at < ?", (current_time,))
    conn.commit()
    conn.close()


def ensure_users_model_name_column(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    has_model_name = any(row[1] == "model_name" for row in columns)
    if not has_model_name:
        if _try_add_users_column(
            cursor,
            "ALTER TABLE users ADD COLUMN model_name TEXT DEFAULT NULL",
        ):
            conn.commit()
    conn.close()


def ensure_users_base_url_column(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    has_base_url = any(row[1] == "base_url" for row in columns)
    if not has_base_url:
        if _try_add_users_column(
            cursor,
            "ALTER TABLE users ADD COLUMN base_url TEXT DEFAULT NULL",
        ):
            conn.commit()
    conn.close()


def ensure_users_policy_router_model_name_column(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    has_column = any(row[1] == "policy_router_model_name" for row in columns)
    if not has_column:
        if _try_add_users_column(
            cursor,
            "ALTER TABLE users ADD COLUMN policy_router_model_name TEXT DEFAULT NULL",
        ):
            conn.commit()
    conn.close()


def ensure_users_policy_router_base_url_column(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    has_column = any(row[1] == "policy_router_base_url" for row in columns)
    if not has_column:
        if _try_add_users_column(
            cursor,
            "ALTER TABLE users ADD COLUMN policy_router_base_url TEXT DEFAULT NULL",
        ):
            conn.commit()
    conn.close()


def ensure_users_policy_router_api_key_column(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    has_column = any(row[1] == "policy_router_api_key" for row in columns)
    if not has_column:
        if _try_add_users_column(
            cursor,
            "ALTER TABLE users ADD COLUMN policy_router_api_key TEXT DEFAULT NULL",
        ):
            conn.commit()
    conn.close()


def _try_add_users_column(cursor, sql: str) -> bool:
    try:
        cursor.execute(sql)
        return True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return False
        raise


def ensure_users_runtime_tuning_columns(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in cursor.fetchall()}

    altered = False
    if "agent_policy_async_enabled" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN agent_policy_async_enabled INTEGER DEFAULT NULL",
            )
            or altered
        )
    if "agent_policy_async_refresh_seconds" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN agent_policy_async_refresh_seconds REAL DEFAULT NULL",
            )
            or altered
        )
    if "agent_policy_async_min_confidence" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN agent_policy_async_min_confidence REAL DEFAULT NULL",
            )
            or altered
        )
    if "agent_policy_async_max_staleness_seconds" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN agent_policy_async_max_staleness_seconds REAL DEFAULT NULL",
            )
            or altered
        )
    if "rag_index_batch_size" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN rag_index_batch_size INTEGER DEFAULT NULL",
            )
            or altered
        )
    if "agent_document_text_cache_max_chars" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN agent_document_text_cache_max_chars INTEGER DEFAULT NULL",
            )
            or altered
        )
    if "local_rag_project_max_chars" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN local_rag_project_max_chars INTEGER DEFAULT NULL",
            )
            or altered
        )
    if "local_rag_project_max_chunks" not in columns:
        altered = (
            _try_add_users_column(
                cursor,
                "ALTER TABLE users ADD COLUMN local_rag_project_max_chunks INTEGER DEFAULT NULL",
            )
            or altered
        )

    if altered:
        conn.commit()
    conn.close()


def get_content_by_uid(
    uid: str, content_type: str, table_name="contents", db_name="./database.sqlite"
):
    """
    根据文件名获取文件的内容

    Args:
        uid (str): uid

    Returns:
        str: 文件内容，若未找到则返回 None
        :param uid:
        :param content_type:
        :param table_name:
        :param db_name:
        :param table_name:
        :param content_type:
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"SELECT {content_type} FROM {table_name} WHERE uid = ?", (uid,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None


def check_file_exists(md5: str, db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    """根据 MD5 值检查文件是否存在"""
    cursor.execute("SELECT 1 FROM files WHERE md5 = ?", (md5,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def save_file_to_database(
    original_file_name: str,
    uid: str,
    uuid_value: str,
    md5_value: str,
    full_file_path: str,
    current_time: str,
    auto_bind_default_project: bool = True,
):
    ensure_files_table_columns("./database.sqlite")
    column_names = _files_table_columns("./database.sqlite")

    file_extension = os.path.splitext(original_file_name)[-1].lstrip(".")
    file_size = os.path.getsize(full_file_path) if os.path.exists(full_file_path) else 0

    values_by_column: dict[str, Any] = {
        "id": uid,
        "uid": uid,
        "original_filename": original_file_name,
        "filename": uid,
        "md5": md5_value,
        "file_path": full_file_path,
        "file_ext": file_extension,
        "file_size": file_size,
        "mime_type": "application/octet-stream",
        "user_uuid": uuid_value,
        "uuid": uuid_value,
        "processing_status": "uploaded",
        "created_at": current_time,
        "updated_at": current_time,
        "is_favorite": 0,
    }

    conn = sqlite3.connect("./database.sqlite")
    cursor = conn.cursor()
    owner_filters: list[str] = []
    owner_params: list[Any] = []
    if "uuid" in column_names:
        owner_filters.append("uuid = ?")
        owner_params.append(uuid_value)
    if "user_uuid" in column_names:
        owner_filters.append("user_uuid = ?")
        owner_params.append(uuid_value)
    where_sql = "uid = ?"
    where_params: list[Any] = [uid]
    if owner_filters:
        where_sql = f"{where_sql} AND ({' OR '.join(owner_filters)})"
        where_params.extend(owner_params)

    cursor.execute(
        f"SELECT rowid FROM files WHERE {where_sql} ORDER BY rowid DESC LIMIT 1",
        tuple(where_params),
    )
    existing = cursor.fetchone()
    if existing:
        update_columns = [
            name
            for name in column_names
            if name in values_by_column and name not in {"id", "uid", "created_at"}
        ]
        if update_columns:
            update_sql = ", ".join(f"{name} = ?" for name in update_columns)
            update_values = [values_by_column[name] for name in update_columns]
            cursor.execute(
                f"UPDATE files SET {update_sql} WHERE rowid = ?",
                tuple(update_values + [existing[0]]),
            )
    else:
        insert_columns: list[str] = []
        for name in column_names:
            if name == "id" and "uid" in column_names:
                continue
            if name in values_by_column:
                insert_columns.append(name)
        insert_values = [values_by_column[name] for name in insert_columns]
        placeholders = ", ".join(["?"] * len(insert_columns))
        columns_sql = ", ".join(insert_columns)
        cursor.execute(
            f"INSERT INTO files ({columns_sql}) VALUES ({placeholders})",
            tuple(insert_values),
        )
    conn.commit()
    conn.close()
    if auto_bind_default_project:
        try:
            default_project_uid = ensure_default_project_for_user(
                uuid_value,
                db_name="./database.sqlite",
                sync_existing_files=False,
            )
            add_file_to_project(
                project_uid=default_project_uid,
                file_uid=uid,
                uuid=uuid_value,
                is_active=1,
                db_name="./database.sqlite",
            )
        except Exception:
            pass


# Return a dict including result and text,judge the result,1:success,-1:failed.
def _extract_text_with_pymupdf(file_path: str) -> str:
    """使用 pymupdf 提取文本，支持 PDF / DOCX / DOC / TXT 等格式。"""
    import fitz  # pymupdf

    doc = fitz.open(file_path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)


def _extract_text_with_markitdown(file_path: str) -> str:
    """使用 MarkItDown 提取 Markdown 文本。"""
    from markitdown import MarkItDown

    converter = MarkItDown()
    result = converter.convert(file_path)
    text = getattr(result, "text_content", None)
    if not isinstance(text, str):
        raise ValueError("MarkItDown did not return text_content")
    return text


def _extract_text_with_legacy(file_path: str, file_type: str) -> tuple[str, str, str]:
    """旧解析链路：txt 直读，其它用 PyMuPDF。"""
    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "txt", "plain"
    return _extract_text_with_pymupdf(file_path), "pymupdf", "plain"


def _extract_text_with_mineru_api(file_path: str) -> str:
    """通过 MinerU API 解析 PDF，并返回 Markdown 文本。"""
    import httpx

    endpoint = os.getenv("MINERU_API_URL", "http://mineru-api:8000/file_parse").strip()
    parse_backend = os.getenv("MINERU_PARSE_BACKEND", "pipeline").strip()
    parse_method = os.getenv("MINERU_PARSE_METHOD", "auto").strip()
    timeout = float(os.getenv("MINERU_TIMEOUT_SECONDS", "180"))
    lang_list_raw = os.getenv("MINERU_LANG_LIST", "").strip()
    lang_list = [item.strip() for item in lang_list_raw.split(",") if item.strip()]

    data: list[tuple[str, str]] = [
        ("parse_method", parse_method),
        ("backend", parse_backend),
        ("return_md", "true"),
        ("return_middle_json", "false"),
        ("return_model_output", "false"),
        ("return_content_list", "false"),
        ("return_images", "false"),
    ]
    for lang in lang_list:
        data.append(("lang_list", lang))

    with open(file_path, "rb") as f:
        files = {"files": (Path(file_path).name, f, "application/pdf")}
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, data=data, files=files)
            response.raise_for_status()
            payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("MinerU API 返回格式异常")

    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("MinerU API 返回中缺少 results")

    stem = Path(file_path).stem
    candidates: list[Any] = []
    if stem in results:
        candidates.append(results[stem])
    candidates.extend(results.values())

    for item in candidates:
        if not isinstance(item, dict):
            continue
        md_content = item.get("md_content")
        if isinstance(md_content, str):
            return md_content

    raise ValueError("MinerU API 返回中缺少 md_content")


def extract_files(file_path: str):
    file_type = file_path.split(".")[-1].lower()
    if file_type in ["doc", "docx", "pdf", "txt"]:
        try:
            backend = os.getenv("DOC_PARSE_BACKEND", "auto").strip().lower()
            text = ""
            parser = ""
            format_name = ""

            if backend == "mineru":
                if file_type == "pdf":
                    try:
                        text = _extract_text_with_mineru_api(file_path)
                        parser = "mineru-api"
                        format_name = "markdown"
                    except Exception:
                        try:
                            text, parser, format_name = _extract_text_with_legacy(
                                file_path=file_path,
                                file_type=file_type,
                            )
                        except TypeError:
                            text, parser, format_name = _extract_text_with_legacy(
                                file_path,
                                file_type,
                            )
                else:
                    text, parser, format_name = _extract_text_with_legacy(
                        file_path=file_path,
                        file_type=file_type,
                    )
            else:
                markitdown_enabled = backend in {"auto", "markitdown"}
                if markitdown_enabled:
                    try:
                        text = _extract_text_with_markitdown(file_path)
                        parser = "markitdown"
                        format_name = "markdown"
                    except Exception:
                        if backend == "markitdown":
                            raise

                if not text:
                    try:
                        text, parser, format_name = _extract_text_with_legacy(
                            file_path=file_path,
                            file_type=file_type,
                        )
                    except TypeError:
                        text, parser, format_name = _extract_text_with_legacy(
                            file_path,
                            file_type,
                        )

            # 替换'{'和'}'防止解析为变量
            safe_text = text.replace("{", "{{").replace("}", "}}")
            return {
                "result": 1,
                "text": safe_text,
                "format": format_name,
                "parser": parser,
            }
        except Exception as e:
            print(e)
            return {"result": -1, "text": str(e)}
    else:
        return {"result": -1, "text": "Unexpect file type!"}


def optimize_text(text: str):
    system_prompt = """你是一个专业的论文优化助手。你的任务是:
        1. 优化用户输入的文本，使其表达更加流畅、逻辑更加清晰
        2. 替换同义词和调整句式，以降低查重率
        3. 保证原文的核心意思不变
        4. 保证论文专业性,包括用词的专业性以及句式的专业性
        5. 使文本更加符合其语言的语法规范,更像母语者写出来的文章
        请按以下格式输出：
        #### 优化后的文本
        ...
        """
    # 使用当前用户的 API key 和模型名称
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")
    user_model = get_user_model_name()
    if not user_model:
        raise ValueError("请先在侧边栏设置中配置模型名称")
    user_base_url = get_user_base_url()
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=user_model,
        base_url=user_base_url,
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "用户输入:" + text)]
    )
    chain = prompt_template | llm
    return chain.stream({"text": text})


def generate_mindmap_data(text: str) -> dict[str, Any]:
    """生成思维导图数据"""
    system_prompt = """你是一个专业的文献分析专家。请分析给定的文献内容，生成一个结构清晰的思维导图。

    分析要求：
    1. 主题提取
       - 准确识别文档的核心主题作为根节点
       - 确保主题概括准确且简洁

    2. 结构设计
       - 第一层：识别文档的主要章节或核心概念（3-5个）
       - 第二层：提取每个主要章节下的关键要点（2-4个）
       - 第三层：补充具体的细节和示例（如果必要）
       - 最多不超过4层结构

    3. 内容处理
       - 使用简洁的关键词或短语
       - 每个节点内容控制在15字以内
       - 保持逻辑连贯性和层次关系
       - 确保专业术语的准确性

    4. 特殊注意
       - 研究类文献：突出研究背景、方法、结果、结论等关键环节
       - 综述类文献：强调研究现状、问题、趋势等主要方面
       - 技术类文献：注重技术原理、应用场景、优缺点等要素

    输出格式要求：
    必须是严格的JSON格式，不要有任何额外字符，结构如下：
    {{
        "name": "根节点名称",
        "children": [
            {{
                "name": "一级节点1",
                "children": [
                    {{
                        "name": "二级节点1",
                        "children": [...]
                    }}
                ]
            }}
        ]
    }}
    """

    # 使用当前用户的 API key 和模型名称
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")

    user_model = get_user_model_name()
    if not user_model:
        raise ValueError("请先在侧边栏设置中配置模型名称")
    user_base_url = get_user_base_url()

    try:
        llm = build_openai_compatible_chat_model(
            api_key=api_key,
            model_name=user_model,
            base_url=user_base_url,
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "以下是需要分析的文献内容：\n {text}")]
        )

        chain = prompt_template | llm
        result = chain.invoke({"text": text})
        result_content = result.content
        content_text = (
            result_content
            if isinstance(result_content, str)
            else json.dumps(result_content, ensure_ascii=False)
        )
        print(content_text)
        try:
            # 确保返回的是有效的JSON字符串
            json_str = extract_json_string(content_text)
            mindmap_data = json.loads(json_str)
            return mindmap_data
        except json.JSONDecodeError:
            # 如果解析失败，返回一个基本的结构
            return {
                "name": "解析失败",
                "children": [{"name": "文档解析出错", "children": []}],
            }
    except Exception as e:
        raise ValueError(f"生成思维导图时出错: {str(e)}")


class LoggerManager:
    def __init__(self, log_level=logging.INFO):
        level_name = (
            logging.getLevelName(log_level)
            if isinstance(log_level, int)
            else str(log_level)
        )
        configure_application_logging(default_level=str(level_name))
        self.logger = logging.getLogger("llm_app.file_center")
        self.logger.setLevel(log_level)

    def get_logger(self):
        return self.logger


def text_extraction(file_path: str):
    res = extract_files(file_path)
    if res["result"] == 1:
        extracted_text = res["text"]
        if not isinstance(extracted_text, str):
            return False, "文档提取结果格式错误"
        file_content = "以下为一篇论文的原文:\n" + extracted_text
    else:
        return False, ""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": file_content,  # <-- 这里，我们将抽取后的文件内容（注意是文件内容，不是文件 ID）放在请求中
        },
        {
            "role": "user",
            "content": """
         阅读论文,划出**关键语句**,并按照"研究背景，研究目的，研究方法，研究结果，未来展望"五个标签分类.
         label为中文,text为原文,text可能有多句,并以json格式输出.
         注意!!text内是论文原文!!.
         以下为示例:
         {'label1':['text',...],'label2':['text',...],...}
         """,
        },
    ]

    # 使用当前用户的 API key 创建 client
    try:
        client = get_openai_client()
    except ValueError as e:
        return False, str(e)

    # 获取用户选择的模型名称
    user_model = get_user_model_name()
    if not user_model:
        return False, "请先在侧边栏设置中配置模型名称"

    try:
        completion = client.chat.completions.create(
            model=user_model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        # 这边返回的就是json对象了
        response_content = completion.choices[0].message.content
        if not isinstance(response_content, str):
            return False, "模型没有返回有效的 JSON 字符串"
        return True, json.loads(response_content)
    except Exception as e:
        return False, str(e)


def file_summary(file_path: str) -> Tuple[bool, str]:
    res = extract_files(file_path)
    if res["result"] == 1:
        content = res["text"]
        if not isinstance(content, str):
            return False, "文档提取结果格式错误"
    else:
        return False, ""

    system_prompt = """你是一个文书助手。你的客户会交给你一篇文章，你需要用尽可能简洁的语言，总结这篇文章的内容。不得使用 markdown 记号。"""

    # 使用当前用户的 API key 和模型名称
    api_key = get_user_api_key()
    if not api_key:
        return False, "请先在设置中配置您的 API Key"

    user_model = get_user_model_name()
    if not user_model:
        return False, "请先在侧边栏设置中配置模型名称"

    try:
        llm = build_openai_compatible_chat_model(
            api_key=api_key,
            model_name=user_model,
        )

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", content)]
        )
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({})
        st.markdown("### 总结如下：")
        st.text(summary)
        return True, summary
    except Exception as e:
        return False, str(e)


def delete_content_by_uid(uid: str, content_type: str, db_name="./database.sqlite"):
    """删除指定记录的特定内容类型

    Args:
        uid (str): 记录的唯一标识
        content_type (str): 要删除的内容类型 (如 'file_mindmap', 'file_extraction' 等)
        db_name (str): 数据库文件路径

    Returns:
        bool: 操作是否成功
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # 将指定字段设置为 NULL
        cursor.execute(
            f"""
            UPDATE contents
            SET {content_type} = NULL
            WHERE uid = ?
        """,
            (uid,),
        )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"删除内容时出错: {e}")
        return False


def extract_json_string(text: str) -> str:
    """
    从字符串中提取有效的JSON部分
    Args:
        text: 包含JSON的字符串
    Returns:
        str: 提取出的JSON字符串
    """
    if not isinstance(text, str):
        return str(text)

    decoder = json.JSONDecoder()
    lowered = text.lower()
    for tag_name in ("mindmap", "json"):
        open_tag = f"<{tag_name}>"
        close_tag = f"</{tag_name}>"
        start_idx = 0
        while True:
            tag_start = lowered.find(open_tag, start_idx)
            if tag_start < 0:
                break
            content_start = tag_start + len(open_tag)
            tag_end = lowered.find(close_tag, content_start)
            if tag_end < 0:
                break
            candidate = text[content_start:tag_end].strip()
            for idx, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    _payload, end = decoder.raw_decode(candidate[idx:])
                except json.JSONDecodeError:
                    continue
                return candidate[idx : idx + end]
            start_idx = tag_end + len(close_tag)

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            _payload, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return text[idx : idx + end]
    return text


def llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    return json.dumps(content, ensure_ascii=False)


def detect_language(text: str) -> str:
    """兼容入口，实际实现已迁移到 `agent.application.language`。"""
    return _detect_language(text)


def translate_text(
    text: str,
    temperature: float,
    model_name: str,
    optimization_history: list[dict[str, Any]],
) -> str:
    """智能翻译的具体实现"""
    # 使用当前用户的 API key
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        base_url=get_user_base_url(),
    )

    # 检测源语言
    source_lang = detect_language(text)
    target_lang = "en" if source_lang == "zh" else "zh"

    prompt = f"""请将以下文本从{"中文" if source_lang == "zh" else "英文"}翻译成{"英文" if target_lang == "en" else "中文"}。
优化历史:
{optimization_history}
原文：{text}

要求：
1. 保持专业术语的准确性
2. 确保译文流畅自然
3. 保持原文的语气和风格
4. 适当本地化表达方式
5. 注意上下文连贯性

注意!!警告!!提示!!返回要求:只返回翻译后的文本,不要有多余解释,不要有多余的话.
"""
    response = llm.invoke(prompt)
    return llm_content_to_text(response.content)


def process_multy_optimization(
    text: str,
    opt_type: str,
    temperature: float,
    optimization_steps: list[str],
    keywords: list[str],
    special_reqs: str,
) -> Iterator[tuple[str, str]]:
    """
    根据选择的优化步骤进行处理，并记录优化历史
    """
    current_text = text
    user_model = get_user_model_name()
    if not user_model:
        raise ValueError("请先在侧边栏设置中配置模型名称")
    model_name = user_model

    step_functions = {
        "表达优化": (
            optimize_expression,
            "分析：需要改善文本的基础表达方式，使其更加流畅自然。",
        ),
        "专业优化": (
            professionalize_text,
            "分析：需要优化专业术语，提升文本的学术性。",
        ),
        "降重处理": (
            reduce_similarity,
            "分析：需要通过同义词替换和句式重组降低重复率。",
        ),
        "智能翻译": (translate_text, "分析：需要进行中英互译转换。"),
    }

    optimization_history: list[dict[str, Any]] = []

    for step in optimization_steps:
        try:
            func, thought = step_functions[step]

            # 添加优化参数信息到思考过程
            thought += f"\n优化类型：{opt_type}"
            thought += f"\n调整程度：{temperature}"
            if keywords:
                thought += f"\n保留关键词：{', '.join(keywords)}"
            if special_reqs:
                thought += f"\n特殊要求：{special_reqs}"

            # 记录当前步骤的优化历史
            history = {
                "step": step,
                "before": current_text,
                "parameters": {
                    "optimization_type": opt_type,
                    "temperature": temperature,
                    "keywords": keywords,
                    "special_requirements": special_reqs,
                },
            }

            # 执行优化
            current_text = func(
                current_text, temperature, model_name, optimization_history
            )

            # 更新历史记录
            history["after"] = current_text
            optimization_history.append(history)

            yield thought, current_text

        except Exception as e:
            print(f"Error in step {step}: {str(e)}")
            yield f"优化过程中出现错误: {str(e)}", current_text


def optimize_expression(
    text: str,
    temperature: float,
    model_name: str,
    optimization_history: list[dict[str, Any]],
) -> str:
    """改善表达的具体实现"""
    # 使用当前用户的 API key
    # 注意：这里的 model_name 参数是从 process_multy_optimization 传入的，已经考虑了语言检测
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        base_url=get_user_base_url(),
    )

    prompt = f"""请改善以下文本的表达方式，使其更加流畅自然,重要提示：**必须使用与原文相同的语言进行回复！中文或英文或其他语言**
优化历史:
{optimization_history}
原文：{text}

要求：
1. 必须使用与原文完全相同的语言
2. 调整句式使表达更流畅
3. 优化用词使其更自然
4. 保持原有意思不变
5. 确保逻辑连贯性

注意!!警告!!提示!!返回要求:只返回降重后的文本,不要有多余解释,不要有多余的话.
"""
    response = llm.invoke(prompt)
    return llm_content_to_text(response.content)


def professionalize_text(
    text: str,
    temperature: float,
    model_name: str,
    optimization_history: list[dict[str, Any]],
) -> str:
    """专业化处理的具体实现"""
    # 使用当前用户的 API key
    # 注意：这里的 model_name 参数是从 process_multy_optimization 传入的，已经考虑了语言检测
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        base_url=get_user_base_url(),
    )

    prompt = f"""请对以下文本进行专业化处理，优化适当的专业术语和学术表达,重要提示：**必须使用与原文相同的语言进行回复！中文或英文或其它语言**
优化历史:
{optimization_history}
原文：{text}

要求：
1. 必须使用与原文完全相同的语言
2. 优化合适的专业术语
3. 使用更学术的表达方式
4. 保持准确性和可读性
5. 确保专业性和权威性

注意!!警告!!提示!!返回要求:只返回降重后的文本,不要有多余解释,不要有多余的话.
"""
    response = llm.invoke(prompt)
    return llm_content_to_text(response.content)


def reduce_similarity(
    text: str,
    temperature: float,
    model_name: str,
    optimization_history: list[dict[str, Any]],
) -> str:
    """降重处理的具体实现"""
    # 使用当前用户的 API key
    # 注意：这里的 model_name 参数是从 process_multy_optimization 传入的，已经考虑了语言检测
    api_key = get_user_api_key()
    if not api_key:
        raise ValueError("请先在设置中配置您的 API Key")
    llm = build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        base_url=get_user_base_url(),
    )

    prompt = f"""请对以下原文的内容进行降重处理，通过同义词替换和句式重组等方式降低重复率,重要提示：**必须使用与原文相同的语言进行回复！中文或英文或其它语言**
优化历史:
{optimization_history}
**原文**：{text}
--原文结束--
要求：
1. 必须使用与原文完全相同的语言
2. 使用同义词替换
3. 调整句式结构
4. 保持原意不变
5. 确保文本通顺

注意!!警告!!提示!!返回要求:只返回降重后的文本,不要有多余解释,不要有多余的话.
"""
    response = llm.invoke(prompt)
    return llm_content_to_text(response.content)


def save_api_key(uuid: str, api_key: str, db_name="./database.sqlite"):
    """保存用户的 API key"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 更新用户的 API key
    cursor.execute(
        """
        UPDATE users SET api_key = ? WHERE uuid = ?
    """,
        (api_key, uuid),
    )

    conn.commit()
    conn.close()


def get_api_key(uuid: str, db_name="./database.sqlite") -> str:
    """获取用户的 API key"""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT api_key FROM users WHERE uuid = ?", (uuid,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else ""


def save_model_name(uuid: str, model_name: str, db_name="./database.sqlite"):
    """保存用户选择的模型名称"""
    ensure_users_model_name_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 更新用户的模型名称
    cursor.execute(
        """
        UPDATE users SET model_name = ? WHERE uuid = ?
    """,
        (model_name, uuid),
    )

    conn.commit()
    conn.close()


def save_base_url(uuid: str, base_url: str | None, db_name="./database.sqlite"):
    ensure_users_base_url_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE users SET base_url = ? WHERE uuid = ?
    """,
        (base_url, uuid),
    )
    conn.commit()
    conn.close()


def get_model_name(uuid: str, db_name="./database.sqlite") -> str | None:
    ensure_users_model_name_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT model_name FROM users WHERE uuid = ?", (uuid,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def get_base_url(uuid: str, db_name="./database.sqlite") -> str | None:
    ensure_users_base_url_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT base_url FROM users WHERE uuid = ?", (uuid,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def save_policy_router_model_name(
    uuid: str,
    model_name: str | None,
    db_name="./database.sqlite",
):
    ensure_users_policy_router_model_name_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    normalized = str(model_name or "").strip() or None
    cursor.execute(
        """
        UPDATE users SET policy_router_model_name = ? WHERE uuid = ?
    """,
        (normalized, uuid),
    )
    conn.commit()
    conn.close()


def get_policy_router_model_name(uuid: str, db_name="./database.sqlite") -> str | None:
    ensure_users_policy_router_model_name_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT policy_router_model_name FROM users WHERE uuid = ?",
        (uuid,),
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def save_policy_router_base_url(
    uuid: str,
    base_url: str | None,
    db_name="./database.sqlite",
):
    ensure_users_policy_router_base_url_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    normalized = str(base_url or "").strip() or None
    cursor.execute(
        """
        UPDATE users SET policy_router_base_url = ? WHERE uuid = ?
    """,
        (normalized, uuid),
    )
    conn.commit()
    conn.close()


def get_policy_router_base_url(uuid: str, db_name="./database.sqlite") -> str | None:
    ensure_users_policy_router_base_url_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT policy_router_base_url FROM users WHERE uuid = ?",
        (uuid,),
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def save_policy_router_api_key(
    uuid: str,
    api_key: str | None,
    db_name="./database.sqlite",
):
    ensure_users_policy_router_api_key_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    normalized = str(api_key or "").strip() or None
    cursor.execute(
        """
        UPDATE users SET policy_router_api_key = ? WHERE uuid = ?
    """,
        (normalized, uuid),
    )
    conn.commit()
    conn.close()


def get_policy_router_api_key(uuid: str, db_name="./database.sqlite") -> str | None:
    ensure_users_policy_router_api_key_column(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT policy_router_api_key FROM users WHERE uuid = ?",
        (uuid,),
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else None


def save_runtime_tuning_settings(
    uuid: str,
    *,
    agent_policy_async_enabled: bool | None,
    agent_policy_async_refresh_seconds: float | None,
    agent_policy_async_min_confidence: float | None,
    agent_policy_async_max_staleness_seconds: float | None,
    rag_index_batch_size: int | None,
    agent_document_text_cache_max_chars: int | None,
    local_rag_project_max_chars: int | None,
    local_rag_project_max_chunks: int | None,
    db_name="./database.sqlite",
):
    ensure_users_runtime_tuning_columns(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    normalized_async_enabled = (
        int(bool(agent_policy_async_enabled))
        if agent_policy_async_enabled is not None
        else None
    )
    normalized_refresh_seconds = (
        max(0.5, float(agent_policy_async_refresh_seconds))
        if agent_policy_async_refresh_seconds is not None
        else None
    )
    normalized_min_confidence = (
        min(1.0, max(0.0, float(agent_policy_async_min_confidence)))
        if agent_policy_async_min_confidence is not None
        else None
    )
    normalized_max_staleness = (
        max(1.0, float(agent_policy_async_max_staleness_seconds))
        if agent_policy_async_max_staleness_seconds is not None
        else None
    )
    normalized_batch_size = (
        max(1, int(rag_index_batch_size))
        if rag_index_batch_size is not None
        else None
    )
    normalized_document_cache_max_chars = (
        max(0, int(agent_document_text_cache_max_chars))
        if agent_document_text_cache_max_chars is not None
        else None
    )
    normalized_project_max_chars = (
        max(0, int(local_rag_project_max_chars))
        if local_rag_project_max_chars is not None
        else None
    )
    normalized_project_max_chunks = (
        max(0, int(local_rag_project_max_chunks))
        if local_rag_project_max_chunks is not None
        else None
    )

    cursor.execute(
        """
        UPDATE users
        SET
            agent_policy_async_enabled = ?,
            agent_policy_async_refresh_seconds = ?,
            agent_policy_async_min_confidence = ?,
            agent_policy_async_max_staleness_seconds = ?,
            rag_index_batch_size = ?,
            agent_document_text_cache_max_chars = ?,
            local_rag_project_max_chars = ?,
            local_rag_project_max_chunks = ?
        WHERE uuid = ?
    """,
        (
            normalized_async_enabled,
            normalized_refresh_seconds,
            normalized_min_confidence,
            normalized_max_staleness,
            normalized_batch_size,
            normalized_document_cache_max_chars,
            normalized_project_max_chars,
            normalized_project_max_chunks,
            uuid,
        ),
    )
    conn.commit()
    conn.close()


def get_runtime_tuning_settings(
    uuid: str,
    db_name="./database.sqlite",
) -> dict[str, bool | int | float | None]:
    ensure_users_runtime_tuning_columns(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            agent_policy_async_enabled,
            agent_policy_async_refresh_seconds,
            agent_policy_async_min_confidence,
            agent_policy_async_max_staleness_seconds,
            rag_index_batch_size,
            agent_document_text_cache_max_chars,
            local_rag_project_max_chars,
            local_rag_project_max_chunks
        FROM users
        WHERE uuid = ?
    """,
        (uuid,),
    )
    result = cursor.fetchone()
    conn.close()
    if not result:
        return {
            "agent_policy_async_enabled": None,
            "agent_policy_async_refresh_seconds": None,
            "agent_policy_async_min_confidence": None,
            "agent_policy_async_max_staleness_seconds": None,
            "rag_index_batch_size": None,
            "agent_document_text_cache_max_chars": None,
            "local_rag_project_max_chars": None,
            "local_rag_project_max_chunks": None,
        }
    return {
        "agent_policy_async_enabled": (
            None if result[0] is None else bool(int(result[0]))
        ),
        "agent_policy_async_refresh_seconds": (
            None if result[1] is None else float(result[1])
        ),
        "agent_policy_async_min_confidence": (
            None if result[2] is None else float(result[2])
        ),
        "agent_policy_async_max_staleness_seconds": (
            None if result[3] is None else float(result[3])
        ),
        "rag_index_batch_size": None if result[4] is None else int(result[4]),
        "agent_document_text_cache_max_chars": (
            None if result[5] is None else int(result[5])
        ),
        "local_rag_project_max_chars": None if result[6] is None else int(result[6]),
        "local_rag_project_max_chunks": None if result[7] is None else int(result[7]),
    }


def get_user_runtime_tuning_settings(
    uuid: str | None = None,
) -> dict[str, bool | int | float | None]:
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return {
                "agent_policy_async_enabled": None,
                "agent_policy_async_refresh_seconds": None,
                "agent_policy_async_min_confidence": None,
                "agent_policy_async_max_staleness_seconds": None,
                "rag_index_batch_size": None,
                "agent_document_text_cache_max_chars": None,
                "local_rag_project_max_chars": None,
                "local_rag_project_max_chunks": None,
            }
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return {
                "agent_policy_async_enabled": None,
                "agent_policy_async_refresh_seconds": None,
                "agent_policy_async_min_confidence": None,
                "agent_policy_async_max_staleness_seconds": None,
                "rag_index_batch_size": None,
                "agent_document_text_cache_max_chars": None,
                "local_rag_project_max_chars": None,
                "local_rag_project_max_chunks": None,
            }
        uuid = candidate_uuid
    return get_runtime_tuning_settings(uuid)


def apply_user_runtime_tuning_env(uuid: str | None = None) -> dict[str, str]:
    settings = get_user_runtime_tuning_settings(uuid)
    return apply_runtime_tuning_env(settings=settings, environ=os.environ)


def get_user_model_name(uuid: str | None = None) -> str | None:
    """
    获取指定用户的模型名称（从数据库获取，确保隔离）
    如果没有提供 uuid，尝试从 session_state 获取 uuid
    """
    # 如果没有提供 uuid，尝试从 session_state 获取
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return None
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return None
        uuid = candidate_uuid

    # 始终从数据库获取
    return get_model_name(uuid)


def get_user_base_url(uuid: str | None = None) -> str | None:
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return None
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return None
        uuid = candidate_uuid

    return get_base_url(uuid)


def get_user_policy_router_model_name(uuid: str | None = None) -> str | None:
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return None
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return None
        uuid = candidate_uuid
    return get_policy_router_model_name(uuid)


def get_user_policy_router_base_url(uuid: str | None = None) -> str | None:
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return None
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return None
        uuid = candidate_uuid
    return get_policy_router_base_url(uuid)


def get_user_policy_router_api_key(uuid: str | None = None) -> str | None:
    if not uuid:
        if "uuid" not in st.session_state or not st.session_state["uuid"]:
            return None
        candidate_uuid = st.session_state["uuid"]
        if not isinstance(candidate_uuid, str):
            return None
        uuid = candidate_uuid
    return get_policy_router_api_key(uuid)


def show_sidebar_api_key_setting():
    """
    显示侧边栏 API Key 和模型设置
    应该在每个页面中调用，用于统一显示 API Key 和模型配置界面
    """
    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"

    ensure_local_user(st.session_state["uuid"])

    with st.sidebar:
        st.header("设置")

        # API Key 设置
        # 始终从数据库获取，确保每个用户只看到自己的 API key，避免 session 共享问题
        saved_api_key = get_api_key(st.session_state["uuid"])

        # 使用 key 参数，确保每次渲染都从数据库读取最新值
        current_api_key = st.text_input(
            "API Key:",
            value=saved_api_key,
            type="password",
            help="请输入您的 API key",
            key=f"api_key_input_{st.session_state['uuid']}",  # 使用 uuid 作为 key 的一部分
        )

        st.divider()

        # 模型选择 - 允许自定义输入
        saved_model_name = get_model_name(st.session_state["uuid"])
        if not saved_model_name:
            saved_model_name = ""

        current_model_name = st.text_input(
            "模型名称:",
            value=saved_model_name,
            help="请输入要使用的 AI 模型名称",
            key=f"model_input_{st.session_state['uuid']}",
            placeholder="例如: qwen-plus / gpt-4o-mini / any-openai-compatible-model",
        )

        if not saved_model_name:
            st.warning("⚠️ 尚未配置模型名称，部分功能将不可用")

        st.divider()

        settings = load_agent_settings()
        saved_base_url = get_base_url(st.session_state["uuid"])
        current_base_url = st.text_input(
            "OpenAI Compatible Base URL:",
            value=saved_base_url
            if saved_base_url
            else settings.openai_compatible_base_url,
            help="用于 OpenAI Compatible 接口的 Base URL",
            key=f"base_url_input_{st.session_state['uuid']}",
            placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if st.button("保存设置", use_container_width=True, key="save_settings_btn"):
            normalized_model_name = (
                current_model_name.strip() if current_model_name else ""
            )
            normalized_base_url = current_base_url.strip() if current_base_url else ""

            save_api_key(st.session_state["uuid"], current_api_key)
            save_model_name(
                st.session_state["uuid"],
                normalized_model_name,
            )
            save_base_url(
                st.session_state["uuid"],
                normalized_base_url if normalized_base_url else None,
            )
            persisted_api_key = get_api_key(st.session_state["uuid"])
            persisted_model_name = get_model_name(st.session_state["uuid"])
            persisted_base_url = get_base_url(st.session_state["uuid"])

            model_saved = (persisted_model_name or "") == normalized_model_name
            base_url_saved = (persisted_base_url or "") == normalized_base_url
            api_key_saved = persisted_api_key == current_api_key

            if api_key_saved and model_saved and base_url_saved:
                st.toast("✅ 设置已保存")
                st.rerun()
            else:
                st.error("设置保存失败，请重试")
