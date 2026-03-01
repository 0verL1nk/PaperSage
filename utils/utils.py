import datetime
import hashlib
import json
import logging
import os
import random
import sqlite3
import string
import uuid
from typing import Any, Iterator, Tuple

import streamlit as st
import textract
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.settings import load_agent_settings
from agent.llm_provider import build_openai_compatible_chat_model
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
            "SELECT * FROM files WHERE uuid = ? OR user_uuid = ?",
            (uuid_value, uuid_value),
        )
    elif "uuid" in column_names:
        cursor.execute("SELECT * FROM files WHERE uuid = ?", (uuid_value,))
    elif "user_uuid" in column_names:
        cursor.execute("SELECT * FROM files WHERE user_uuid = ?", (uuid_value,))
    else:
        cursor.execute("SELECT * FROM files")

    rows = cursor.fetchall()
    selected_columns = (
        [item[0] for item in cursor.description] if cursor.description else []
    )
    conn.close()

    normalized_rows: list[dict[str, Any]] = []
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
        normalized_rows.append(record.model_dump())
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
        f"""
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
        cursor.execute("SELECT uid FROM files WHERE md5=?", (md5_value,))
    elif "id" in column_names:
        cursor.execute("SELECT id FROM files WHERE md5=?", (md5_value,))
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
        cursor.execute("ALTER TABLE users ADD COLUMN model_name TEXT DEFAULT NULL")
        conn.commit()
    conn.close()


def ensure_users_base_url_column(db_name="./database.sqlite"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    has_base_url = any(row[1] == "base_url" for row in columns)
    if not has_base_url:
        cursor.execute("ALTER TABLE users ADD COLUMN base_url TEXT DEFAULT NULL")
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

    insert_columns: list[str] = []
    for name in column_names:
        if name == "id" and "uid" in column_names:
            continue
        if name in values_by_column:
            insert_columns.append(name)

    insert_values = [values_by_column[name] for name in insert_columns]
    placeholders = ", ".join(["?"] * len(insert_columns))
    columns_sql = ", ".join(insert_columns)

    conn = sqlite3.connect("./database.sqlite")
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT INTO files ({columns_sql}) VALUES ({placeholders})",
        tuple(insert_values),
    )
    conn.commit()
    conn.close()


# Return a dict including result and text,judge the result,1:success,-1:failed.
def extract_files(file_path: str):
    file_type = file_path.split(".")[-1]
    if file_type in ["doc", "docx", "pdf", "txt"]:
        try:
            text = textract.process(file_path).decode("utf-8")
            # 替换'{'和'}'防止解析为变量
            safe_text = text.replace("{", "{{").replace("}", "}}")
            return {"result": 1, "text": safe_text}
        except Exception as e:
            print(e)
            return {"result": -1, "text": e}
    else:
        return {"result": -1, "text": "Unexpect file type!"}


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(base_dir, "logs")
        self.log_level = log_level
        os.makedirs(self.log_dir, exist_ok=True)

        # 动态生成日志文件名（按日期）
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"{current_date}.log")

        # 配置日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        # 检查是否已添加处理器，避免重复
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            # 文件处理器
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # 控制台处理器（可选）
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

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
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
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
    """
    检测文本语言类型
    返回 'zh' 表示中文，'en' 表示英文，'other' 表示其他语言
    """
    # 统计中文字符数量
    chinese_chars = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
    # 统计英文字符数量
    english_chars = len([c for c in text if c.isascii() and c.isalpha()])

    # 计算中英文字符占比
    total_chars = len(text.strip())
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    english_ratio = english_chars / total_chars if total_chars > 0 else 0

    # 判断语言类型
    if chinese_ratio > 0.3:  # 如果中文字符占比超过30%，认为是中文文本
        return "zh"
    elif english_ratio > 0.5:  # 如果英文字符占比超过50%，认为是英文文本
        return "en"
    else:
        return "other"


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
