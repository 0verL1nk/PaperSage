import datetime
import hashlib
import os
from pathlib import Path
import uuid

import streamlit as st

from utils.utils import (
    LoggerManager,
    check_file_exists,
    ensure_local_user,
    get_uid_by_md5,
    get_user_files,
    init_database,
    save_file_to_database,
)


def calculate_md5(uploaded_file) -> str:
    md5_hash = hashlib.md5()
    for chunk in iter(lambda: uploaded_file.read(4096), b""):
        md5_hash.update(chunk)
    return md5_hash.hexdigest()


def load_files() -> None:
    files = get_user_files(st.session_state["uuid"])
    st.session_state["files"] = []
    for file in files:
        st.session_state["files"].append(
            {
                "file_path": file["file_path"],
                "file_name": file["file_name"],
                "uid": file["uid"],
                "created_at": file["created_at"],
            }
        )


def upload_file(save_dir: str, logger) -> None:
    uploaded_file = st.file_uploader("请上传文档", type=["txt", "doc", "docx", "pdf"])
    if uploaded_file is None:
        return

    md5_value = calculate_md5(uploaded_file)
    if not check_file_exists(md5_value):
        file_uid = str(uuid.uuid4())
    else:
        file_uid = get_uid_by_md5(md5_value)
        if not file_uid:
            st.error("无法读取已存在文件标识，请重试。")
            return

    original_filename = uploaded_file.name
    file_extension = os.path.splitext(original_filename)[-1]
    file_name = os.path.splitext(original_filename)[0]
    saved_filename = f"{file_uid}{file_extension}"
    file_path = os.path.join(save_dir, saved_filename)

    if not os.path.exists(file_path):
        uploaded_file.seek(0)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_file_to_database(
        original_filename,
        file_uid,
        st.session_state["uuid"],
        md5_value,
        file_path,
        current_time,
    )
    st.toast("文档上传成功", icon="👌")
    logger.info(f"uuid:{st.session_state['uuid']}\tuploaded file: {original_filename}")

    st.session_state["files"].append(
        {
            "file_path": file_path,
            "file_name": file_name,
            "uid": file_uid,
            "created_at": current_time,
        }
    )


def render_file_list() -> None:
    st.markdown("### 已上传文档")
    header_left, header_right = st.columns([3, 2])
    header_left.markdown("**文件名**")
    header_right.markdown("**创建时间**")
    for file in st.session_state["files"]:
        name_col, time_col = st.columns([3, 2])
        name_col.write(file["file_name"])
        time_col.write(file["created_at"])


def main() -> None:
    st.set_page_config(page_title="文件中心", page_icon="📁")
    st.title("📁 文件中心")

    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"
    if "files" not in st.session_state:
        st.session_state["files"] = []

    init_database("./database.sqlite")
    ensure_local_user(st.session_state["uuid"], db_name="./database.sqlite")

    repo_root = Path(__file__).resolve().parents[1]
    save_dir = str(repo_root / "uploads")
    os.makedirs(save_dir, exist_ok=True)
    logger = LoggerManager().get_logger()

    upload_file(save_dir, logger)
    load_files()
    if st.session_state["files"]:
        render_file_list()
    else:
        st.caption("还没有上传文档。")


main()
