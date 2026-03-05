import datetime
import hashlib
import os
from pathlib import Path
import uuid

import streamlit as st

from ui.theme import inject_global_theme
from utils.utils import (
    LoggerManager,
    add_file_to_project,
    check_file_exists,
    ensure_default_project_for_user,
    ensure_local_user,
    get_file_project_counts,
    get_uid_by_md5,
    list_project_files,
    list_projects,
    remove_file_from_project,
    get_user_files,
    init_database,
    save_file_to_database,
)


def calculate_md5(uploaded_file) -> str:
    md5_hash = hashlib.md5()
    for chunk in iter(lambda: uploaded_file.read(4096), b""):
        md5_hash.update(chunk)
    return md5_hash.hexdigest()


def load_all_files() -> None:
    files = get_user_files(st.session_state["uuid"])
    st.session_state["all_files"] = []
    for file in files:
        st.session_state["all_files"].append(
            {
                "file_path": file["file_path"],
                "file_name": file["file_name"],
                "uid": file["uid"],
                "created_at": file["created_at"],
            }
        )


def load_files(project_uid: str | None) -> None:
    if project_uid:
        files = list_project_files(
            project_uid=project_uid,
            uuid=st.session_state["uuid"],
            active_only=False,
        )
    else:
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


def load_projects() -> None:
    st.session_state["projects"] = list_projects(st.session_state["uuid"])


def selected_project_uid() -> str | None:
    projects = st.session_state.get("projects", [])
    if not isinstance(projects, list) or not projects:
        return None
    names = [str(item.get("project_name") or "未命名项目") for item in projects]
    selected_name = st.selectbox("当前项目", names, key="file_center_project_selector")
    index = names.index(selected_name)
    project_uid = projects[index].get("project_uid")
    return str(project_uid) if isinstance(project_uid, str) else None


def upload_file(save_dir: str, logger, project_uid: str | None) -> None:
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
        auto_bind_default_project=False,
    )
    if project_uid:
        add_file_to_project(project_uid, file_uid, st.session_state["uuid"])
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
    project_counts = get_file_project_counts(st.session_state["uuid"])
    st.markdown("### 已上传文档")
    header_left, header_mid, header_right = st.columns([3, 2, 1])
    header_left.markdown("**文件名**")
    header_mid.markdown("**创建时间**")
    header_right.markdown("**项目数**")
    for file in st.session_state["files"]:
        count = int(project_counts.get(str(file["uid"]), 0))
        name_col, time_col, count_col = st.columns([3, 2, 1])
        name_col.write(file["file_name"])
        time_col.write(file["created_at"])
        count_col.write(str(count))


def render_project_binding(project_uid: str) -> None:
    all_files = st.session_state.get("all_files", [])
    if not isinstance(all_files, list) or not all_files:
        return
    project_files = list_project_files(
        project_uid=project_uid,
        uuid=st.session_state["uuid"],
        active_only=False,
    )
    selected_current = {str(item["uid"]) for item in project_files}
    options = [str(item["uid"]) for item in all_files]
    label_map = {str(item["uid"]): str(item["file_name"]) for item in all_files}
    selected = st.multiselect(
        "项目文档绑定",
        options=options,
        default=[uid for uid in options if uid in selected_current],
        format_func=lambda uid: f"{label_map.get(uid, uid)} ({uid[:8]})",
        key=f"file_center_binding_{project_uid}",
    )
    if st.button("保存项目文档绑定", key=f"file_center_binding_save_{project_uid}"):
        selected_set = {str(item) for item in selected}
        for file_uid in selected_set:
            add_file_to_project(project_uid, file_uid, st.session_state["uuid"])
        for file_uid in selected_current - selected_set:
            remove_file_from_project(project_uid, file_uid, st.session_state["uuid"])
        st.success("项目文档绑定已更新")
        load_all_files()
        load_files(project_uid)
        load_projects()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="文件中心", page_icon="📁")
    st.title("📁 文件中心")
    inject_global_theme()

    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"
    if "files" not in st.session_state:
        st.session_state["files"] = []
    if "all_files" not in st.session_state:
        st.session_state["all_files"] = []

    init_database("./database.sqlite")
    ensure_local_user(st.session_state["uuid"], db_name="./database.sqlite")
    ensure_default_project_for_user(
        st.session_state["uuid"],
        db_name="./database.sqlite",
        sync_existing_files=False,
    )

    repo_root = Path(__file__).resolve().parents[1]
    save_dir = str(repo_root / "uploads")
    os.makedirs(save_dir, exist_ok=True)
    logger = LoggerManager().get_logger()

    load_projects()
    project_uid = selected_project_uid()
    load_all_files()
    load_files(project_uid)
    if project_uid:
        st.markdown(
            (
                "<div class=\"llm-section-card\">"
                "<h4>项目摘要</h4>"
                f"<div class=\"llm-chip-row\"><span class=\"llm-chip\">项目文档数 {len(st.session_state.get('files', []))}</span>"
                f"<span class=\"llm-chip\">用户总文档数 {len(st.session_state.get('all_files', []))}</span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    upload_file(save_dir, logger, project_uid)
    load_all_files()
    load_files(project_uid)
    if st.session_state["files"]:
        render_file_list()
    else:
        st.caption("当前项目还没有文档。")
    if project_uid:
        with st.expander("项目文档管理", expanded=False):
            render_project_binding(project_uid)


main()
