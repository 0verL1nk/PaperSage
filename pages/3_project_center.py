import streamlit as st

from ui.theme import inject_global_theme
from utils.utils import (
    create_project,
    ensure_default_project_for_user,
    ensure_local_user,
    init_database,
    list_project_files,
    list_projects,
    update_project,
)


def _project_summary(project_uid: str, user_uuid: str) -> str:
    files = list_project_files(project_uid=project_uid, uuid=user_uuid, active_only=False)
    return f"文档数: {len(files)}"


def main() -> None:
    st.set_page_config(page_title="项目中心", page_icon="🗂️")
    st.title("🗂️ 项目中心")
    inject_global_theme()

    init_database("./database.sqlite")
    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"
    user_uuid = st.session_state["uuid"]
    ensure_local_user(user_uuid, db_name="./database.sqlite")
    ensure_default_project_for_user(
        user_uuid,
        db_name="./database.sqlite",
        sync_existing_files=False,
    )

    with st.expander("新建项目", expanded=False):
        new_project_name = st.text_input("项目名称", key="project_center_new_name")
        new_description = st.text_area("项目描述", key="project_center_new_desc")
        if st.button("创建项目", key="project_center_create_btn"):
            if not new_project_name.strip():
                st.warning("项目名称不能为空")
            else:
                create_project(
                    uuid=user_uuid,
                    project_name=new_project_name.strip(),
                    description=new_description.strip(),
                )
                st.success("项目已创建")
                st.rerun()

    projects = list_projects(user_uuid, include_archived=True)
    if not projects:
        st.caption("暂无项目")
        return

    active_count = len([item for item in projects if int(item.get("archived", 0)) == 0])
    archived_count = len(projects) - active_count
    st.markdown(
        (
            "<div class=\"llm-section-card\">"
            "<h4>项目总览</h4>"
            f"<div class=\"llm-chip-row\"><span class=\"llm-chip\">总项目 {len(projects)}</span>"
            f"<span class=\"llm-chip\">活跃 {active_count}</span>"
            f"<span class=\"llm-chip\">归档 {archived_count}</span></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    for project in projects:
        project_uid = str(project["project_uid"])
        project_name = str(project["project_name"])
        archived = int(project.get("archived", 0)) == 1
        with st.container(border=True):
            st.markdown(f"### {project_name}")
            st.caption(
                f"{_project_summary(project_uid, user_uuid)} | 更新时间: {project.get('updated_at', '')}"
            )
            if project.get("description"):
                st.write(str(project["description"]))

            col_rename, col_archive = st.columns(2)
            with col_rename:
                new_name = st.text_input(
                    "重命名项目",
                    value=project_name,
                    key=f"project_center_rename_{project_uid}",
                )
                if st.button("保存名称", key=f"project_center_rename_btn_{project_uid}"):
                    update_project(
                        project_uid=project_uid,
                        uuid=user_uuid,
                        project_name=new_name.strip(),
                    )
                    st.success("名称已更新")
                    st.rerun()
            with col_archive:
                if archived:
                    if st.button("取消归档", key=f"project_center_unarchive_{project_uid}"):
                        update_project(
                            project_uid=project_uid,
                            uuid=user_uuid,
                            archived=0,
                        )
                        st.success("已取消归档")
                        st.rerun()
                else:
                    if st.button("归档项目", key=f"project_center_archive_{project_uid}"):
                        update_project(
                            project_uid=project_uid,
                            uuid=user_uuid,
                            archived=1,
                        )
                        st.success("项目已归档")
                        st.rerun()


main()
