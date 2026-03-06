import streamlit as st

from utils.utils import (
    ensure_local_user,
    get_api_key,
    get_base_url,
    get_model_name,
    get_policy_router_api_key,
    get_policy_router_base_url,
    get_policy_router_model_name,
    init_database,
    save_api_key,
    save_base_url,
    save_model_name,
    save_policy_router_api_key,
    save_policy_router_base_url,
    save_policy_router_model_name,
)


def main() -> None:
    st.set_page_config(page_title="设置中心", page_icon="⚙️")
    st.title("⚙️ 设置中心")

    init_database("./database.sqlite")
    if "uuid" not in st.session_state or not st.session_state["uuid"]:
        st.session_state["uuid"] = "local-user"
    ensure_local_user(st.session_state["uuid"], db_name="./database.sqlite")

    user_uuid = st.session_state["uuid"]
    saved_api_key = get_api_key(user_uuid)
    saved_model_name = get_model_name(user_uuid) or ""
    saved_base_url = get_base_url(user_uuid) or ""
    saved_policy_router_model_name = get_policy_router_model_name(user_uuid) or ""
    saved_policy_router_base_url = get_policy_router_base_url(user_uuid) or ""
    saved_policy_router_api_key = get_policy_router_api_key(user_uuid) or ""

    st.caption("这里用于集中管理当前用户的模型配置。")
    api_key = st.text_input("API Key", value=saved_api_key, type="password")
    model_name = st.text_input(
        "模型名称",
        value=saved_model_name,
        placeholder="例如: gpt-4o-mini / qwen-plus",
    )
    base_url = st.text_input(
        "OpenAI Compatible Base URL",
        value=saved_base_url,
        placeholder="例如: https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    st.markdown("### 策略路由小模型（可选）")
    st.caption("用于模式选择与异步拦截；留空则回退主模型配置。")
    policy_router_model_name = st.text_input(
        "路由模型名称",
        value=saved_policy_router_model_name,
        placeholder="例如: qwen-turbo / gpt-4.1-mini",
    )
    policy_router_base_url = st.text_input(
        "路由模型 Base URL",
        value=saved_policy_router_base_url,
        placeholder="留空则复用主 Base URL",
    )
    policy_router_api_key = st.text_input(
        "路由模型 API Key",
        value=saved_policy_router_api_key,
        type="password",
        placeholder="留空则复用主 API Key",
    )

    if st.button("保存设置", type="primary"):
        save_api_key(user_uuid, api_key)
        save_model_name(user_uuid, model_name.strip())
        save_base_url(user_uuid, base_url.strip() or None)
        save_policy_router_model_name(user_uuid, policy_router_model_name.strip() or None)
        save_policy_router_base_url(user_uuid, policy_router_base_url.strip() or None)
        save_policy_router_api_key(user_uuid, policy_router_api_key.strip() or None)
        st.success("设置已保存")


main()
