import streamlit as st

from agent.adapters import (
    apply_runtime_tuning_env_for_user,
    read_api_key_for_user,
    read_base_url_for_user,
    read_model_name_for_user,
    read_policy_router_api_key_for_user,
    read_policy_router_base_url_for_user,
    read_policy_router_model_name_for_user,
    read_runtime_tuning_settings_for_user,
    save_api_key_for_user,
    save_base_url_for_user,
    save_model_name_for_user,
    save_policy_router_api_key_for_user,
    save_policy_router_base_url_for_user,
    save_policy_router_model_name_for_user,
    save_runtime_tuning_settings_for_user,
)
from agent.settings import load_agent_settings
from ui.page_bootstrap import bootstrap_page_context


def main() -> None:
    st.set_page_config(page_title="设置中心", page_icon="⚙️")
    st.title("⚙️ 设置中心")

    user_uuid = bootstrap_page_context(
        session_state=st.session_state,
        db_name="./database.sqlite",
    )
    saved_api_key = read_api_key_for_user(uuid=user_uuid)
    saved_model_name = read_model_name_for_user(uuid=user_uuid) or ""
    saved_base_url = read_base_url_for_user(uuid=user_uuid) or ""
    saved_policy_router_model_name = (
        read_policy_router_model_name_for_user(uuid=user_uuid) or ""
    )
    saved_policy_router_base_url = (
        read_policy_router_base_url_for_user(uuid=user_uuid) or ""
    )
    saved_policy_router_api_key = (
        read_policy_router_api_key_for_user(uuid=user_uuid) or ""
    )
    saved_runtime_tuning = read_runtime_tuning_settings_for_user(uuid=user_uuid)
    settings = load_agent_settings()

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
    st.markdown("### 异步策略与内存控制")
    st.caption("用于异步模式拦截、RAG 建索引批量与文档缓存上限。")
    async_enabled_default = settings.agent_policy_async_enabled
    async_enabled = st.checkbox(
        "启用异步策略拦截",
        value=(
            saved_runtime_tuning["agent_policy_async_enabled"]
            if saved_runtime_tuning["agent_policy_async_enabled"] is not None
            else async_enabled_default
        ),
    )
    async_refresh_seconds = st.number_input(
        "异步刷新间隔（秒）",
        min_value=0.5,
        step=0.5,
        value=float(
            saved_runtime_tuning["agent_policy_async_refresh_seconds"]
            if saved_runtime_tuning["agent_policy_async_refresh_seconds"] is not None
            else settings.agent_policy_async_refresh_seconds
        ),
    )
    async_min_confidence = st.number_input(
        "异步最低置信度（0-1）",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=float(
            saved_runtime_tuning["agent_policy_async_min_confidence"]
            if saved_runtime_tuning["agent_policy_async_min_confidence"] is not None
            else settings.agent_policy_async_min_confidence
        ),
    )
    async_max_staleness_seconds = st.number_input(
        "异步结果最大过期时间（秒）",
        min_value=1.0,
        step=1.0,
        value=float(
            saved_runtime_tuning["agent_policy_async_max_staleness_seconds"]
            if saved_runtime_tuning["agent_policy_async_max_staleness_seconds"] is not None
            else settings.agent_policy_async_max_staleness_seconds
        ),
    )

    rag_index_batch_size = st.number_input(
        "向量索引批大小",
        min_value=1,
        step=32,
        value=int(
            saved_runtime_tuning["rag_index_batch_size"]
            if saved_runtime_tuning["rag_index_batch_size"] is not None
            else 256
        ),
        help="越小越省内存，索引构建会更慢。",
    )
    document_text_cache_max_chars = st.number_input(
        "文档文本缓存上限（字符）",
        min_value=0,
        step=100000,
        value=int(
            saved_runtime_tuning["agent_document_text_cache_max_chars"]
            if saved_runtime_tuning["agent_document_text_cache_max_chars"] is not None
            else 1_200_000
        ),
        help="0 表示不限制。",
    )
    local_rag_project_max_chars = st.number_input(
        "项目 RAG 最大字符数",
        min_value=0,
        step=10000,
        value=int(
            saved_runtime_tuning["local_rag_project_max_chars"]
            if saved_runtime_tuning["local_rag_project_max_chars"] is not None
            else settings.rag_project_max_chars
        ),
        help="0 表示不限制。",
    )
    local_rag_project_max_chunks = st.number_input(
        "项目 RAG 最大分块数",
        min_value=0,
        step=100,
        value=int(
            saved_runtime_tuning["local_rag_project_max_chunks"]
            if saved_runtime_tuning["local_rag_project_max_chunks"] is not None
            else settings.rag_project_max_chunks
        ),
        help="0 表示不限制。",
    )

    if st.button("保存设置", type="primary"):
        save_api_key_for_user(uuid=user_uuid, api_key=api_key)
        save_model_name_for_user(uuid=user_uuid, model_name=model_name.strip())
        save_base_url_for_user(uuid=user_uuid, base_url=base_url.strip() or None)
        save_policy_router_model_name_for_user(
            uuid=user_uuid,
            model_name=policy_router_model_name.strip() or None,
        )
        save_policy_router_base_url_for_user(
            uuid=user_uuid,
            base_url=policy_router_base_url.strip() or None,
        )
        save_policy_router_api_key_for_user(
            uuid=user_uuid,
            api_key=policy_router_api_key.strip() or None,
        )
        save_runtime_tuning_settings_for_user(
            user_uuid,
            agent_policy_async_enabled=async_enabled,
            agent_policy_async_refresh_seconds=float(async_refresh_seconds),
            agent_policy_async_min_confidence=float(async_min_confidence),
            agent_policy_async_max_staleness_seconds=float(async_max_staleness_seconds),
            rag_index_batch_size=int(rag_index_batch_size),
            agent_document_text_cache_max_chars=int(document_text_cache_max_chars),
            local_rag_project_max_chars=int(local_rag_project_max_chars),
            local_rag_project_max_chunks=int(local_rag_project_max_chunks),
        )
        apply_runtime_tuning_env_for_user(uuid=user_uuid)
        st.success("设置已保存")


main()
