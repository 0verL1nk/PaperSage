from __future__ import annotations

from agent.application.runtime_tuning import apply_runtime_tuning_env
from utils.utils import (
    get_api_key,
    get_base_url,
    get_model_name,
    get_policy_router_api_key,
    get_policy_router_base_url,
    get_policy_router_model_name,
    get_runtime_tuning_settings,
    get_user_api_key,
    get_user_base_url,
    get_user_files,
    get_user_model_name,
    get_user_policy_router_api_key,
    get_user_policy_router_base_url,
    get_user_policy_router_model_name,
    save_api_key,
    save_base_url,
    save_model_name,
    save_policy_router_api_key,
    save_policy_router_base_url,
    save_policy_router_model_name,
    save_runtime_tuning_settings,
)


def read_user_api_key() -> str | None:
    return get_user_api_key()


def read_user_base_url() -> str | None:
    return get_user_base_url()


def read_user_model_name() -> str | None:
    return get_user_model_name()


def read_user_policy_router_model_name() -> str | None:
    return get_user_policy_router_model_name()


def read_user_policy_router_base_url() -> str | None:
    return get_user_policy_router_base_url()


def read_user_policy_router_api_key() -> str | None:
    return get_user_policy_router_api_key()


def list_user_files(*, uuid: str):
    return get_user_files(uuid)


def read_api_key_for_user(*, uuid: str) -> str:
    return get_api_key(uuid)


def read_model_name_for_user(*, uuid: str) -> str | None:
    return get_model_name(uuid)


def read_base_url_for_user(*, uuid: str) -> str | None:
    return get_base_url(uuid)


def read_policy_router_model_name_for_user(*, uuid: str) -> str | None:
    return get_policy_router_model_name(uuid)


def read_policy_router_base_url_for_user(*, uuid: str) -> str | None:
    return get_policy_router_base_url(uuid)


def read_policy_router_api_key_for_user(*, uuid: str) -> str | None:
    return get_policy_router_api_key(uuid)


def save_api_key_for_user(*, uuid: str, api_key: str) -> None:
    save_api_key(uuid, api_key)


def save_model_name_for_user(*, uuid: str, model_name: str | None) -> None:
    save_model_name(uuid, model_name)


def save_base_url_for_user(*, uuid: str, base_url: str | None) -> None:
    save_base_url(uuid, base_url)


def save_policy_router_model_name_for_user(
    *, uuid: str, model_name: str | None
) -> None:
    save_policy_router_model_name(uuid, model_name)


def save_policy_router_base_url_for_user(*, uuid: str, base_url: str | None) -> None:
    save_policy_router_base_url(uuid, base_url)


def save_policy_router_api_key_for_user(*, uuid: str, api_key: str | None) -> None:
    save_policy_router_api_key(uuid, api_key)


def read_runtime_tuning_settings_for_user(
    *, uuid: str
) -> dict[str, bool | int | float | None]:
    return get_runtime_tuning_settings(uuid)


def save_runtime_tuning_settings_for_user(
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
) -> None:
    save_runtime_tuning_settings(
        uuid,
        agent_policy_async_enabled=agent_policy_async_enabled,
        agent_policy_async_refresh_seconds=agent_policy_async_refresh_seconds,
        agent_policy_async_min_confidence=agent_policy_async_min_confidence,
        agent_policy_async_max_staleness_seconds=agent_policy_async_max_staleness_seconds,
        rag_index_batch_size=rag_index_batch_size,
        agent_document_text_cache_max_chars=agent_document_text_cache_max_chars,
        local_rag_project_max_chars=local_rag_project_max_chars,
        local_rag_project_max_chunks=local_rag_project_max_chunks,
    )


def apply_runtime_tuning_env_for_user(*, uuid: str) -> dict[str, str]:
    settings = read_runtime_tuning_settings_for_user(uuid=uuid)
    return apply_runtime_tuning_env(settings=settings)
