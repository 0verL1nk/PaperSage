from utils.utils import (
    get_user_api_key,
    get_user_base_url,
    get_user_files,
    get_user_model_name,
    get_user_policy_router_api_key,
    get_user_policy_router_base_url,
    get_user_policy_router_model_name,
)


def read_user_api_key() -> str:
    return get_user_api_key()


def read_user_base_url() -> str:
    return get_user_base_url()


def read_user_model_name() -> str:
    return get_user_model_name()


def read_user_policy_router_model_name() -> str:
    return get_user_policy_router_model_name()


def read_user_policy_router_base_url() -> str:
    return get_user_policy_router_base_url()


def read_user_policy_router_api_key() -> str:
    return get_user_policy_router_api_key()


def list_user_files(*, uuid: str):
    return get_user_files(uuid)
