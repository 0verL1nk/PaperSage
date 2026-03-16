from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .settings import load_agent_settings


def _get_model_max_input_tokens(model_name: str) -> int:
    """根据模型名称返回最大输入token数,默认200,000"""
    # 统一默认值为 200,000
    return 200000


def _provider_supports_reasoning_effort(base_url: str) -> bool:
    normalized = base_url.lower()
    return "api.openai.com" in normalized


def _provider_supports_enable_thinking_flag(base_url: str) -> bool:
    normalized = base_url.lower()
    return "dashscope.aliyuncs.com" in normalized


def build_openai_compatible_chat_model(
    api_key: str,
    model_name: str,
    temperature: float | None = None,
    base_url: str | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> ChatOpenAI:
    settings = load_agent_settings()
    resolved_temperature = (
        settings.agent_temperature if temperature is None else temperature
    )
    resolved_base_url = (
        settings.openai_compatible_base_url if not base_url else base_url
    )
    resolved_enable_thinking = (
        settings.agent_enable_thinking if enable_thinking is None else enable_thinking
    )
    resolved_reasoning_effort = (
        settings.agent_reasoning_effort
        if reasoning_effort is None
        else reasoning_effort.strip()
    )
    resolved_timeout = timeout if timeout is not None else settings.agent_llm_request_timeout

    resolved_reasoning: str | None = None
    resolved_extra_body: dict[str, object] | None = None
    if resolved_enable_thinking:
        if (
            resolved_reasoning_effort
            and _provider_supports_reasoning_effort(resolved_base_url)
        ):
            resolved_reasoning = resolved_reasoning_effort
        if _provider_supports_enable_thinking_flag(resolved_base_url):
            resolved_extra_body = {"enable_thinking": True}

    max_input_tokens = _get_model_max_input_tokens(model_name)
    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=resolved_base_url,
        temperature=resolved_temperature,
        timeout=resolved_timeout,
        reasoning_effort=resolved_reasoning,
        extra_body=resolved_extra_body,
        profile={"max_input_tokens": max_input_tokens},
    )
