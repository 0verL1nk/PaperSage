from typing import Any

from ..llm_provider import build_openai_compatible_chat_model


def create_chat_model(
    *,
    api_key: str,
    model_name: str,
    base_url: str | None = None,
    temperature: float | None = None,
) -> Any:
    return build_openai_compatible_chat_model(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
    )
