"""Tool selector middleware configuration."""

from typing import Any

from langchain.agents.middleware import LLMToolSelectorMiddleware


def build_tool_selector_middleware(model: Any) -> LLMToolSelectorMiddleware:
    """Build LLM tool selector middleware with default configuration.

    Args:
        model: The model to use for tool selection.

    Returns:
        Configured LLMToolSelectorMiddleware instance.
    """
    return LLMToolSelectorMiddleware(
        model=model,
        max_tools=8,
        always_include=["ask_human", "search_document"],
    )


__all__ = ["build_tool_selector_middleware"]
