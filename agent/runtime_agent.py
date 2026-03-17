from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent

from .capabilities import (
    build_agent_tools,
    discover_available_tools,
)
from .middlewares import build_middleware_list, todolist_middleware

SPAWN_TOOL_NAMES = {"start_plan", "start_team"}


def build_runtime_tools(
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
    list_documents_fn: Callable[[], list[dict[str, Any]]] | None = None,
    allowed_tools: set[str] | None = None,
    blocked_tools: set[str] | None = None,
) -> list[Any]:
    normalized_allowlist: set[str] | None = None
    if allowed_tools is not None:
        normalized_allowlist = {name.strip().lower() for name in allowed_tools if name.strip()}

    normalized_blocklist = {name.strip().lower() for name in blocked_tools or set() if name.strip()}
    if normalized_blocklist:
        if normalized_allowlist is None:
            discovered_tools = discover_available_tools(
                read_document_enabled=read_document_fn is not None,
                list_document_enabled=list_documents_fn is not None,
            )
            normalized_allowlist = {
                item.name for item in discovered_tools if item.name not in normalized_blocklist
            }
        else:
            normalized_allowlist = normalized_allowlist - normalized_blocklist

    return build_agent_tools(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
        list_documents_fn=list_documents_fn,
        allowed_tools=normalized_allowlist,
    )


def create_runtime_agent(
    *,
    model: Any,
    system_prompt: str,
    tools: list[Any],
    checkpointer: Any | None = None,
    enable_auto_summarization: bool = True,
) -> Any:
    # Add TodoListMiddleware tools to the tools list
    all_tools = list(tools) + list(todolist_middleware.tools)

    # Build middleware list
    middleware_list = build_middleware_list(
        model=model,
        enable_auto_summarization=enable_auto_summarization,
    )

    create_kwargs: dict[str, Any] = {
        "model": model,
        "tools": all_tools,
        "system_prompt": system_prompt,
    }
    if checkpointer is not None:
        create_kwargs["checkpointer"] = checkpointer
    if middleware_list:
        create_kwargs["middleware"] = middleware_list
    return create_agent(**create_kwargs)
