from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent

from .middlewares import build_middleware_list
from .tools.builder import build_agent_tools


def build_runtime_tools(
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
    list_documents_fn: Callable[[], list[dict[str, Any]]] | None = None,
) -> list[Any]:
    return build_agent_tools(
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
        list_documents_fn=list_documents_fn,
    )


def create_runtime_agent(
    *,
    model: Any,
    system_prompt: str,
    tools: list[Any],
    checkpointer: Any | None = None,
    enable_auto_summarization: bool = True,
    enable_tool_selector: bool = True,
) -> Any:
    # Build middleware list
    middleware_list = build_middleware_list(
        model=model,
        enable_auto_summarization=enable_auto_summarization,
        enable_tool_selector=enable_tool_selector,
    )

    # Collect tools from all middleware
    all_tools = list(tools)
    for middleware in middleware_list:
        if hasattr(middleware, "tools"):
            all_tools.extend(middleware.tools)

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
