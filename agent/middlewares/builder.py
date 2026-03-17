"""Middleware builder for agent runtime."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware

from .orchestration import OrchestrationMiddleware
from .plan import plan_middleware
from .todolist import todolist_middleware
from .tool_selector import build_tool_selector_middleware
from .trace import TraceMiddleware


def build_middleware_list(
    model: Any,
    enable_auto_summarization: bool = True,
) -> list[AgentMiddleware]:
    """Build complete middleware list for agent runtime.

    Args:
        model: The model to use for middleware that require it.
        enable_auto_summarization: Whether to enable auto summarization.

    Returns:
        List of configured middleware instances.
    """
    middleware_list: list[AgentMiddleware] = []

    # Trace middleware (first to record all middleware execution)
    middleware_list.append(TraceMiddleware())

    # Orchestration middleware (for complex task guidance)
    middleware_list.append(OrchestrationMiddleware(llm=model))

    # TodoList middleware (from LangChain)
    middleware_list.append(todolist_middleware)

    # Plan middleware (extends state to support plan field)
    middleware_list.append(plan_middleware)

    # LLM Tool Selector middleware (official implementation)
    tool_selector = build_tool_selector_middleware(model)
    middleware_list.append(tool_selector)

    # Auto summarization middleware
    if enable_auto_summarization:
        summarization_middleware = SummarizationMiddleware(
            model=model,
            trigger=[("fraction", 0.55)],
            keep=("messages", 20),
        )
        middleware_list.append(summarization_middleware)

    return middleware_list


__all__ = ["build_middleware_list"]
