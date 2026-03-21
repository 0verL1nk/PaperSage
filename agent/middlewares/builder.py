"""Middleware builder for agent runtime."""

from typing import Any

from deepagents import CompiledSubAgent, SubAgent
from deepagents.backends import StateBackend
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)
from openai import RateLimitError

from ..subagent.loader import load_subagent_configs
from .llm_logger import llm_logger_middleware
from .orchestration import OrchestrationMiddleware
from .plan import plan_middleware
from .team import TeamMiddleware
from .todolist import todolist_middleware
from .tool_selector import build_tool_selector_middleware
from .trace import TraceMiddleware
from .turn_context import turn_context_middleware


def _build_runtime_subagent_specs(model: Any) -> list[SubAgent | CompiledSubAgent]:
    """Expand file-based subagent configs to the fully-specified deepagents shape."""
    subagent_specs: list[SubAgent | CompiledSubAgent] = []

    for config in load_subagent_configs():
        spec: SubAgent = {
            "name": config["name"],
            "description": config["description"],
            "system_prompt": config["system_prompt"],
            "model": config["model"] if "model" in config else model,
            "tools": [],
        }
        subagent_specs.append(spec)

    return subagent_specs


def build_middleware_list(
    model: Any,
    enable_auto_summarization: bool = True,
    enable_tool_selector: bool = True,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build complete middleware list for agent runtime.

    Args:
        model: The model to use for middleware that require it.
        enable_auto_summarization: Whether to enable auto summarization.
        enable_tool_selector: Whether to enable LLM tool selector (requires JSON mode support).

    Returns:
        List of configured middleware instances.
    """
    middleware_list: list[AgentMiddleware[Any, Any, Any]] = []

    # Trace middleware (first to record all middleware execution)
    middleware_list.append(TraceMiddleware())

    # LLM logger middleware (logs complete LLM input/output)
    middleware_list.append(llm_logger_middleware)

    # Model retry middleware (handles rate limits with exponential backoff)
    middleware_list.append(
        ModelRetryMiddleware(
            max_retries=3,
            retry_on=(RateLimitError,),
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )
    )

    # Dynamic per-turn system context injection
    middleware_list.append(turn_context_middleware)

    # Orchestration middleware (for complex task guidance)
    middleware_list.append(OrchestrationMiddleware(llm=model))

    # SubAgent middleware (provides task tool for spawning subagents)
    subagent_specs = _build_runtime_subagent_specs(model)
    if subagent_specs:
        middleware_list.append(
            SubAgentMiddleware(
                backend=StateBackend,
                subagents=subagent_specs,
            )
        )

    # Team middleware (provides team management tools)
    middleware_list.append(TeamMiddleware(default_model=model))

    # Enhanced TodoList middleware (supports dependencies)
    middleware_list.append(todolist_middleware)

    # Plan middleware (extends state to support plan field)
    middleware_list.append(plan_middleware)

    # LLM Tool Selector middleware (requires JSON mode support)
    if enable_tool_selector:
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
