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
from .mindmap_format import mindmap_format_middleware
from .orchestration import OrchestrationMiddleware
from .plan import plan_middleware
from .team import TeamMiddleware
from .todolist import todolist_middleware
from .tool_selector import build_tool_selector_middleware
from .trace import TraceMiddleware
from .turn_context import turn_context_middleware

_RUNTIME_MIDDLEWARE_IDS = (
    "trace",
    "llm_logger",
    "orchestration",
    "subagent",
    "team",
    "todolist",
    "plan",
)


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


def _is_enabled(profile: Any | None, middleware_id: str) -> bool:
    if profile is None:
        return middleware_id in _RUNTIME_MIDDLEWARE_IDS
    return middleware_id in set(getattr(profile, "middleware_ids", ()))


def build_middleware_list(
    model: Any,
    profile: Any | None = None,
    deps: Any | None = None,
    enable_auto_summarization: bool = True,
    enable_tool_selector: bool = True,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build complete middleware list for agent runtime."""
    middleware_list: list[AgentMiddleware[Any, Any, Any]] = []

    if _is_enabled(profile, "trace"):
        middleware_list.append(TraceMiddleware())

    if _is_enabled(profile, "llm_logger"):
        middleware_list.append(llm_logger_middleware)

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

    middleware_list.append(turn_context_middleware)

    if _is_enabled(profile, "orchestration"):
        middleware_list.append(OrchestrationMiddleware(llm=model))

    # Enforce strict tagged JSON contract for mindmap outputs.
    middleware_list.append(mindmap_format_middleware)

    if _is_enabled(profile, "subagent"):
        subagent_specs = _build_runtime_subagent_specs(model)
        if subagent_specs:
            middleware_list.append(
                SubAgentMiddleware(
                    backend=StateBackend,
                    subagents=subagent_specs,
                )
            )

    if _is_enabled(profile, "team"):
        middleware_list.append(TeamMiddleware(default_model=model, dependencies=deps))

    if _is_enabled(profile, "todolist"):
        middleware_list.append(todolist_middleware)

    if _is_enabled(profile, "plan"):
        middleware_list.append(plan_middleware)

    if enable_tool_selector:
        middleware_list.append(build_tool_selector_middleware(model))

    if enable_auto_summarization:
        middleware_list.append(
            SummarizationMiddleware(
                model=model,
                trigger=[("fraction", 0.55)],
                keep=("messages", 20),
            )
        )

    return middleware_list


__all__ = ["build_middleware_list"]
