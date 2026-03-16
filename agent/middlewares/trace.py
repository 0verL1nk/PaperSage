"""Trace middleware for tracking agent execution phases."""

import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from .types import AgentState

logger = logging.getLogger(__name__)


class TraceMiddleware(AgentMiddleware):
    """Middleware that emits trace events during agent execution."""

    def _get_on_event(self, runtime: Runtime) -> Callable[[dict[str, Any]], None] | None:
        """Extract on_event callback from runtime context."""
        # Try to get from runtime.context (which should contain configurable)
        if hasattr(runtime, "context") and isinstance(runtime.context, dict):
            on_event = runtime.context.get("on_event")
            if on_event:
                return on_event

        # Log for debugging
        logger.debug("TraceMiddleware: on_event not found in runtime.context")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Emit model_call event before model invocation."""
        on_event = self._get_on_event(runtime)
        if on_event:
            on_event({
                "sender": "agent",
                "receiver": "model",
                "performative": "model_call",
                "content": "调用模型",
            })
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Emit model_response event after model returns."""
        on_event = self._get_on_event(runtime)
        if on_event:
            on_event({
                "sender": "model",
                "receiver": "agent",
                "performative": "model_response",
                "content": "模型响应",
            })
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Emit agent_complete event after agent finishes."""
        on_event = self._get_on_event(runtime)
        if on_event:
            on_event({
                "sender": "agent",
                "receiver": "coordinator",
                "performative": "agent_complete",
                "content": "Agent执行完成",
            })
        return None
