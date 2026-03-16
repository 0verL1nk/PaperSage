"""Trace middleware for tracking agent execution phases."""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from .types import AgentState


class TraceMiddleware(AgentMiddleware):
    """Middleware that emits trace events during agent execution."""

    def before_model(
        self, state: AgentState, runtime: Runtime, config: RunnableConfig
    ) -> dict[str, Any] | None:
        """Emit model_call event before model invocation."""
        on_event = config.get("configurable", {}).get("on_event")
        if callable(on_event):
            on_event({
                "sender": "agent",
                "receiver": "model",
                "performative": "model_call",
                "content": "调用模型",
            })
        return None

    def after_model(
        self, state: AgentState, runtime: Runtime, config: RunnableConfig
    ) -> dict[str, Any] | None:
        """Emit model_response event after model returns."""
        on_event = config.get("configurable", {}).get("on_event")
        if callable(on_event):
            on_event({
                "sender": "model",
                "receiver": "agent",
                "performative": "model_response",
                "content": "模型响应",
            })
        return None

    def after_agent(
        self, state: AgentState, runtime: Runtime, config: RunnableConfig
    ) -> dict[str, Any] | None:
        """Emit agent_complete event after agent finishes."""
        on_event = config.get("configurable", {}).get("on_event")
        if callable(on_event):
            on_event({
                "sender": "agent",
                "receiver": "coordinator",
                "performative": "agent_complete",
                "content": "Agent执行完成",
            })
        return None
