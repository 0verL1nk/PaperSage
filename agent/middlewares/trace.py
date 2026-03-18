"""Trace middleware for tracking agent execution phases."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from .types import AgentState


class TraceMiddleware(AgentMiddleware):
    """Middleware that emits trace events during agent execution."""

    def before_model(  # type: ignore[override]
        self, state: AgentState, runtime: Runtime, config: RunnableConfig = None
    ) -> dict[str, Any] | None:
        """Emit model_call event before model invocation."""
        if config:
            on_event = config.get("configurable", {}).get("on_event")
            if callable(on_event):
                on_event({
                    "sender": "agent",
                    "receiver": "model",
                    "performative": "model_call",
                    "content": "调用模型",
                })
        return None

    def after_model(  # type: ignore[override]
        self, state: AgentState, runtime: Runtime, config: RunnableConfig = None
    ) -> dict[str, Any] | None:
        """Emit model_response event with tool call details."""
        if not config:
            return None
        on_event = config.get("configurable", {}).get("on_event")
        if not callable(on_event):
            return None

        # Extract tool calls from the latest AIMessage
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)

        if tool_calls and isinstance(tool_calls, list) and tool_calls:
            # Model called tools
            tool_names = [str(call.get("name", "unknown") if isinstance(call, dict) else getattr(call, "name", "unknown")) for call in tool_calls]
            content = f"调用工具: {', '.join(tool_names)}"
        else:
            # Model returned text response
            content = "模型响应"

        on_event({
            "sender": "model",
            "receiver": "agent",
            "performative": "model_response",
            "content": content,
        })
        return None

    def after_agent(  # type: ignore[override]
        self, state: AgentState, runtime: Runtime, config: RunnableConfig = None
    ) -> dict[str, Any] | None:
        """Emit agent_complete event after agent finishes."""
        if not config:
            return None
        on_event = config.get("configurable", {}).get("on_event")
        if callable(on_event):
            on_event({
                "sender": "agent",
                "receiver": "coordinator",
                "performative": "agent_complete",
                "content": "Agent执行完成",
            })
        return None
