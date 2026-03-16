"""Trace middleware for tracking agent execution phases."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from ..domain.trace import phase_label_from_performative, phase_summary
from .types import AgentState


class TraceMiddleware(AgentMiddleware):
    """Middleware that automatically tracks agent execution phases."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Extract and record performative before model call."""
        # Initialize trace_labels if not exists
        if "trace_labels" not in state:
            state["trace_labels"] = []

        # Extract performative from last message if available
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            performative = getattr(last_msg, "performative", None)
            if performative:
                label = phase_label_from_performative(str(performative))
                state["trace_labels"].append(label)

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Record performative from model response."""
        # Extract performative from the newly added AIMessage
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            performative = getattr(last_msg, "performative", None)
            if performative:
                label = phase_label_from_performative(str(performative))
                if "trace_labels" not in state:
                    state["trace_labels"] = []
                state["trace_labels"].append(label)

        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Generate phase summary after agent completes."""
        trace_labels = state.get("trace_labels", [])
        summary = phase_summary(trace_labels)
        state["trace_summary"] = summary
        return None
