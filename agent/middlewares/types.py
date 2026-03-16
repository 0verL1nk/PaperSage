"""Type definitions for middleware."""

from typing_extensions import NotRequired

from langchain.agents.middleware import AgentState as BaseAgentState


class AgentState(BaseAgentState):
    """Extended AgentState with trace fields."""

    trace_labels: NotRequired[list[str]]
    trace_summary: NotRequired[str]
