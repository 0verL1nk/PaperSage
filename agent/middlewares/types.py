"""Type definitions for middleware."""

from langchain.agents.middleware import AgentState as BaseAgentState
from typing_extensions import NotRequired


class AgentState(BaseAgentState):
    """Extended AgentState with trace fields."""

    trace_labels: NotRequired[list[str]]
    trace_summary: NotRequired[str]
