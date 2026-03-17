"""Plan middleware for managing execution plans in agent state."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.todo import PlanningState
from typing_extensions import NotRequired, TypedDict


class PlanStateExtension(TypedDict):
    """Extension to PlanningState that adds plan field."""

    plan: NotRequired[dict[str, str] | None]
    """Current execution plan with goal and description."""


class ExtendedPlanningState(PlanningState[Any], PlanStateExtension):
    """Extended state schema that includes both todos and plan."""

    pass


class PlanMiddleware(AgentMiddleware[ExtendedPlanningState, Any, Any]):
    """Middleware that extends PlanningState to support plan management.

    This middleware adds a 'plan' field to the agent state, allowing
    plan_tools to persist plan data across agent invocations.
    """

    state_schema = ExtendedPlanningState

    def __init__(self) -> None:
        """Initialize the PlanMiddleware."""
        super().__init__()


# Create singleton instance
plan_middleware = PlanMiddleware()

__all__ = ["plan_middleware", "ExtendedPlanningState"]
