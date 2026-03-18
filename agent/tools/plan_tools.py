"""Plan management tools for agent-centric orchestration.

These tools allow the leader agent to create and manage execution plans
based on their own understanding of the task context.
"""

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field


class WritePlanInput(BaseModel):
    """Input for write_plan tool."""

    goal: str | None = Field(default=None, description="The overall goal (required for new plan)")
    description: str | None = Field(default=None, description="Plan description (empty to delete)")


@tool("write_plan", args_schema=WritePlanInput)
def write_plan(
    goal: str | None,
    description: str | None,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
) -> Command[Any]:
    """Write or delete execution plan.

    - Empty description: delete plan
    - With description: create/update plan (goal required for new plan)
    """
    # Delete if description is empty
    if not description or not description.strip():
        if not state.get("plan"):
            return Command(update={"messages": [ToolMessage("No plan to delete.", tool_call_id=tool_call_id)]})
        return Command(
            update={
                "plan": None,
                "messages": [ToolMessage("Plan deleted.", tool_call_id=tool_call_id)]
            }
        )

    # Create/update plan
    existing_plan = state.get("plan")

    # If no existing plan, goal is required
    if not existing_plan and not goal:
        return Command(
            update={
                "messages": [ToolMessage("Error: goal required for new plan.", tool_call_id=tool_call_id)]
            }
        )

    # Use existing goal if not provided
    final_goal = goal if goal else (existing_plan or {}).get("goal", "")

    plan_data = {"goal": final_goal, "description": description.strip()}

    message = "Plan created." if not existing_plan else "Plan updated."

    return Command(
        update={
            "plan": plan_data,
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)]
        }
    )


@tool("read_plan")
def read_plan(state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Read the current execution plan.

    Returns the plan you previously created, including goal and description.

    Returns:
        Plan content or message if no plan exists
    """
    plan = state.get("plan")
    if not plan:
        return "No active plan. Create one with write_plan if needed."

    goal = plan.get("goal", "")
    description = plan.get("description", "")

    return f"**Goal:** {goal}\n\n**Strategy:**\n{description}"


# Export tools
PLAN_TOOLS = [write_plan, read_plan]
