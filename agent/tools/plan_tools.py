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


class CreatePlanInput(BaseModel):
    """Input for create_plan tool."""

    goal: str = Field(description="The overall goal of this plan")
    description: str = Field(description="Detailed strategy and approach written by the agent")


class UpdatePlanInput(BaseModel):
    """Input for update_plan tool."""

    description: str = Field(description="Updated plan description")


@tool("create_plan", args_schema=CreatePlanInput)
def create_plan(
    goal: str,
    description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
) -> Command[Any]:
    """Create a new execution plan.

    This tool stores the plan you write based on your understanding of the task.
    The plan content comes from you - this tool only stores it in agent state.

    If a plan already exists, it will be replaced with the new one.

    Use this when you need to organize a complex multi-step task.
    Write your own strategy based on the conversation context.

    Args:
        goal: The overall goal you want to achieve
        description: Your detailed strategy and approach

    Returns:
        Command to update state
    """
    # Store plan in agent state (replace if exists)
    plan_data = {"goal": goal, "description": description}

    message = (
        f"Plan created successfully. Goal: {goal}\n\n"
        f"Now execute the plan step by step. Start with the first step immediately."
    )

    # Add note if replacing existing plan
    if state.get("plan"):
        message = f"Previous plan replaced. {message}"

    return Command(
        update={
            "plan": plan_data,
            "messages": [
                ToolMessage(message, tool_call_id=tool_call_id)
            ],
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
        return "No active plan. Create one with create_plan if needed."

    goal = plan.get("goal", "")
    description = plan.get("description", "")

    return f"**Goal:** {goal}\n\n**Strategy:**\n{description}"


@tool("update_plan", args_schema=UpdatePlanInput)
def update_plan(
    description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
) -> Command[Any]:
    """Update the plan description.

    Modify your plan based on new information or changed circumstances.
    This updates the description while keeping the same goal.

    Args:
        description: Your updated strategy and approach

    Returns:
        Command to update state
    """
    plan = state.get("plan")

    if not plan:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: No active plan to update. Create one with create_plan first.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    # Update plan description
    updated_plan = dict(plan)
    updated_plan["description"] = description

    return Command(
        update={
            "plan": updated_plan,
            "messages": [
                ToolMessage("Plan updated successfully.", tool_call_id=tool_call_id)
            ],
        }
    )


@tool("delete_plan")
def delete_plan(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
) -> Command[Any]:
    """Delete the current plan.

    Remove the plan from agent state when the task is complete.
    This keeps your working memory clean.

    Returns:
        Command to update state
    """
    if not state.get("plan"):
        return Command(
            update={
                "messages": [
                    ToolMessage("No plan to delete.", tool_call_id=tool_call_id)
                ],
            }
        )

    # Remove plan from state
    return Command(
        update={
            "plan": None,
            "messages": [
                ToolMessage("Plan deleted successfully.", tool_call_id=tool_call_id)
            ],
        }
    )


# Export tools
PLAN_TOOLS = [create_plan, read_plan, update_plan, delete_plan]
