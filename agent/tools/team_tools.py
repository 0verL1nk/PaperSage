"""Team mode activation tools for agent-centric orchestration.

These tools allow the leader agent to activate team collaboration mode
when facing complex tasks that benefit from multi-agent coordination.
"""

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ActivateTeamModeInput(BaseModel):
    """Input for activate_team_mode tool."""

    reason: str = Field(description="Why team collaboration is needed for this task")
    max_members: int = Field(default=3, description="Maximum number of team members (1-5)")
    max_rounds: int = Field(default=2, description="Maximum collaboration rounds (1-3)")


@tool("activate_team_mode", args_schema=ActivateTeamModeInput)
def activate_team_mode(
    reason: str,
    max_members: int = 3,
    max_rounds: int = 2,
    *,
    config: dict[str, Any] | None = None,
) -> str:
    """Activate team collaboration mode for complex tasks.

    Use this when a task requires:
    - Multiple specialized perspectives or expertise
    - Parallel work on independent subtasks
    - Complex problem-solving that benefits from collaboration

    Team mode will:
    - Generate specialized roles based on the task
    - Coordinate multi-agent collaboration
    - Aggregate results from team members

    Args:
        reason: Explanation of why team mode is needed
        max_members: Maximum team size (1-5, default 3)
        max_rounds: Maximum collaboration rounds (1-3, default 2)

    Returns:
        Confirmation message with team configuration
    """
    if config is None:
        return "Error: config not provided"

    # Validate parameters
    bounded_members = max(1, min(5, int(max_members)))
    bounded_rounds = max(1, min(3, int(max_rounds)))

    # Get agent state from config
    state = config.get("configurable", {}).get("state", {})

    # Store team mode activation in agent state
    team_config = {
        "enabled": True,
        "reason": reason,
        "max_members": bounded_members,
        "max_rounds": bounded_rounds,
    }
    state["team_mode"] = team_config

    return (
        f"Team mode activated.\n"
        f"Reason: {reason}\n"
        f"Max members: {bounded_members}\n"
        f"Max rounds: {bounded_rounds}\n\n"
        f"The orchestrator will coordinate team collaboration for this task."
    )


# Export tools
TEAM_TOOLS = [activate_team_mode]
