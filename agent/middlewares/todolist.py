"""Enhanced TodoList middleware with dependency management.

Based on LangChain's official TodoListMiddleware, with added support for
task dependencies, cycle detection, and topological sorting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    OmitFromInput,
)
from langchain.tools import InjectedToolCallId

from ..domain.todo_graph import TodoGraph


class Todo(TypedDict):
    """A single todo item with content, status, and optional dependencies."""

    id: str
    """Unique identifier for the todo item."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""

    depends_on: NotRequired[list[str]]
    """Optional list of todo IDs that this todo depends on."""


class PlanningState(AgentState):
    """State schema for the enhanced todo middleware."""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items with dependency tracking."""


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list with dependency tracking.

## When to Use
- Complex multi-step tasks (3+ steps)
- Tasks with dependencies between steps
- User explicitly requests todo list
- User provides multiple tasks

## When NOT to Use
- Single straightforward task
- Trivial tasks (<3 steps)
- Purely conversational requests

## Task Dependencies
- Use `depends_on` field to specify task dependencies
- System will detect circular dependencies
- Only tasks with satisfied dependencies are executable

## Task States
- pending: Not yet started
- in_progress: Currently working on
- completed: Finished successfully

## Best Practices
- Mark tasks in_progress BEFORE starting
- Mark completed IMMEDIATELY after finishing
- Update dependencies as you discover new requirements
- Remove irrelevant tasks from the list"""


WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool with dependency management.
Use this for complex objectives to track progress and manage task dependencies.

Key features:
- Specify task dependencies using `depends_on` field
- System detects circular dependencies automatically
- View executable tasks (dependencies satisfied)

Mark todos as completed immediately after finishing each step."""


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Create and manage a structured task list with dependencies."""
    # Validate no circular dependencies
    graph = TodoGraph(todos)
    if graph.has_cycle():
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: Circular dependency detected in todo list. "
                        "Please check the depends_on fields.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    # Get executable todos for user feedback
    executable = graph.get_executable_todos()
    executable_ids = [t["id"] for t in executable]

    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    f"Updated todo list. Executable tasks: {executable_ids}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


class EnhancedTodoListMiddleware(AgentMiddleware):
    """Enhanced TodoList middleware with dependency management."""

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize the middleware with optional custom prompts."""
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        @tool(description=self.tool_description)
        def write_todos_impl(
            todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command[Any]:
            """Create and manage a structured task list with dependencies."""
            graph = TodoGraph(todos)
            if graph.has_cycle():
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                "Error: Circular dependency detected. Check depends_on fields.",
                                tool_call_id=tool_call_id,
                            )
                        ]
                    }
                )

            executable = graph.get_executable_todos()
            executable_ids = [t["id"] for t in executable]

            return Command(
                update={
                    "todos": todos,
                    "messages": [
                        ToolMessage(
                            f"Updated todo list. Executable: {executable_ids}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        self.tools = [write_todos_impl]

    def before_model(
        self, state: PlanningState, runtime: Runtime, config: Any
    ) -> dict[str, Any] | None:
        """Inject system prompt before model invocation."""
        messages = state.get("messages", [])
        if not messages:
            return None

        # Inject system prompt if this is a user message
        last_msg = messages[-1]
        if hasattr(last_msg, "type") and last_msg.type == "human":
            from langchain_core.messages import SystemMessage

            return {"messages": messages[:-1] + [SystemMessage(self.system_prompt), last_msg]}

        return None


# Create default instance for backward compatibility
todolist_middleware = EnhancedTodoListMiddleware()

__all__ = ["EnhancedTodoListMiddleware", "todolist_middleware", "Todo", "PlanningState"]
