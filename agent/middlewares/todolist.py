"""TodoList middleware integration for task tracking.

This module integrates LangChain's TodoListMiddleware to provide
todolist management capabilities for the agent.
"""

from langchain.agents.middleware import TodoListMiddleware

# Create TodoListMiddleware instance with default configuration
# The middleware automatically injects write_todos tool and manages
# todolist state in PlanningState, persisted via checkpointer
todolist_middleware = TodoListMiddleware()

__all__ = ["todolist_middleware"]
