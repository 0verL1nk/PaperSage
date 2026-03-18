"""Enhanced TodoList 中间件：支持依赖关系和拓扑排序"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from ..tools import todolist


class EnhancedTodoListMiddleware:
    """Enhanced TodoList 中间件"""

    def __call__(self, state: dict, config: RunnableConfig) -> dict:
        """中间件处理逻辑"""
        return state

    @property
    def tools(self) -> list[Any]:
        """返回 todolist 工具"""
        return [todolist.write_todos, todolist.update_todo, todolist.list_todos]
