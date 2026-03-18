"""TodoList 工具：支持依赖关系的任务管理"""

import json
import logging
from typing import Any

from langchain_core.tools import tool

from ..domain.todo_graph import TodoGraph

logger = logging.getLogger(__name__)

# 全局 todos 存储
_todos: list[dict[str, Any]] = []


@tool
def write_todos(todos_json: str) -> str:
    """写入 todos 列表（支持依赖关系）

    Args:
        todos_json: JSON 格式的 todos 列表,每个 todo 包含:
            - id: todo ID
            - content: 任务内容
            - status: 状态 (pending/completed)
            - depends_on: 依赖的 todo IDs (可选)
            - assigned_to: 分配给谁 (可选)

    Returns:
        状态信息
    """
    global _todos
    try:
        todos = json.loads(todos_json)
        _todos = todos

        # 检测依赖环
        graph = TodoGraph(todos)
        if graph.has_cycle():
            return "Error: 检测到依赖环,请检查 depends_on 字段"

        return f"已写入 {len(todos)} 个 todos"
    except json.JSONDecodeError as e:
        return f"Error: JSON 解析失败 - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def update_todo(todo_id: str, status: str) -> str:
    """更新 todo 状态

    Args:
        todo_id: todo ID
        status: 新状态 (pending/completed)

    Returns:
        状态信息
    """
    global _todos
    for todo in _todos:
        if todo["id"] == todo_id:
            todo["status"] = status
            return f"Todo {todo_id} 状态已更新为 {status}"
    return f"Error: Todo {todo_id} 不存在"


@tool
def list_todos() -> str:
    """列出所有 todos 及依赖关系

    Returns:
        todos 列表（包含可执行的 todos）
    """
    global _todos
    if not _todos:
        return "暂无 todos"

    graph = TodoGraph(_todos)
    executable = graph.get_executable_todos()

    result = "## 所有 Todos\n\n"
    for todo in _todos:
        status_icon = "✓" if todo.get("status") == "completed" else "○"
        result += f"{status_icon} [{todo['id']}] {todo['content']}\n"
        if todo.get("depends_on"):
            result += f"  依赖: {', '.join(todo['depends_on'])}\n"
        if todo.get("assigned_to"):
            result += f"  分配给: {todo['assigned_to']}\n"

    result += f"\n## 当前可执行 ({len(executable)} 个)\n\n"
    for todo in executable:
        result += f"- [{todo['id']}] {todo['content']}\n"

    return result
