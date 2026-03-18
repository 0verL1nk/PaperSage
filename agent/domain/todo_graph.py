"""Todo 依赖图和拓扑排序"""

from typing import Any


class TodoGraph:
    """Todo 依赖图"""

    def __init__(self, todos: list[dict[str, Any]]):
        self.todos = {t["id"]: t for t in todos}
        self.graph = self._build_graph()

    def _build_graph(self) -> dict[str, list[str]]:
        """构建依赖图：todo_id -> [依赖的 todo_id]"""
        graph = {}
        for todo_id, todo in self.todos.items():
            depends_on = todo.get("depends_on", [])
            if not isinstance(depends_on, list):
                depends_on = []
            graph[todo_id] = depends_on
        return graph

    def has_cycle(self) -> bool:
        """检测是否有依赖环"""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def topological_sort(self) -> list[str]:
        """拓扑排序：返回可执行的 todo 顺序"""
        in_degree = {node: 0 for node in self.graph}
        adj = {node: [] for node in self.graph}

        # 计算入度和构建邻接表
        for node, dependencies in self.graph.items():
            in_degree[node] = len(dependencies)
            for dep in dependencies:
                if dep in adj:
                    adj[dep].append(node)

        queue = [node for node in self.graph if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result if len(result) == len(self.graph) else []

    def get_executable_todos(self) -> list[dict[str, Any]]:
        """获取当前可执行的 todos（依赖已完成且自身未完成）"""
        executable = []
        for todo_id, todo in self.todos.items():
            if todo.get("status") == "completed":
                continue

            depends_on = self.graph.get(todo_id, [])
            all_deps_done = all(
                self.todos.get(dep_id, {}).get("status") == "completed" for dep_id in depends_on
            )

            if all_deps_done:
                executable.append(todo)

        return executable
