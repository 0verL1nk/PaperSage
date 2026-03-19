from __future__ import annotations

from collections.abc import Sequence

from agent.middlewares.todolist import Todo


class LeaderTodoScheduler:
    """Deterministic ready-todo selector for Leader-teammate execution."""

    TERMINAL_STATUSES = {"completed", "failed", "canceled"}
    BLOCKING_DEPENDENCY_STATUSES = {"failed", "canceled"}

    def refresh_todo_states(self, todos: Sequence[Todo]) -> list[Todo]:
        indexed = {
            str(todo.id).strip(): todo
            for todo in todos
            if str(todo.id).strip()
        }
        refreshed: list[Todo] = []
        for todo in todos:
            status = str(todo.status or "pending").strip().lower()
            if status not in {"pending", "ready"}:
                refreshed.append(todo)
                continue
            dependencies = [str(item).strip() for item in (todo.depends_on or []) if str(item).strip()]
            if any(
                str(indexed.get(dep_id).status).strip().lower() in self.BLOCKING_DEPENDENCY_STATUSES
                for dep_id in dependencies
                if dep_id in indexed
            ):
                refreshed.append(todo.model_copy(update={"status": "blocked"}))
                continue
            if all(
                str(indexed.get(dep_id).status).strip().lower() == "completed"
                for dep_id in dependencies
                if dep_id in indexed
            ):
                refreshed.append(todo.model_copy(update={"status": "ready"}))
                continue
            refreshed.append(todo.model_copy(update={"status": "pending"}))
        return refreshed

    def select_ready_todos(self, todos: Sequence[Todo]) -> list[Todo]:
        indexed = {
            str(todo.id).strip(): todo
            for todo in self.refresh_todo_states(todos)
            if str(todo.id).strip()
        }
        ready: list[Todo] = []
        for todo in indexed.values():
            status = str(todo.status or "pending").strip().lower()
            if status in {"completed", "failed", "canceled", "blocked", "in_progress"}:
                continue
            dependencies = [str(item).strip() for item in (todo.depends_on or []) if str(item).strip()]
            if any(
                str(indexed.get(dep_id).status).strip().lower() in self.BLOCKING_DEPENDENCY_STATUSES
                for dep_id in dependencies
                if dep_id in indexed
            ):
                continue
            if all(
                str(indexed.get(dep_id).status).strip().lower() == "completed"
                for dep_id in dependencies
                if dep_id in indexed
            ):
                ready.append(todo)
        return ready
