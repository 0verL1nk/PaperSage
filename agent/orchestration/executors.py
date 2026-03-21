from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.domain.orchestration import TeamTodoRecord


@dataclass(frozen=True)
class TaskExecutionResult:
    todo_id: str
    backend: str
    status: str
    output: str = ""
    error: str = ""
    artifact_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "todo_id": self.todo_id,
            "backend": self.backend,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "artifact_ref": self.artifact_ref,
            "metadata": dict(self.metadata),
        }


def normalize_task_execution_result(
    payload: TaskExecutionResult | dict[str, Any] | str | None,
    *,
    todo_id: str,
    backend: str,
) -> TaskExecutionResult:
    if isinstance(payload, TaskExecutionResult):
        return payload
    if isinstance(payload, str):
        return TaskExecutionResult(
            todo_id=todo_id,
            backend=backend,
            status="completed",
            output=payload,
        )
    if not isinstance(payload, dict):
        return TaskExecutionResult(
            todo_id=todo_id,
            backend=backend,
            status="failed",
            error="Invalid task execution result payload",
        )
    raw_metadata = payload.get("metadata")
    metadata: dict[str, Any] = {}
    if isinstance(raw_metadata, dict):
        metadata = {str(key): value for key, value in raw_metadata.items()}
    return TaskExecutionResult(
        todo_id=str(payload.get("todo_id") or todo_id).strip() or todo_id,
        backend=str(payload.get("backend") or backend).strip() or backend,
        status=str(payload.get("status") or "completed").strip() or "completed",
        output=str(payload.get("output") or "").strip(),
        error=str(payload.get("error") or "").strip(),
        artifact_ref=str(payload.get("artifact_ref") or "").strip(),
        metadata=metadata,
    )


class A2ATaskExecutor:
    """A thin A2A transport adapter with a normalized task-result contract."""

    def __init__(self, transport: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._transport = transport

    def execute(
        self,
        todo: TeamTodoRecord,
        *,
        run_id: str,
        teammate_name: str,
    ) -> TaskExecutionResult:
        payload = {
            "todo_id": todo.id,
            "run_id": run_id,
            "teammate_name": teammate_name,
            "content": todo.content,
            "assignee": todo.assignee,
            "execution_backend": todo.execution_backend,
            "retry_count": todo.retry_count,
            "depends_on": list(todo.depends_on),
        }
        try:
            response = self._transport(payload)
        except Exception as exc:
            return TaskExecutionResult(
                todo_id=todo.id,
                backend="a2a",
                status="failed",
                error=str(exc),
            )
        return normalize_task_execution_result(response, todo_id=todo.id, backend="a2a")
