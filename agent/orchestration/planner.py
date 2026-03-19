from __future__ import annotations

from collections.abc import Iterable, Sequence

from agent.domain.orchestration import RoleSpec, TeamPlan, TeamTodoRecord
from agent.middlewares.todolist import Todo
from agent.orchestration.todo_scheduler import LeaderTodoScheduler


def _normalize_role_specs(roles: Iterable[RoleSpec | dict[str, object]]) -> list[RoleSpec]:
    normalized: list[RoleSpec] = []
    for item in roles:
        if isinstance(item, RoleSpec):
            normalized.append(item)
            continue
        if not isinstance(item, dict):
            continue
        normalized.append(
            RoleSpec(
                name=str(item.get("name") or "").strip(),
                description=str(item.get("description") or "").strip(),
                goal=str(item.get("goal") or "").strip(),
                system_prompt=str(item.get("system_prompt") or "").strip(),
                expected_output=str(item.get("expected_output") or "").strip(),
                allowed_tools=list(item.get("allowed_tools") or []),
            )
        )
    return normalized


def _coerce_todo(item: Todo | TeamTodoRecord | dict[str, object]) -> Todo | None:
    if isinstance(item, Todo):
        return item
    if isinstance(item, TeamTodoRecord):
        return Todo(
            id=item.id,
            content=item.content,
            status=item.status,
            depends_on=list(item.depends_on),
            assignee=item.assignee,
            execution_backend=item.execution_backend,
            retry_count=item.retry_count,
            result=item.result,
            artifact_ref=item.artifact_ref,
            error=item.error,
        )
    if not isinstance(item, dict):
        return None
    return Todo(
        id=str(item.get("id") or "").strip(),
        content=str(item.get("content") or "").strip(),
        status=str(item.get("status") or "pending").strip() or "pending",
        depends_on=list(item.get("depends_on") or []),
        assignee=str(item.get("assignee") or "").strip(),
        execution_backend=(
            str(item.get("execution_backend") or "local").strip() or "local"
        ),
        retry_count=int(item.get("retry_count") or 0),
        result=dict(item.get("result")) if isinstance(item.get("result"), dict) else None,
        artifact_ref=str(item.get("artifact_ref") or "").strip(),
        error=str(item.get("error") or "").strip(),
    )


def _normalize_todos(
    todos: Sequence[Todo | TeamTodoRecord | dict[str, object]],
) -> list[TeamTodoRecord]:
    scheduler = LeaderTodoScheduler()
    normalized_input = [todo for item in todos if (todo := _coerce_todo(item)) is not None]
    refreshed = scheduler.refresh_todo_states(normalized_input)
    return [
        TeamTodoRecord(
            id=todo.id,
            content=todo.content,
            status=todo.status,
            depends_on=list(todo.depends_on or []),
            assignee=todo.assignee,
            execution_backend=todo.execution_backend,
            retry_count=todo.retry_count,
            result=todo.result,
            artifact_ref=todo.artifact_ref,
            error=todo.error,
        )
        for todo in refreshed
    ]


def build_leader_team_plan(
    *,
    goal: str,
    roles: Sequence[RoleSpec | dict[str, object]] | None = None,
    todos: Sequence[Todo | TeamTodoRecord | dict[str, object]] | None = None,
    constraints: Sequence[str] | None = None,
    done_when: str = "",
) -> TeamPlan:
    """Build a structured team plan from team-mode activation inputs."""
    return TeamPlan(
        goal=str(goal or "").strip(),
        roles=_normalize_role_specs(roles or []),
        todos=_normalize_todos(todos or []),
        constraints=[str(item).strip() for item in (constraints or []) if str(item).strip()],
        done_when=str(done_when or "").strip(),
    )
