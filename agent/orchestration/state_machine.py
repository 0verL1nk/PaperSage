from __future__ import annotations

from dataclasses import replace

from agent.domain.orchestration import TeamRunState, TeamTodoRecord

_TODO_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"ready", "blocked", "failed", "canceled"},
    "ready": {"in_progress", "blocked", "canceled"},
    "in_progress": {"completed", "failed", "canceled"},
    "blocked": {"pending", "ready", "canceled"},
    "failed": {"pending", "ready", "canceled"},
    "completed": set(),
    "canceled": set(),
}

_RUN_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"scheduled", "failed", "canceled"},
    "scheduled": {"running", "failed", "canceled"},
    "running": {"reviewing", "replanning", "failed", "canceled"},
    "reviewing": {"completed", "replanning", "failed"},
    "replanning": {"scheduled", "failed", "canceled"},
    "completed": set(),
    "failed": set(),
    "canceled": set(),
}


def _ensure_transition(
    transitions: dict[str, set[str]],
    current_status: str,
    next_status: str,
    entity: str,
) -> None:
    if current_status == next_status:
        return
    allowed = transitions.get(current_status, set())
    if next_status in allowed:
        return
    raise ValueError(
        f"Invalid {entity} transition: {current_status!r} -> {next_status!r}"
    )


def transition_team_todo_record(
    todo: TeamTodoRecord,
    next_status: str,
    *,
    result: dict[str, object] | None = None,
    artifact_ref: str | None = None,
    error: str | None = None,
    retry_count: int | None = None,
) -> TeamTodoRecord:
    current_status = str(todo.status or "pending").strip()
    target_status = str(next_status or current_status).strip() or current_status
    _ensure_transition(_TODO_TRANSITIONS, current_status, target_status, "todo")
    return replace(
        todo,
        status=target_status,
        result=dict(result) if isinstance(result, dict) else todo.result,
        artifact_ref=todo.artifact_ref if artifact_ref is None else str(artifact_ref).strip(),
        error=todo.error if error is None else str(error).strip(),
        retry_count=todo.retry_count if retry_count is None else max(0, int(retry_count)),
    )


def transition_team_run_state(
    run_state: TeamRunState,
    next_state: str,
    *,
    review_decision: str | None = None,
    error: str | None = None,
) -> TeamRunState:
    current_state = str(run_state.state or "draft").strip()
    target_state = str(next_state or current_state).strip() or current_state
    _ensure_transition(_RUN_TRANSITIONS, current_state, target_state, "run")
    errors = list(run_state.errors)
    if error:
        errors.append(str(error).strip())
    return replace(
        run_state,
        state=target_state,
        review_decision=(
            run_state.review_decision
            if review_decision is None
            else str(review_decision).strip()
        ),
        errors=errors,
    )
