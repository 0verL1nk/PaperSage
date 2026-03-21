from __future__ import annotations

from dataclasses import replace

from agent.domain.orchestration import TeamPlan, TeamRunState, TeamTodoRecord
from agent.middlewares.todolist import Todo
from agent.orchestration.executors import A2ATaskExecutor, TaskExecutionResult
from agent.orchestration.state_machine import (
    transition_team_run_state,
    transition_team_todo_record,
)
from agent.orchestration.todo_scheduler import LeaderTodoScheduler


def _team_todo_to_todo(todo: TeamTodoRecord) -> Todo:
    return Todo(
        id=todo.id,
        content=todo.content,
        status=todo.status,
        depends_on=list(todo.depends_on),
        assignee=todo.assignee,
        execution_backend=todo.execution_backend,
        retry_count=todo.retry_count,
        result=todo.result,
        artifact_ref=todo.artifact_ref,
        error=todo.error,
    )


def _todo_to_team_todo(todo: Todo, original: TeamTodoRecord) -> TeamTodoRecord:
    return replace(
        original,
        status=todo.status,
        depends_on=list(todo.depends_on or []),
        assignee=todo.assignee,
        execution_backend=todo.execution_backend,
        retry_count=todo.retry_count,
        result=todo.result,
        artifact_ref=todo.artifact_ref,
        error=todo.error,
    )


class LeaderTeammateCoordinator:
    """Apply reviewer decisions to a structured Leader-teammate run."""

    def __init__(self, a2a_executor: A2ATaskExecutor | None = None) -> None:
        self._scheduler = LeaderTodoScheduler()
        self._a2a_executor = a2a_executor

    def dispatch_ready_todos(
        self,
        run_state: TeamRunState,
        *,
        local_runtime,
    ) -> TeamRunState:
        if run_state.plan is None:
            raise ValueError("Team run state does not contain a plan")

        next_state = self.refresh_run_state(run_state)
        while True:
            ready_todos = self.select_ready_todos(next_state)
            if not ready_todos:
                break
            for ready_todo in ready_todos:
                next_state = self.mark_todo_in_progress(next_state, ready_todo.id)
                result = self._execute_single_todo(
                    ready_todo,
                    run_id=run_state.run_id,
                    local_runtime=local_runtime,
                )
                next_state = self.apply_task_result(next_state, result)
        return next_state

    def refresh_run_state(self, run_state: TeamRunState) -> TeamRunState:
        if run_state.plan is None:
            raise ValueError("Team run state does not contain a plan")
        refreshed_plan = self._refresh_plan(run_state.plan)
        completed_ids = [
            todo.id for todo in refreshed_plan.todos if todo.status == "completed"
        ]
        return replace(
            run_state,
            plan=refreshed_plan,
            completed_todo_ids=completed_ids,
        )

    def select_ready_todos(self, run_state: TeamRunState) -> list[TeamTodoRecord]:
        refreshed_state = self.refresh_run_state(run_state)
        if refreshed_state.plan is None:
            return []
        return [todo for todo in refreshed_state.plan.todos if todo.status == "ready"]

    def mark_todo_in_progress(self, run_state: TeamRunState, todo_id: str) -> TeamRunState:
        refreshed_state = self.refresh_run_state(run_state)
        if refreshed_state.plan is None:
            raise ValueError("Team run state does not contain a plan")
        updated_todos: list[TeamTodoRecord] = []
        target_id = str(todo_id).strip()
        for todo in refreshed_state.plan.todos:
            if todo.id == target_id:
                updated_todos.append(transition_team_todo_record(todo, "in_progress"))
            else:
                updated_todos.append(todo)
        next_state = replace(
            refreshed_state,
            plan=replace(refreshed_state.plan, todos=updated_todos),
        )
        if next_state.state == "scheduled":
            return transition_team_run_state(next_state, "running")
        return next_state

    def apply_task_result(
        self,
        run_state: TeamRunState,
        result: TaskExecutionResult,
    ) -> TeamRunState:
        refreshed_state = self.refresh_run_state(run_state)
        if refreshed_state.plan is None:
            raise ValueError("Team run state does not contain a plan")
        terminal_status = (
            result.status
            if result.status in {"completed", "failed", "canceled"}
            else "completed"
        )
        updated_todos: list[TeamTodoRecord] = []
        for todo in refreshed_state.plan.todos:
            if todo.id != result.todo_id:
                updated_todos.append(todo)
                continue
            if todo.status == "ready":
                todo = transition_team_todo_record(todo, "in_progress")
            updated_todos.append(
                transition_team_todo_record(
                    todo,
                    terminal_status,
                    result=result.to_dict(),
                    artifact_ref=result.artifact_ref,
                    error=result.error,
                )
            )
        next_state = replace(
            refreshed_state,
            plan=replace(refreshed_state.plan, todos=updated_todos),
        )
        next_state = self.refresh_run_state(next_state)
        if (
            next_state.plan is not None
            and next_state.plan.todos
            and all(todo.status == "completed" for todo in next_state.plan.todos)
            and next_state.state == "running"
        ):
            return transition_team_run_state(next_state, "reviewing")
        return next_state

    def apply_review_decision(
        self,
        run_state: TeamRunState,
        decision: str,
    ) -> TeamRunState:
        normalized = str(decision or "").strip().lower()
        tagged_state = replace(run_state, review_decision=normalized)
        if normalized in {"pass", "approved", "accept", "accepted"}:
            return transition_team_run_state(
                tagged_state,
                "completed",
                review_decision=normalized,
            )
        if normalized in {"revise", "revision", "replan", "retry"}:
            return transition_team_run_state(
                tagged_state,
                "replanning",
                review_decision=normalized,
            )
        return transition_team_run_state(
            tagged_state,
            "failed",
            review_decision=normalized,
            error=f"review_decision={normalized or 'unknown'}",
        )

    def _refresh_plan(self, plan: TeamPlan) -> TeamPlan:
        indexed_original = {todo.id: todo for todo in plan.todos}
        refreshed = self._scheduler.refresh_todo_states(
            [_team_todo_to_todo(todo) for todo in plan.todos]
        )
        return replace(
            plan,
            todos=[
                _todo_to_team_todo(todo, indexed_original[todo.id])
                for todo in refreshed
            ],
        )

    def _execute_single_todo(
        self,
        todo: TeamTodoRecord,
        *,
        run_id: str,
        local_runtime,
    ) -> TaskExecutionResult:
        if todo.execution_backend == "a2a":
            if self._a2a_executor is None:
                return TaskExecutionResult(
                    todo_id=todo.id,
                    backend="a2a",
                    status="failed",
                    error="A2A executor is not configured",
                )
            return self._a2a_executor.execute(
                todo,
                run_id=run_id,
                teammate_name=todo.assignee or "teammate",
            )
        return local_runtime.execute_todo(todo, message=todo.content)
