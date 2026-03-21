from agent.domain.orchestration import RoleSpec, TeamRunState
from agent.middlewares.todolist import Todo
from agent.orchestration.coordinator import LeaderTeammateCoordinator
from agent.orchestration.executors import A2ATaskExecutor, TaskExecutionResult
from agent.orchestration.planner import build_leader_team_plan
from agent.orchestration.state_machine import transition_team_run_state
from agent.team.runtime import TeamRuntime


def test_leader_teammate_execution_dispatches_local_then_a2a_backend() -> None:
    local_calls: list[str] = []
    remote_calls: list[str] = []

    def _local_handler(todo, message: str) -> TaskExecutionResult:
        local_calls.append(f"{todo.id}:{message}")
        return TaskExecutionResult(
            todo_id=todo.id,
            backend="local",
            status="completed",
            output=f"local:{message}",
        )

    def _remote_transport(payload: dict[str, object]) -> dict[str, object]:
        remote_calls.append(str(payload["todo_id"]))
        return {
            "status": "completed",
            "output": f"remote:{payload['todo_id']}",
        }

    plan = build_leader_team_plan(
        goal="完成调研与复核",
        roles=[
            RoleSpec(name="teammate", description="执行任务", goal="完成 todo"),
            RoleSpec(name="reviewer", description="复核结果", goal="给出审查结论"),
        ],
        todos=[
            Todo(
                id="1",
                content="本地整理证据",
                status="pending",
                assignee="teammate",
                execution_backend="local",
                depends_on=[],
            ),
            Todo(
                id="2",
                content="远端复核",
                status="pending",
                assignee="reviewer",
                execution_backend="a2a",
                depends_on=["1"],
            ),
        ],
        done_when="形成最终结论",
    )

    run_state = TeamRunState(run_id="run-1", plan=plan)
    run_state = transition_team_run_state(run_state, "scheduled")

    runtime = TeamRuntime("team-1", execution_handler=_local_handler)
    coordinator = LeaderTeammateCoordinator(a2a_executor=A2ATaskExecutor(_remote_transport))
    updated = coordinator.dispatch_ready_todos(run_state, local_runtime=runtime)

    assert updated.state == "reviewing"
    assert updated.completed_todo_ids == ["1", "2"]
    assert local_calls == ["1:本地整理证据"]
    assert remote_calls == ["2"]
    assert updated.plan is not None
    assert updated.plan.todos[0].result is not None
    assert updated.plan.todos[0].result["backend"] == "local"
    assert updated.plan.todos[1].result is not None
    assert updated.plan.todos[1].result["backend"] == "a2a"

    finalized = coordinator.apply_review_decision(updated, "pass")
    assert finalized.state == "completed"

    runtime.cleanup()
