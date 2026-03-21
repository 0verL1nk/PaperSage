from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent.domain.orchestration import RoleSpec, TeamPlan, TeamRunState, TeamTodoRecord
from agent.middlewares.orchestration import OrchestrationMiddleware
from agent.middlewares.todolist import Todo
from agent.orchestration.coordinator import LeaderTeammateCoordinator
from agent.orchestration.executors import A2ATaskExecutor, TaskExecutionResult
from agent.orchestration.planner import build_leader_team_plan
from agent.orchestration.state_machine import (
    transition_team_run_state,
    transition_team_todo_record,
)
from agent.team.runtime import TeamRuntime


class _FakeRouterLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, _prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self._response)


def test_build_leader_team_plan_normalizes_roles_and_todos() -> None:
    plan = build_leader_team_plan(
        goal="完成多角色分析",
        roles=[
            RoleSpec(
                name="teammate",
                description="执行子任务",
                goal="完成分配任务",
            )
        ],
        todos=[
            Todo(id="1", content="检索资料", status="completed", depends_on=[]),
            Todo(id="2", content="整理证据", status="pending", depends_on=["1"]),
        ],
        done_when="给出最终结论",
    )

    assert isinstance(plan, TeamPlan)
    assert plan.roles[0].name == "teammate"
    assert plan.todos[1].status == "ready"
    assert plan.done_when == "给出最终结论"


def test_orchestration_middleware_emits_team_handoff_state() -> None:
    middleware = OrchestrationMiddleware(
        llm=_FakeRouterLLM('{"is_complex": true, "needs_team": true, "reason": "multi-role"}')
    )
    update = middleware.before_model(
        {"messages": [HumanMessage(content="请多角色协作完成分析")]},
        runtime=None,
        config={"configurable": {"state": {}}},
    )

    assert update is not None
    assert update["needs_team"] is True
    assert update["team_handoff"]["mode"] == "leader_teammate"
    assert update["team_handoff"]["reason"] == "multi-role"


def test_team_runtime_execute_todo_uses_normalized_execution_result() -> None:
    def _handler(todo: TeamTodoRecord, message: str) -> TaskExecutionResult:
        return TaskExecutionResult(
            todo_id=todo.id,
            backend="local",
            status="completed",
            output=f"done:{message}",
        )

    runtime = TeamRuntime("team-1", execution_handler=_handler)
    result = runtime.execute_todo(
        TeamTodoRecord(id="todo-1", content="执行子任务", assignee="teammate"),
        message="payload",
    )

    assert result.todo_id == "todo-1"
    assert result.backend == "local"
    assert result.status == "completed"
    assert result.output == "done:payload"
    runtime.cleanup()


def test_a2a_executor_preserves_task_metadata() -> None:
    captured: dict[str, object] = {}

    def _transport(payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {"status": "completed", "output": "remote-done"}

    executor = A2ATaskExecutor(transport=_transport)
    result = executor.execute(
        TeamTodoRecord(
            id="todo-a2a",
            content="调用远端 agent",
            assignee="teammate",
            execution_backend="a2a",
        ),
        run_id="run-1",
        teammate_name="teammate",
    )

    assert captured["todo_id"] == "todo-a2a"
    assert captured["run_id"] == "run-1"
    assert captured["teammate_name"] == "teammate"
    assert result.backend == "a2a"
    assert result.output == "remote-done"


def test_state_machine_and_coordinator_support_review_completion() -> None:
    todo = TeamTodoRecord(id="todo-1", content="执行任务", status="ready")
    running_todo = transition_team_todo_record(todo, "in_progress")
    completed_todo = transition_team_todo_record(running_todo, "completed")

    run_state = TeamRunState(run_id="run-1", state="draft", plan=TeamPlan(goal="goal"))
    run_state = transition_team_run_state(run_state, "scheduled")
    run_state = transition_team_run_state(run_state, "running")
    run_state = transition_team_run_state(run_state, "reviewing")

    coordinator = LeaderTeammateCoordinator()
    finished = coordinator.apply_review_decision(run_state, "pass")

    assert completed_todo.status == "completed"
    assert finished.state == "completed"


def test_coordinator_supports_leader_controlled_stepwise_dispatch() -> None:
    coordinator = LeaderTeammateCoordinator()
    run_state = TeamRunState(
        run_id="run-stepwise",
        plan=build_leader_team_plan(
            goal="完成分步协作",
            roles=[RoleSpec(name="teammate", description="执行任务", goal="完成 todo")],
            todos=[
                Todo(id="1", content="先做检索", status="pending", depends_on=[]),
                Todo(id="2", content="再做汇总", status="pending", depends_on=["1"]),
            ],
            done_when="产出最终结论",
        ),
    )
    run_state = transition_team_run_state(run_state, "scheduled")

    ready = coordinator.select_ready_todos(run_state)
    assert [todo.id for todo in ready] == ["1"]

    run_state = coordinator.mark_todo_in_progress(run_state, "1")
    run_state = coordinator.apply_task_result(
        run_state,
        TaskExecutionResult(
            todo_id="1",
            backend="local",
            status="completed",
            output="done-1",
        ),
    )

    ready = coordinator.select_ready_todos(run_state)
    assert [todo.id for todo in ready] == ["2"]
