from agent.domain.orchestration import (
    RoleSpec,
    TeamPlan,
    TeamRunState,
    TeamTodoRecord,
)


def test_team_plan_to_dict_serializes_roles_and_todos() -> None:
    plan = TeamPlan(
        goal="完成项目级调研回答",
        roles=[
            RoleSpec(
                name="teammate",
                description="执行受限子任务",
                goal="完成单个调研任务",
                system_prompt="你是一个 teammate。",
                expected_output="结构化任务结果",
                allowed_tools=["search_document", "search_web"],
            )
        ],
        todos=[
            TeamTodoRecord(
                id="todo-1",
                content="检索相关文档",
                status="ready",
                assignee="teammate",
                execution_backend="local",
                depends_on=[],
                done_when="返回至少两条证据",
            )
        ],
        done_when="输出最终结论",
    )

    payload = plan.to_dict()

    assert payload["goal"] == "完成项目级调研回答"
    assert payload["roles"][0]["name"] == "teammate"
    assert payload["roles"][0]["allowed_tools"] == ["search_document", "search_web"]
    assert payload["todos"][0]["status"] == "ready"
    assert payload["todos"][0]["execution_backend"] == "local"


def test_team_run_state_to_dict_contains_plan_and_progress() -> None:
    plan = TeamPlan(goal="完成目标")
    state = TeamRunState(
        run_id="run-1",
        state="scheduled",
        plan=plan,
        completed_todo_ids=["todo-1"],
        review_decision="pass",
    )

    payload = state.to_dict()

    assert payload["run_id"] == "run-1"
    assert payload["state"] == "scheduled"
    assert payload["plan"]["goal"] == "完成目标"
    assert payload["completed_todo_ids"] == ["todo-1"]
    assert payload["review_decision"] == "pass"
