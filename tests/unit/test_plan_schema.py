from agent.domain.orchestration import ExecutionPlan, PlanStep


def test_execution_plan_to_dict_excludes_budget_and_keeps_step_schema() -> None:
    plan = ExecutionPlan(
        goal="回答问题",
        constraints=["只基于文档"],
        steps=[
            PlanStep(
                id="step_1",
                title="检索证据",
                description="先获取证据",
                depends_on=[],
                tool_hints=["search_document"],
                done_when="至少有一条直接证据",
                status="todo",
            )
        ],
        tool_hints=["search_document"],
        done_when="输出最终回答",
    )

    payload = plan.to_dict()

    assert "budget" not in payload
    assert payload["goal"] == "回答问题"
    assert payload["constraints"] == ["只基于文档"]
    assert payload["tool_hints"] == ["search_document"]
    assert payload["steps"][0] == {
        "id": "step_1",
        "title": "检索证据",
        "description": "先获取证据",
        "depends_on": [],
        "tool_hints": ["search_document"],
        "done_when": "至少有一条直接证据",
        "status": "todo",
    }
