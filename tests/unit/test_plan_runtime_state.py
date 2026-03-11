from agent.domain.orchestration import (
    ExecutionPlan,
    PlanStep,
    create_plan_runtime_state,
    evolve_plan_runtime_state,
    list_unready_step_ids,
    next_ready_plan_step,
    render_execution_plan,
)
from agent.orchestration.planning_service import build_execution_plan, revise_execution_plan


def test_render_execution_plan_uses_goal_and_numbered_steps() -> None:
    plan = ExecutionPlan(
        goal="回答问题",
        steps=[
            PlanStep(id="step_1", title="检索证据"),
            PlanStep(id="step_2", title="总结结论"),
        ],
    )

    assert render_execution_plan(plan) == "目标：回答问题\n1. 检索证据\n2. 总结结论"


def test_create_and_evolve_plan_runtime_state_tracks_progress() -> None:
    plan = ExecutionPlan(
        goal="回答问题",
        constraints=["只基于文档"],
        steps=[
            PlanStep(id="step_1", title="检索证据"),
            PlanStep(id="step_2", title="总结结论", depends_on=["step_1"]),
        ],
    )

    state = create_plan_runtime_state(
        user_goal="请总结",
        current_plan=plan,
        context_summary="session context",
    )

    assert state.current_step_id == "step_1"
    assert state.completed_step_ids == []
    assert state.constraints == ["只基于文档"]

    progressed = evolve_plan_runtime_state(
        state,
        completed_step_id="step_1",
        artifact={"type": "evidence", "content": "chunk-1"},
        budget_usage_delta={"planner_calls": 1},
    )

    assert progressed.current_step_id == "step_2"
    assert progressed.completed_step_ids == ["step_1"]
    assert progressed.artifacts == [{"type": "evidence", "content": "chunk-1"}]
    assert progressed.budget_usage == {"planner_calls": 1}


def test_build_execution_plan_without_llm_returns_structured_fallback() -> None:
    plan = build_execution_plan("请比较两种方法", llm=None)

    assert plan.goal
    assert len(plan.steps) >= 1
    assert all(step.id.startswith("step_") for step in plan.steps)
    assert any(step.tool_hints == ["search_document"] for step in plan.steps)


def test_revise_execution_plan_without_llm_returns_structured_fallback() -> None:
    current_plan = ExecutionPlan(
        goal="回答问题",
        steps=[
            PlanStep(
                id="step_1",
                title="检索证据",
                description="检索证据",
                tool_hints=["search_document"],
                done_when="需要文档证据",
            )
        ],
    )

    revised = revise_execution_plan(
        prompt="请回答",
        current_plan=current_plan,
        failed_step_id="step_1",
        failure_reason="missing_required_tool:search_document",
        llm=None,
    )

    assert revised.goal == "回答问题"
    assert revised.steps[0].title == "检索证据"
    assert revised.steps[0].tool_hints == ["search_document"]
    assert "失败原因" in revised.constraints[-1]


def test_runtime_state_selects_ready_step_when_dependencies_out_of_order() -> None:
    plan = ExecutionPlan(
        goal="回答问题",
        steps=[
            PlanStep(id="step_1", title="总结", depends_on=["step_2"]),
            PlanStep(id="step_2", title="检索证据"),
        ],
    )

    state = create_plan_runtime_state(user_goal="请总结", current_plan=plan)
    assert state.current_step_id == "step_2"
    assert next_ready_plan_step(plan, state.completed_step_ids) is not None

    progressed = evolve_plan_runtime_state(state, completed_step_id="step_2")
    assert progressed.current_step_id == "step_1"
    assert list_unready_step_ids(plan, progressed.completed_step_ids) == []


def test_runtime_state_marks_cycle_dependencies_as_unready() -> None:
    plan = ExecutionPlan(
        goal="回答问题",
        steps=[
            PlanStep(id="step_1", title="步骤1", depends_on=["step_2"]),
            PlanStep(id="step_2", title="步骤2", depends_on=["step_1"]),
        ],
    )

    state = create_plan_runtime_state(user_goal="请回答", current_plan=plan)
    assert state.current_step_id == "step_1"
    assert next_ready_plan_step(plan, state.completed_step_ids) is None
    assert list_unready_step_ids(plan, state.completed_step_ids) == ["step_1", "step_2"]
