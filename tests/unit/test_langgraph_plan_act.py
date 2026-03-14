from pytest import MonkeyPatch

from agent.domain.orchestration import ExecutionPlan, PlanStep
from agent.orchestration import planning_service
from agent.orchestration.langgraph_plan_act import run_plan_act_graph


def _build_demo_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="回答用户问题",
        steps=[
            PlanStep(id="step_1", title="检索证据"),
            PlanStep(id="step_2", title="输出结论", depends_on=["step_1"]),
        ],
        tool_hints=["search_document"],
        done_when="输出带证据的最终回答",
    )


def test_run_plan_act_graph_retries_until_plan_is_available() -> None:
    call_count = {"count": 0}
    expected_plan = _build_demo_plan()

    def flaky_planner(_prompt: str) -> ExecutionPlan | None:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return None
        return expected_plan

    plan = run_plan_act_graph(
        prompt="请先检索再总结",
        planner=flaky_planner,
        max_attempts=2,
    )

    assert call_count["count"] == 2
    assert plan == expected_plan


def test_run_plan_act_graph_raises_when_attempts_exhausted() -> None:
    def always_fail_planner(_prompt: str) -> ExecutionPlan | None:
        return None

    try:
        _ = run_plan_act_graph(
            prompt="无法生成计划",
            planner=always_fail_planner,
            max_attempts=2,
        )
    except RuntimeError as exc:
        assert "max attempts" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when planner never returns a plan")


def test_build_execution_plan_direct_cutover_to_langgraph(monkeypatch: MonkeyPatch) -> None:
    expected_plan = _build_demo_plan()
    call_recorder = {"called": False, "prompt": ""}

    def fake_run_plan_act_graph(
        *,
        prompt: str,
        planner: object,
        max_attempts: int,
    ) -> ExecutionPlan:
        call_recorder["called"] = True
        call_recorder["prompt"] = prompt
        assert max_attempts >= 1
        assert callable(planner)
        return expected_plan

    monkeypatch.setattr(planning_service, "run_plan_act_graph", fake_run_plan_act_graph)

    plan = planning_service.build_execution_plan("请给出结论", llm=None)

    assert call_recorder["called"] is True
    assert call_recorder["prompt"] == "请给出结论"
    assert plan == expected_plan
