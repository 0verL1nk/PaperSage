"""Performance benchmark tests for LangGraph migration.

Offline benchmarks are deterministic and CI-safe.
Live benchmark is opt-in via environment variables.
"""

from __future__ import annotations

import os
import time
from typing import Protocol, runtime_checkable

import pytest

from agent.domain.orchestration import ExecutionPlan, PlanStep
from agent.orchestration.checkpointer import create_checkpointer
from agent.orchestration.langgraph_plan_act import create_plan_act_graph, run_plan_act_graph
from agent.orchestration.planning_service import build_execution_plan

CHECKPOINTER_ITERATIONS = 500
CHECKPOINTER_MAX_SECONDS = 2.0

GRAPH_BUILD_ITERATIONS = 30
GRAPH_BUILD_MAX_SECONDS = 6.0

GRAPH_EXECUTION_ITERATIONS = 20
GRAPH_EXECUTION_MAX_SECONDS = 8.0

LIVE_BENCHMARK_ENV = "BENCHMARK_LIVE"
LIVE_MODEL_ENV = "BENCHMARK_LIVE_MODEL"
DEFAULT_LIVE_MODEL = "gpt-4o-mini"


@runtime_checkable
class _TerminalReporter(Protocol):
    def write_line(self, line: str, **markup: object) -> None: ...


def _build_demo_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="基于证据回答用户问题",
        steps=[
            PlanStep(id="step_1", title="检索证据"),
            PlanStep(id="step_2", title="整理结论", depends_on=["step_1"]),
        ],
        tool_hints=["search_document"],
        done_when="输出带证据依据的结论",
    )


def _emit_metric(request: pytest.FixtureRequest, metric: str, seconds: float) -> None:
    reporter: object | None = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None or not isinstance(reporter, _TerminalReporter):
        return
    reporter.write_line(f"[benchmark] {metric}={seconds:.6f}s")


def test_benchmark_checkpointer_creation_offline(request: pytest.FixtureRequest) -> None:
    start = time.perf_counter()
    for _ in range(CHECKPOINTER_ITERATIONS):
        _ = create_checkpointer("memory")
    elapsed = time.perf_counter() - start
    _emit_metric(request, "checkpointer_creation_total", elapsed)
    assert elapsed < CHECKPOINTER_MAX_SECONDS


def test_benchmark_plan_act_graph_build_offline(request: pytest.FixtureRequest) -> None:
    expected_plan = _build_demo_plan()

    def planner(_prompt: str) -> ExecutionPlan | None:
        return expected_plan

    start = time.perf_counter()
    for _ in range(GRAPH_BUILD_ITERATIONS):
        _ = create_plan_act_graph(planner=planner)
    elapsed = time.perf_counter() - start
    _emit_metric(request, "plan_act_graph_build_total", elapsed)
    assert elapsed < GRAPH_BUILD_MAX_SECONDS


def test_benchmark_plan_act_graph_execution_offline(request: pytest.FixtureRequest) -> None:
    expected_plan = _build_demo_plan()

    def planner(_prompt: str) -> ExecutionPlan | None:
        return expected_plan

    start = time.perf_counter()
    for _ in range(GRAPH_EXECUTION_ITERATIONS):
        plan = run_plan_act_graph(prompt="请总结核心贡献", planner=planner, max_attempts=1)
        assert plan.goal == expected_plan.goal
    elapsed = time.perf_counter() - start
    _emit_metric(request, "plan_act_graph_execution_total", elapsed)
    assert elapsed < GRAPH_EXECUTION_MAX_SECONDS


@pytest.mark.skipif(
    os.getenv(LIVE_BENCHMARK_ENV) != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Live benchmark requires BENCHMARK_LIVE=1 and OPENAI_API_KEY",
)
def test_benchmark_plan_act_live_execution(request: pytest.FixtureRequest) -> None:
    from langchain_openai import ChatOpenAI

    model = os.getenv(LIVE_MODEL_ENV, DEFAULT_LIVE_MODEL)
    llm = ChatOpenAI(model=model)
    start = time.perf_counter()
    plan = build_execution_plan("请给出论文分析步骤", llm=llm)
    elapsed = time.perf_counter() - start
    _emit_metric(request, "plan_act_live_execution_total", elapsed)
    assert len(plan.steps) >= 1
