# pyright: reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnusedCallResult=false
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import START, StateGraph

from agent.orchestration.retry_loop import add_retry_loop_edge, build_retry_route


class _RetryState(TypedDict):
    attempt: int
    max_attempts: int
    succeeded: bool


def _build_retry_graph(*, succeed_on_attempt: int | None) -> Any:
    def attempt_once(state: _RetryState) -> dict[str, object]:
        next_attempt = int(state.get("attempt", 0)) + 1
        succeeded = False
        if succeed_on_attempt is not None:
            succeeded = next_attempt >= succeed_on_attempt
        return {
            "attempt": next_attempt,
            "succeeded": succeeded,
        }

    graph = StateGraph(_RetryState)
    graph.add_node("attempt_once", attempt_once)
    graph.add_edge(START, "attempt_once")
    add_retry_loop_edge(
        graph,
        source="attempt_once",
        continue_to="attempt_once",
        route=build_retry_route(is_success=lambda state: bool(state.get("succeeded"))),
    )
    return graph.compile()


def test_retry_route_exits_on_success() -> None:
    graph = _build_retry_graph(succeed_on_attempt=2)

    result = graph.invoke({"attempt": 0, "max_attempts": 4, "succeeded": False})

    assert result["attempt"] == 2
    assert result["succeeded"] is True


def test_retry_route_exits_on_max_attempts() -> None:
    graph = _build_retry_graph(succeed_on_attempt=None)

    result = graph.invoke({"attempt": 0, "max_attempts": 3, "succeeded": False})

    assert result["attempt"] == 3
    assert result["succeeded"] is False
