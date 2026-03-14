# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from langgraph.graph import END

RetryRouteDecision = Literal["continue", "end"]
RetryRouteFn = Callable[[Mapping[str, object]], RetryRouteDecision]


def build_retry_route(
    *,
    is_success: Callable[[Mapping[str, object]], bool],
    attempt_key: str = "attempt",
    max_attempts_key: str = "max_attempts",
) -> RetryRouteFn:
    def _route(state: Mapping[str, object]) -> RetryRouteDecision:
        if is_success(state):
            return "end"
        attempt_raw = state.get(attempt_key, 0)
        max_attempts_raw = state.get(max_attempts_key, 1)
        attempt = int(attempt_raw) if isinstance(attempt_raw, (int, str)) else 0
        max_attempts = 1
        if isinstance(max_attempts_raw, (int, str)):
            max_attempts = max(1, int(max_attempts_raw))
        if attempt >= max_attempts:
            return "end"
        return "continue"

    return _route


def add_retry_loop_edge(
    graph: object,
    *,
    source: str,
    continue_to: str,
    route: RetryRouteFn,
) -> None:
    graph_obj = graph
    if not hasattr(graph_obj, "add_conditional_edges"):
        raise TypeError("graph must implement add_conditional_edges")
    graph_obj.add_conditional_edges(
        source,
        route,
        {
            "continue": continue_to,
            "end": END,
        },
    )
