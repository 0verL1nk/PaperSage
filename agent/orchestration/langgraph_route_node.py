from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ..a2a.coordinator import WORKFLOW_PLAN_ACT, WORKFLOW_REACT, WORKFLOW_TEAM
from ..domain.orchestration import PolicyDecision

WorkflowMode = Literal["react", "plan_act", "team"]


class RouteNodeState(TypedDict):
    decision: PolicyDecision
    workflow_mode: WorkflowMode


def route_from_policy_decision(decision: PolicyDecision) -> WorkflowMode:
    if decision.team_enabled:
        return WORKFLOW_TEAM
    if decision.plan_enabled:
        return WORKFLOW_PLAN_ACT
    return WORKFLOW_REACT


def _resolve_route(state: RouteNodeState) -> dict[str, WorkflowMode]:
    decision = state["decision"]
    return {"workflow_mode": route_from_policy_decision(decision)}


def _route_next(state: RouteNodeState) -> WorkflowMode:
    return state["workflow_mode"]


@lru_cache(maxsize=1)
def _compiled_route_graph():
    graph = StateGraph(RouteNodeState)
    graph.add_node("resolve_route", _resolve_route)
    graph.add_edge(START, "resolve_route")
    graph.add_conditional_edges(
        "resolve_route",
        _route_next,
        {
            WORKFLOW_REACT: END,
            WORKFLOW_PLAN_ACT: END,
            WORKFLOW_TEAM: END,
        },
    )
    return graph.compile()


def run_policy_route_node(decision: PolicyDecision) -> WorkflowMode:
    graph = _compiled_route_graph()
    final_state = graph.invoke({"decision": decision, "workflow_mode": WORKFLOW_REACT})
    route = final_state.get("workflow_mode")
    if route in {WORKFLOW_REACT, WORKFLOW_PLAN_ACT, WORKFLOW_TEAM}:
        return route
    return route_from_policy_decision(decision)
