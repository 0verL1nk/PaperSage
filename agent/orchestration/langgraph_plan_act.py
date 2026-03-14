# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..domain.orchestration import ExecutionPlan, PlanStep
from .retry_loop import add_retry_loop_edge, build_retry_route

PlannerCallable = Callable[[str], ExecutionPlan | None]


class PlanActState(TypedDict):
    prompt: str
    attempt: int
    max_attempts: int
    plan: ExecutionPlan | None
    status: Literal["planning", "completed", "failed"]
    errors: list[str]
    messages: Annotated[list[AnyMessage], add_messages]


def _plan_to_payload(plan: ExecutionPlan | None) -> str:
    if plan is None:
        payload: dict[str, Any] = {"plan": None}
    else:
        payload = {"plan": plan.to_dict()}
    return json.dumps(payload, ensure_ascii=False)


def _plan_from_payload(payload: str) -> ExecutionPlan | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw_plan = data.get("plan")
    if not isinstance(raw_plan, dict):
        return None

    raw_steps = raw_plan.get("steps")
    if not isinstance(raw_steps, list):
        return None
    steps: list[PlanStep] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        steps.append(
            PlanStep(
                id=str(item.get("id") or f"step_{index}"),
                title=title,
                description=str(item.get("description") or ""),
                depends_on=[
                    str(dep).strip() for dep in item.get("depends_on", []) if str(dep).strip()
                ],
                tool_hints=[
                    str(hint).strip() for hint in item.get("tool_hints", []) if str(hint).strip()
                ],
                done_when=str(item.get("done_when") or ""),
                status=str(item.get("status") or "todo"),
            )
        )

    goal = str(raw_plan.get("goal") or "").strip()
    if not goal or not steps:
        return None
    constraints = [
        str(item).strip() for item in raw_plan.get("constraints", []) if str(item).strip()
    ]
    tool_hints = [str(item).strip() for item in raw_plan.get("tool_hints", []) if str(item).strip()]
    return ExecutionPlan(
        goal=goal,
        constraints=constraints,
        steps=steps,
        tool_hints=tool_hints,
        done_when=str(raw_plan.get("done_when") or "").strip(),
    )


def _prepare_plan_call(state: PlanActState) -> dict[str, Any]:
    next_attempt = int(state.get("attempt", 0)) + 1
    prompt = str(state.get("prompt") or "")
    tool_call = {
        "name": "build_execution_plan_tool",
        "args": {"prompt": prompt},
        "id": f"plan_attempt_{next_attempt}",
        "type": "tool_call",
    }
    message = AIMessage(content="", tool_calls=[tool_call])
    return {
        "attempt": next_attempt,
        "status": "planning",
        "messages": [message],
    }


def _consume_plan_result(state: PlanActState) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    existing_errors = list(state.get("errors") or [])
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "build_execution_plan_tool":
            parsed_plan = _plan_from_payload(str(message.content or ""))
            if parsed_plan is not None:
                return {"plan": parsed_plan, "status": "completed", "errors": existing_errors}
            break

    errors = [
        *existing_errors,
        f"planner returned invalid payload at attempt={state.get('attempt', 0)}",
    ]
    return {"plan": None, "status": "failed", "errors": errors}


def create_plan_act_graph(*, planner: PlannerCallable):
    @tool("build_execution_plan_tool")
    def build_execution_plan_tool(prompt: str) -> str:
        """Build execution plan payload for the current prompt."""

        plan = planner(prompt)
        return _plan_to_payload(plan)

    graph = StateGraph(PlanActState)
    graph.add_node("prepare_plan_call", _prepare_plan_call)
    graph.add_node("planner_tools", ToolNode([build_execution_plan_tool]))
    graph.add_node("consume_plan_result", _consume_plan_result)
    graph.add_edge(START, "prepare_plan_call")
    graph.add_edge("prepare_plan_call", "planner_tools")
    graph.add_edge("planner_tools", "consume_plan_result")
    add_retry_loop_edge(
        graph,
        source="consume_plan_result",
        continue_to="prepare_plan_call",
        route=build_retry_route(is_success=lambda state: state.get("plan") is not None),
    )
    return graph.compile()


def run_plan_act_graph(
    *,
    prompt: str,
    planner: PlannerCallable,
    max_attempts: int = 2,
) -> ExecutionPlan:
    graph = create_plan_act_graph(planner=planner)
    final_state = graph.invoke(
        {
            "prompt": prompt,
            "attempt": 0,
            "max_attempts": max(1, int(max_attempts)),
            "plan": None,
            "status": "planning",
            "errors": [],
            "messages": [],
        }
    )
    plan = final_state.get("plan")
    if isinstance(plan, ExecutionPlan):
        return plan
    errors = "; ".join(str(item) for item in final_state.get("errors", []))
    raise RuntimeError(f"plan-act graph ended without plan after max attempts: {errors}")
