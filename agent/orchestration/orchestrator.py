from collections.abc import Callable
from typing import Any
import logging

from ..domain.orchestration import (
    OrchestratedTurn,
    TeamExecution,
    TraceContext,
    TraceEvent,
    TracePayload,
    build_trace_event,
    create_trace_context,
)
from ..output_cleaner import sanitize_public_answer
from ..settings import load_agent_settings
from ..stream import extract_result_text, extract_tool_names_from_result
from .planning_service import build_execution_plan
from .policy_engine import decide_execution_policy
from .team_runtime import run_team_tasks

logger = logging.getLogger(__name__)

def _default_search_document_fn(
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None,
) -> Callable[[str], str]:
    if not callable(search_document_evidence_fn):
        return lambda _query: ""

    def _search(query: str) -> str:
        payload = search_document_evidence_fn(query)
        evidences = payload.get("evidences") if isinstance(payload, dict) else None
        if not isinstance(evidences, list):
            return ""
        chunks: list[str] = []
        for item in evidences:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)

    return _search


def _emit_event(
    *,
    trace_payload: TracePayload,
    trace_context: TraceContext,
    sender: str,
    receiver: str,
    performative: str,
    content: str,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> TraceEvent:
    event = build_trace_event(
        context=trace_context,
        sender=sender,
        receiver=receiver,
        performative=performative,
        content=content,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )
    trace_payload.append(event)
    if on_event is not None:
        on_event(event)
    return event


def _record_existing_event(
    trace_payload: TracePayload,
    event: TraceEvent,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> None:
    trace_payload.append(event)
    if on_event is not None:
        on_event(event)


def _llm_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def execute_orchestrated_turn(
    *,
    prompt: str,
    hinted_prompt: str,
    leader_agent: Any,
    leader_runtime_config: dict[str, Any] | None,
    llm: Any | None = None,
    search_document_fn: Callable[[str], str] | None = None,
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    routing_context: str = "",
    max_team_members: int | None = None,
    max_team_rounds: int | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> OrchestratedTurn:
    if search_document_fn is None:
        search_document_fn = _default_search_document_fn(search_document_evidence_fn)
    settings = load_agent_settings()
    resolved_team_members = (
        settings.agent_team_max_members
        if max_team_members is None
        else int(max_team_members)
    )
    resolved_team_rounds = (
        settings.agent_team_max_rounds
        if max_team_rounds is None
        else int(max_team_rounds)
    )
    resolved_team_members = max(
        1,
        min(resolved_team_members, max(1, settings.agent_team_members_hard_cap)),
    )
    resolved_team_rounds = max(
        1,
        min(resolved_team_rounds, max(1, settings.agent_team_rounds_hard_cap)),
    )

    trace_payload: TracePayload = []
    trace_context = create_trace_context(channel="internal.orchestrator")
    root_event = _emit_event(
        trace_payload=trace_payload,
        trace_context=trace_context,
        sender="user",
        receiver="leader",
        performative="request",
        content=prompt,
        on_event=on_event,
    )
    current_parent_span = str(root_event.get("span_id") or "")

    policy_decision = decide_execution_policy(
        prompt,
        llm=llm,
        force_plan=force_plan,
        force_team=force_team,
        context_digest=routing_context,
    )
    policy_event = _emit_event(
        trace_payload=trace_payload,
        trace_context=trace_context,
        sender="policy_engine",
        receiver="leader",
        performative="policy",
        content=(
            f"plan={policy_decision.plan_enabled}, "
            f"team={policy_decision.team_enabled}, "
            f"reason={policy_decision.reason}"
        ),
        parent_span_id=current_parent_span,
        on_event=on_event,
    )
    current_parent_span = str(policy_event.get("span_id") or current_parent_span)

    plan_text = ""
    if policy_decision.plan_enabled:
        plan_text = build_execution_plan(prompt, llm=llm)
        plan_event = _emit_event(
            trace_payload=trace_payload,
            trace_context=trace_context,
            sender="planner",
            receiver="leader",
            performative="plan",
            content=plan_text,
            parent_span_id=current_parent_span,
            on_event=on_event,
        )
        current_parent_span = str(plan_event.get("span_id") or current_parent_span)

    team_execution = TeamExecution(enabled=False)
    if policy_decision.team_enabled:
        team_execution = run_team_tasks(
            prompt=prompt,
            plan_text=plan_text,
            llm=llm,
            search_document_fn=search_document_fn,
            search_document_evidence_fn=search_document_evidence_fn,
            max_members=resolved_team_members,
            max_rounds=resolved_team_rounds,
            trace_context=trace_context,
            parent_span_id=current_parent_span,
            on_event=lambda event: _record_existing_event(trace_payload, event, on_event),
        )
        if team_execution.trace_events:
            current_parent_span = str(
                team_execution.trace_events[-1].get("span_id") or current_parent_span
            )
        if team_execution.fallback_reason:
            _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="team_runtime",
                receiver="leader",
                performative="fallback",
                content=team_execution.fallback_reason,
                parent_span_id=current_parent_span,
                on_event=on_event,
            )

    leader_prompt_parts = [hinted_prompt]
    if plan_text:
        leader_prompt_parts.append(f"[执行计划]\n{plan_text}")
    if team_execution.summary:
        leader_prompt_parts.append(f"[团队结果]\n{team_execution.summary}")
    leader_prompt = "\n\n".join(part for part in leader_prompt_parts if part.strip())

    result = leader_agent.invoke(
        {"messages": [{"role": "user", "content": leader_prompt}]},
        config=leader_runtime_config or {},
    )
    leader_tool_names: list[str] = []
    if isinstance(result, dict):
        leader_tool_names = sorted(extract_tool_names_from_result(result))
        answer = extract_result_text(result)
    else:
        answer = _llm_content_to_text(getattr(result, "content", result))
    answer = sanitize_public_answer(answer)
    if not answer:
        answer = "抱歉，我暂时没有生成有效回复。"

    _emit_event(
        trace_payload=trace_payload,
        trace_context=trace_context,
        sender="leader",
        receiver="user",
        performative="final",
        content=answer,
        parent_span_id=current_parent_span,
        on_event=on_event,
    )

    replan_rounds = sum(
        1 for e in trace_payload
        if isinstance(e, dict) and e.get("performative") == "replan"
    )
    _log_routing_decision(prompt, policy_decision, team_execution, replan_rounds)

    return OrchestratedTurn(
        answer=answer,
        policy_decision=policy_decision,
        team_execution=team_execution,
        trace_payload=trace_payload,
        plan_text=plan_text,
        leader_tool_names=leader_tool_names,
    )


def _log_routing_decision(
    prompt: str,
    policy_decision: Any,
    team_execution: Any,
    replan_rounds: int,
) -> None:
    """结构化记录路由决策，供离线分析与 prompt 优化使用。"""
    confidence = policy_decision.confidence
    logger.info(
        "routing_decision | prompt_len=%d | plan=%s | team=%s | source=%s"
        " | confidence=%s | replan_rounds=%d | team_rounds=%d | reason=%s",
        len(prompt),
        policy_decision.plan_enabled,
        policy_decision.team_enabled,
        policy_decision.source,
        f"{confidence:.2f}" if isinstance(confidence, float) else "n/a",
        replan_rounds,
        getattr(team_execution, "rounds", 0),
        policy_decision.reason,
    )
