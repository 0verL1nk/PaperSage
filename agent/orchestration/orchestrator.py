import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..domain.orchestration import (
    OrchestratedTurn,
    PolicyDecision,
    TeamExecution,
    TraceContext,
    TraceEvent,
    TracePayload,
    build_trace_event,
    create_trace_context,
)
from ..domain.request_context import RequestContext
from ..output_cleaner import sanitize_public_answer
from ..settings import load_agent_settings
from ..stream import (
    extract_ask_human_requests_from_result,
    extract_result_text,
    extract_skill_activation_events_from_result,
    extract_tool_activation_events_from_result,
    extract_tool_names_from_result,
    extract_tool_trace_events_from_result,
)
from .async_policy import AsyncPolicyPredictor
from .planning_service import build_execution_plan
from .policy_engine import intercept as intercept_policy
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


def _resolve_team_todo_store_path() -> Path:
    root_value = str(os.getenv("AGENT_FILE_TOOLS_ROOT", "") or "").strip()
    file_value = str(os.getenv("AGENT_TODO_FILE", "") or "").strip()
    root = Path(root_value) if root_value else Path.cwd()
    configured = Path(file_value) if file_value else Path(".agent/todo.json")
    candidate = configured if configured.is_absolute() else (root / configured)
    try:
        resolved_root = root.resolve()
    except Exception:
        resolved_root = root.absolute()
    try:
        resolved_candidate = candidate.resolve()
    except Exception:
        resolved_candidate = candidate.absolute()
    try:
        resolved_candidate.relative_to(resolved_root)
    except Exception:
        fallback = resolved_root / ".agent" / "todo.json"
        return fallback
    return resolved_candidate


def _persist_team_todo_records(team_execution: TeamExecution) -> str:
    todo_records = getattr(team_execution, "todo_records", None)
    if not isinstance(todo_records, list) or not todo_records:
        return ""
    path = _resolve_team_todo_store_path()
    existing: list[dict[str, Any]] = []
    if path.exists() and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing = [item for item in payload if isinstance(item, dict)]
        except Exception:
            existing = []
    id_to_index: dict[str, int] = {}
    for idx, item in enumerate(existing):
        todo_id = str(item.get("id") or "").strip()
        if todo_id:
            id_to_index[todo_id] = idx
    for raw in todo_records:
        if not isinstance(raw, dict):
            continue
        todo_id = str(raw.get("id") or "").strip()
        if not todo_id:
            continue
        record = dict(raw)
        index = id_to_index.get(todo_id)
        if index is None:
            id_to_index[todo_id] = len(existing)
            existing.append(record)
        else:
            existing[index] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _build_team_todo_summary(team_execution: TeamExecution) -> str:
    records = list(getattr(team_execution, "todo_records", []) or [])
    if not records:
        return ""
    stats = dict(getattr(team_execution, "todo_stats", {}) or {})
    stats_text = (
        f"done={int(stats.get('done', 0))}, "
        f"in_progress={int(stats.get('in_progress', 0))}, "
        f"todo={int(stats.get('todo', 0))}, "
        f"blocked={int(stats.get('blocked', 0))}"
    )
    lines = [f"任务统计: {stats_text}"]
    preview_items = records[:6]
    for item in preview_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("id") or "task").strip()
        status = str(item.get("status") or "todo").strip().lower()
        assignee = str(item.get("assignee") or "").strip()
        deps = item.get("dependencies")
        dep_count = len(deps) if isinstance(deps, list) else 0
        lines.append(
            f"- {title} | status={status} | assignee={assignee or 'n/a'} | deps={dep_count}"
        )
    return "\n".join(lines)


def _decision_mode(decision: PolicyDecision) -> tuple[bool, bool]:
    return bool(decision.plan_enabled), bool(decision.team_enabled)


def _decision_changed(current: PolicyDecision, candidate: PolicyDecision) -> bool:
    return _decision_mode(current) != _decision_mode(candidate)


def _async_decision_eligible(
    decision: PolicyDecision,
    *,
    min_confidence: float,
) -> bool:
    source = str(decision.source or "").strip().lower()
    if source == "manual":
        return True
    if source == "heuristic":
        return False
    confidence = decision.confidence
    if not isinstance(confidence, (int, float)):
        return False
    return float(confidence) >= float(min_confidence)


def _build_team_runtime_context(
    todo_records: list[dict[str, Any]],
    *,
    preview_limit: int = 6,
) -> str:
    if not isinstance(todo_records, list) or not todo_records:
        return ""
    lines: list[str] = []
    for item in todo_records[: max(1, int(preview_limit))]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("assignee") or "n/a").strip()
        status = str(item.get("status") or "todo").strip().lower()
        title = str(item.get("title") or item.get("id") or "task").strip()
        output = str(item.get("output") or "").strip().replace("\n", " ")
        if len(output) > 120:
            output = f"{output[:120]}..."
        line = f"- {title} | role={role} | status={status}"
        if output:
            line = f"{line} | output={output}"
        lines.append(line)
    return "\n".join(lines)


def execute_orchestrated_turn(
    *,
    prompt: str,
    hinted_prompt: str,
    leader_agent: Any,
    leader_runtime_config: dict[str, Any] | None,
    llm: Any | None = None,
    policy_llm: Any | None = None,
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
    policy_router_llm = policy_llm if policy_llm is not None else llm
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
    async_predictor: AsyncPolicyPredictor | None = None

    try:
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

        if settings.agent_policy_async_enabled and policy_router_llm is not None:
            async_predictor = AsyncPolicyPredictor(
                prompt=prompt,
                llm=policy_router_llm,
                context_digest=routing_context,
                force_plan=force_plan,
                force_team=force_team,
                refresh_interval_sec=settings.agent_policy_async_refresh_seconds,
            )
            async_predictor.start()

        def _maybe_apply_async_policy(
            stage: str,
            current_decision: PolicyDecision,
            *,
            context_digest: str,
        ) -> PolicyDecision:
            nonlocal current_parent_span
            if async_predictor is None:
                return current_decision
            async_predictor.update_context_digest(context_digest)
            snapshot = async_predictor.latest_snapshot(
                max_staleness_seconds=settings.agent_policy_async_max_staleness_seconds,
            )
            if snapshot is None:
                return current_decision
            candidate = snapshot.decision
            if not _async_decision_eligible(
                candidate,
                min_confidence=settings.agent_policy_async_min_confidence,
            ):
                return current_decision
            if not _decision_changed(current_decision, candidate):
                return current_decision
            switch_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="policy_engine",
                receiver="leader",
                performative="replan",
                content=(
                    f"async_switch@{stage}: plan={candidate.plan_enabled}, "
                    f"team={candidate.team_enabled}, reason={candidate.reason}"
                ),
                parent_span_id=current_parent_span,
                metadata={
                    "stage": stage,
                    "version": snapshot.version,
                    "source": candidate.source,
                    "confidence": candidate.confidence,
                },
                on_event=on_event,
            )
            current_parent_span = str(switch_event.get("span_id") or current_parent_span)
            return candidate

        policy_decision = intercept_policy(
            RequestContext(
                prompt=prompt,
                context_digest=routing_context,
            ),
            llm=policy_router_llm,
            force_plan=force_plan,
            force_team=force_team,
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
        policy_decision = _maybe_apply_async_policy(
            "pre_plan",
            policy_decision,
            context_digest=routing_context,
        )

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

        policy_context = routing_context
        if plan_text:
            policy_context = (
                f"{routing_context}\n\n[计划快照]\n"
                f"{plan_text[:1200]}"
            )
        policy_decision = _maybe_apply_async_policy(
            "post_plan",
            policy_decision,
            context_digest=policy_context,
        )
        if policy_decision.plan_enabled and not plan_text:
            plan_text = build_execution_plan(prompt, llm=llm)
            plan_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="planner",
                receiver="leader",
                performative="plan",
                content=plan_text,
                parent_span_id=current_parent_span,
                metadata={"stage": "late_plan"},
                on_event=on_event,
            )
            current_parent_span = str(plan_event.get("span_id") or current_parent_span)
            policy_context = (
                f"{routing_context}\n\n[计划快照]\n"
                f"{plan_text[:1200]}"
            )

        team_execution = TeamExecution(enabled=False)

        def _team_policy_checkpoint(todo_records: list[dict[str, Any]]) -> tuple[bool, str | None]:
            nonlocal policy_decision
            if async_predictor is None:
                return True, None
            runtime_context = policy_context
            team_runtime_context = _build_team_runtime_context(todo_records)
            if team_runtime_context:
                runtime_context = (
                    f"{policy_context}\n\n[团队执行快照]\n"
                    f"{team_runtime_context}"
                )
            refreshed = _maybe_apply_async_policy(
                "team_loop",
                policy_decision,
                context_digest=runtime_context,
            )
            if _decision_changed(policy_decision, refreshed):
                policy_decision = refreshed
            if not policy_decision.team_enabled:
                return False, "policy_switched_to_non_team"
            return True, None

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
                policy_checkpoint_fn=_team_policy_checkpoint,
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
            try:
                synced_todo_path = _persist_team_todo_records(team_execution)
                if synced_todo_path:
                    logger.info("team todo synced: %s", synced_todo_path)
            except Exception as exc:
                logger.warning("team todo sync failed: %s", exc)

        pre_final_context = policy_context
        if team_execution.summary:
            pre_final_context = (
                f"{pre_final_context}\n\n[团队摘要]\n"
                f"{team_execution.summary[:1200]}"
            )
        policy_decision = _maybe_apply_async_policy(
            "pre_final",
            policy_decision,
            context_digest=pre_final_context,
        )

        if policy_decision.team_enabled and not team_execution.enabled:
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
                policy_checkpoint_fn=_team_policy_checkpoint,
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
            try:
                synced_todo_path = _persist_team_todo_records(team_execution)
                if synced_todo_path:
                    logger.info("team todo synced: %s", synced_todo_path)
            except Exception as exc:
                logger.warning("team todo sync failed: %s", exc)

        leader_prompt_parts = [hinted_prompt]
        if plan_text:
            leader_prompt_parts.append(f"[执行计划]\n{plan_text}")
        if team_execution.summary:
            leader_prompt_parts.append(f"[团队结果]\n{team_execution.summary}")
        team_todo_summary = _build_team_todo_summary(team_execution)
        if team_todo_summary:
            leader_prompt_parts.append(f"[团队Todo]\n{team_todo_summary}")
        leader_prompt = "\n\n".join(part for part in leader_prompt_parts if part.strip())

        result = leader_agent.invoke(
            {"messages": [{"role": "user", "content": leader_prompt}]},
            config=leader_runtime_config or {},
        )
        leader_tool_names: list[str] = []
        ask_human_requests: list[dict[str, str]] = []
        tool_trace_events: list[dict[str, str]] = []
        skill_activation_events: list[dict[str, str]] = []
        tool_activation_events: list[dict[str, str]] = []
        if isinstance(result, dict):
            leader_tool_names = sorted(extract_tool_names_from_result(result))
            ask_human_requests = extract_ask_human_requests_from_result(result)
            tool_trace_events = extract_tool_trace_events_from_result(result)
            skill_activation_events = extract_skill_activation_events_from_result(result)
            tool_activation_events = extract_tool_activation_events_from_result(result)
            answer = extract_result_text(result)
        else:
            answer = _llm_content_to_text(getattr(result, "content", result))
        answer = sanitize_public_answer(answer)
        if not answer:
            answer = "抱歉，我暂时没有生成有效回复。"

        for item in tool_trace_events:
            sender = str(item.get("sender") or "leader")
            receiver = str(item.get("receiver") or "tool")
            performative = str(item.get("performative") or "tool_call")
            content = str(item.get("content") or "")
            tool_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender=sender,
                receiver=receiver,
                performative=performative,
                content=content or "(tool event)",
                parent_span_id=current_parent_span,
                on_event=on_event,
            )
            current_parent_span = str(tool_event.get("span_id") or current_parent_span)

        for item in skill_activation_events:
            sender = str(item.get("sender") or "leader")
            receiver = str(item.get("receiver") or "skill")
            content = str(item.get("content") or "")
            skill_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender=sender,
                receiver=receiver,
                performative="skill_activate",
                content=content or "activate skill",
                parent_span_id=current_parent_span,
                on_event=on_event,
            )
            current_parent_span = str(skill_event.get("span_id") or current_parent_span)

        for item in tool_activation_events:
            sender = str(item.get("sender") or "leader")
            receiver = str(item.get("receiver") or "tool")
            content = str(item.get("content") or "")
            activation_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender=sender,
                receiver=receiver,
                performative="tool_activate",
                content=content or "activate tool",
                parent_span_id=current_parent_span,
                on_event=on_event,
            )
            current_parent_span = str(activation_event.get("span_id") or current_parent_span)

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
            ask_human_requests=ask_human_requests,
        )
    finally:
        if async_predictor is not None:
            async_predictor.stop()


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
