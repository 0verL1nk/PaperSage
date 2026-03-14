import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.orchestration import (
    ExecutionPlan,
    OrchestratedTurn,
    PlanRuntimeState,
    PolicyDecision,
    TeamExecution,
    TraceContext,
    TraceEvent,
    TracePayload,
    build_trace_event,
    create_plan_runtime_state,
    create_trace_context,
    evolve_plan_runtime_state,
    list_unready_step_ids,
    next_ready_plan_step,
    render_execution_plan,
)
from ..domain.request_context import RequestContext
from ..domain.revision_policy import failure_needs_revision, has_revision_budget
from ..output_cleaner import sanitize_public_answer
from ..settings import load_agent_settings
from ..stream import (
    extract_ask_human_requests_from_result,
    extract_mode_activation_events_from_result,
    extract_result_text,
    extract_skill_activation_events_from_result,
    extract_tool_activation_events_from_result,
    extract_tool_names_from_result,
    extract_tool_trace_events_from_result,
)
from .langgraph_route_node import run_policy_route_node
from .planning_service import build_execution_plan, revise_execution_plan
from .policy_engine import intercept as intercept_policy
from .team_runtime import run_team_tasks

logger = logging.getLogger(__name__)

MAX_SINGLE_AGENT_STEP_RETRIES = 1
MAX_SINGLE_AGENT_PLAN_CYCLES = 2
WORKFLOW_PLAN_ACT = "plan_act"
WORKFLOW_TEAM = "team"


@dataclass(frozen=True)
class StepExecutionResult:
    answer: str
    leader_tool_names: list[str]
    ask_human_requests: list[dict[str, str]]
    tool_trace_events: list[dict[str, str]]
    skill_activation_events: list[dict[str, str]]
    tool_activation_events: list[dict[str, str]]
    evidence_items: list[dict[str, str]]


@dataclass(frozen=True)
class StepVerificationResult:
    passed: bool
    reason: str


def _requested_modes_from_events(events: list[dict[str, str]]) -> set[str]:
    requested: set[str] = set()
    for item in events:
        if not isinstance(item, dict):
            continue
        receiver = str(item.get("receiver") or "").strip().lower()
        if receiver == "mode:plan":
            requested.add("plan")
        elif receiver == "mode:team":
            requested.add("team")
    return requested


def _build_leader_first_default_decision(advisory_decision: PolicyDecision) -> PolicyDecision:
    return PolicyDecision(
        plan_enabled=False,
        team_enabled=False,
        reason=(
            "leader-first default path"
            + (
                f" | advisory={advisory_decision.reason}"
                if str(advisory_decision.reason or "").strip()
                else ""
            )
        ),
        confidence=advisory_decision.confidence,
        source="leader_first",
    )


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

    # 取本次规划的 plan_id（所有记录共享同一个 plan_id）
    current_plan_id = str(todo_records[0].get("plan_id") or "").strip() if todo_records else ""

    existing: list[dict[str, Any]] = []
    if path.exists() and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing = [item for item in payload if isinstance(item, dict)]
        except Exception:
            existing = []

    # 删除与本次 plan_id 相同的旧记录（同一次规划，全量替换）
    if current_plan_id:
        existing = [
            item for item in existing if str(item.get("plan_id") or "").strip() != current_plan_id
        ]

    # 追加本次规划的所有记录
    for raw in todo_records:
        if isinstance(raw, dict):
            existing.append(dict(raw))

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


def _extract_leader_result_payload(
    result: Any,
) -> tuple[
    str,
    list[str],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
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
    return (
        answer,
        leader_tool_names,
        ask_human_requests,
        tool_trace_events,
        skill_activation_events,
        tool_activation_events,
    )


def _emit_leader_runtime_events(
    *,
    trace_payload: TracePayload,
    trace_context: TraceContext,
    parent_span_id: str,
    tool_trace_events: list[dict[str, str]],
    skill_activation_events: list[dict[str, str]],
    tool_activation_events: list[dict[str, str]],
    on_event: Callable[[TraceEvent], None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    current_parent_span = str(parent_span_id or "")
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
            metadata=metadata,
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
            metadata=metadata,
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
            metadata=metadata,
            on_event=on_event,
        )
        current_parent_span = str(activation_event.get("span_id") or current_parent_span)
    return current_parent_span


def _build_artifact_summary(
    runtime_state: PlanRuntimeState | None,
    *,
    limit: int = 6,
) -> str:
    if runtime_state is None:
        return ""
    artifacts = list(getattr(runtime_state, "artifacts", []) or [])
    if not artifacts:
        return ""
    lines: list[str] = []
    for item in artifacts[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("step_id") or item.get("type") or "artifact").strip()
        content = str(item.get("content") or "").strip().replace("\n", " ")
        if len(content) > 160:
            content = f"{content[:160]}..."
        lines.append(f"- {label}: {content or '(empty)'}")
    return "\n".join(lines)


def _build_completed_steps_summary(runtime_state: PlanRuntimeState | None) -> str:
    if runtime_state is None:
        return ""
    completed_step_ids = list(getattr(runtime_state, "completed_step_ids", []) or [])
    if not completed_step_ids:
        return ""
    return ", ".join(str(item).strip() for item in completed_step_ids if str(item).strip())


def _build_final_leader_prompt(
    *,
    hinted_prompt: str,
    plan_text: str,
    plan: ExecutionPlan | None,
    runtime_state: PlanRuntimeState | None,
    team_execution: TeamExecution,
) -> str:
    leader_prompt_parts = [hinted_prompt]
    if plan_text:
        leader_prompt_parts.append(f"[执行计划]\n{plan_text}")
    if plan is not None and not team_execution.enabled:
        completed_summary = _build_completed_steps_summary(runtime_state)
        if completed_summary:
            leader_prompt_parts.append(f"[已完成步骤]\n{completed_summary}")
        artifact_summary = _build_artifact_summary(runtime_state)
        if artifact_summary:
            leader_prompt_parts.append(f"[步骤产物]\n{artifact_summary}")
        if runtime_state is not None and runtime_state.errors:
            leader_prompt_parts.append(
                "[执行错误]\n" + "\n".join(f"- {item}" for item in runtime_state.errors[-6:])
            )
        leader_prompt_parts.append("请基于以上已完成步骤产物生成最终回答。")
    if team_execution.summary:
        leader_prompt_parts.append(f"[团队结果]\n{team_execution.summary}")
    team_todo_summary = _build_team_todo_summary(team_execution)
    if team_todo_summary:
        leader_prompt_parts.append(f"[团队Todo]\n{team_todo_summary}")
    if team_execution.enabled:
        leader_prompt_parts.append(
            "请直接将上述[团队结果]的核心结论作为你的最终回答展示给用户，不要丢失关键数据。\n"
            "如果有大段的结构化分析、报告内容或长篇大论，请务必将其包裹在 <report></report> 标签内输出，\n"
            "例如:\n"
            "<report>\n# 分析报告标题\n...正文...\n</report>\n"
            "这样前端可以将其渲染为专属的报告框展示给用户。请不要输出多余的口语化引导语。"
        )
    return "\n\n".join(part for part in leader_prompt_parts if part.strip())


def _extract_step_evidence(
    step_answer: str,
    tool_trace_events: list[dict[str, str]],
) -> list[dict[str, str]]:
    evidence_items: list[dict[str, str]] = []
    normalized_answer = str(step_answer or "").strip()
    if normalized_answer and (
        "[chunk_" in normalized_answer
        or ":chunk_" in normalized_answer
        or "证据" in normalized_answer
        or "evidence" in normalized_answer.lower()
    ):
        evidence_items.append(
            {
                "source": "answer",
                "content": normalized_answer,
            }
        )
    for item in tool_trace_events:
        if not isinstance(item, dict):
            continue
        performative = str(item.get("performative") or "").strip()
        if performative != "tool_result":
            continue
        source = str(item.get("sender") or item.get("receiver") or "tool").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        evidence_items.append({"source": source, "content": content})
    return evidence_items


def _can_start_next_plan_cycle(plan_cycle_index: int, *, max_cycles: int) -> bool:
    return has_revision_budget(plan_cycle_index, max_cycles, minimum=1)


def _verify_step_answer(
    answer: str,
    *,
    step_title: str,
    step_done_when: str,
    step_tool_hints: list[str],
    used_tool_names: list[str],
    tool_trace_events: list[dict[str, str]],
) -> tuple[bool, str]:
    normalized = str(answer or "").strip()
    if not normalized:
        return False, "empty_step_result"
    if normalized == "抱歉，我暂时没有生成有效回复。":
        return False, "empty_step_result"
    normalized_tools = {str(item).strip() for item in used_tool_names if str(item).strip()}
    required_tools = [str(item).strip() for item in step_tool_hints if str(item).strip()]
    if required_tools and not any(tool in normalized_tools for tool in required_tools):
        return False, f"missing_required_tool:{required_tools[0]}"
    tool_results = [
        item
        for item in tool_trace_events
        if isinstance(item, dict) and str(item.get("performative") or "").strip() == "tool_result"
    ]
    if required_tools and not tool_results:
        return False, "missing_tool_result"
    verification_text = f"{step_title}\n{step_done_when}".strip()
    evidence_items = _extract_step_evidence(normalized, tool_trace_events)
    if "证据" in verification_text:
        has_evidence_hint = bool(evidence_items) or "search_document" in normalized_tools
        if not has_evidence_hint:
            return False, "missing_evidence"
    return True, "passed"


def _execute_plan_step(
    *,
    prompt: str,
    hinted_prompt: str,
    plan_text: str,
    step: Any,
    runtime_state: PlanRuntimeState,
    leader_agent: Any,
    leader_runtime_config: dict[str, Any] | None,
) -> StepExecutionResult:
    step_prompt = _build_step_prompt(
        prompt=prompt,
        hinted_prompt=hinted_prompt,
        plan_text=plan_text,
        step_id=str(step.id or "").strip(),
        step_title=str(step.title or "").strip(),
        step_description=str(step.description or "").strip(),
        step_done_when=str(step.done_when or "").strip(),
        step_tool_hints=list(step.tool_hints),
        runtime_state=runtime_state,
    )
    result = leader_agent.invoke(
        {"messages": [{"role": "user", "content": step_prompt}]},
        config=leader_runtime_config or {},
    )
    (
        answer,
        leader_tool_names,
        ask_human_requests,
        tool_trace_events,
        skill_activation_events,
        tool_activation_events,
    ) = _extract_leader_result_payload(result)
    return StepExecutionResult(
        answer=answer,
        leader_tool_names=leader_tool_names,
        ask_human_requests=ask_human_requests,
        tool_trace_events=tool_trace_events,
        skill_activation_events=skill_activation_events,
        tool_activation_events=tool_activation_events,
        evidence_items=_extract_step_evidence(answer, tool_trace_events),
    )


def _verify_step_execution(
    *,
    step: Any,
    step_result: StepExecutionResult,
) -> StepVerificationResult:
    passed, reason = _verify_step_answer(
        step_result.answer,
        step_title=str(step.title or "").strip(),
        step_done_when=str(step.done_when or "").strip(),
        step_tool_hints=list(step.tool_hints),
        used_tool_names=step_result.leader_tool_names,
        tool_trace_events=step_result.tool_trace_events,
    )
    return StepVerificationResult(passed=passed, reason=reason)


def _build_step_prompt(
    *,
    prompt: str,
    hinted_prompt: str,
    plan_text: str,
    step_id: str,
    step_title: str,
    step_description: str,
    step_done_when: str,
    step_tool_hints: list[str],
    runtime_state: PlanRuntimeState | None,
) -> str:
    parts = [hinted_prompt, f"[总目标]\n{prompt}"]
    if plan_text:
        parts.append(f"[执行计划]\n{plan_text}")
    completed_summary = _build_completed_steps_summary(runtime_state)
    if completed_summary:
        parts.append(f"[已完成步骤]\n{completed_summary}")
    artifact_summary = _build_artifact_summary(runtime_state)
    if artifact_summary:
        parts.append(f"[已有产物]\n{artifact_summary}")
    parts.append(
        "[当前步骤]\n"
        f"step_id={step_id}\n"
        f"标题：{step_title}\n"
        f"说明：{step_description or step_title}\n"
        f"完成标准：{step_done_when or '完成当前步骤'}"
    )
    if step_tool_hints:
        parts.append(f"[建议工具]\n{', '.join(step_tool_hints)}")
    parts.append("只执行当前步骤，不要直接输出最终完整答案。返回当前步骤结果。")
    return "\n\n".join(part for part in parts if str(part).strip())


def _run_single_agent_plan_steps(
    *,
    prompt: str,
    hinted_prompt: str,
    plan: ExecutionPlan,
    plan_text: str,
    runtime_state: PlanRuntimeState,
    leader_agent: Any,
    leader_runtime_config: dict[str, Any] | None,
    trace_payload: TracePayload,
    trace_context: TraceContext,
    parent_span_id: str,
    on_event: Callable[[TraceEvent], None] | None = None,
    max_retries: int = MAX_SINGLE_AGENT_STEP_RETRIES,
) -> tuple[PlanRuntimeState, list[str], list[dict[str, str]], str, dict[str, str] | None]:
    current_parent_span = str(parent_span_id or "")
    observed_tool_names: set[str] = set()
    ask_human_requests: list[dict[str, str]] = []
    current_state = runtime_state
    failure_payload: dict[str, str] | None = None
    while True:
        step = next_ready_plan_step(plan, current_state.completed_step_ids)
        if step is None:
            unready_step_ids = list_unready_step_ids(plan, current_state.completed_step_ids)
            if unready_step_ids:
                blocked_step_id = str(unready_step_ids[0]).strip()
                blocked_step = next(
                    (item for item in plan.steps if str(item.id or "").strip() == blocked_step_id),
                    None,
                )
                failure_payload = {
                    "step_id": blocked_step_id,
                    "step_title": str(
                        (blocked_step.title if blocked_step is not None else blocked_step_id)
                        or blocked_step_id
                    ).strip(),
                    "reason": "dependency_unresolved",
                }
            break
        step_id = str(step.id or "").strip()
        step_title = str(step.title or step_id or "step").strip()
        if not step_id:
            continue
        current_state = evolve_plan_runtime_state(
            current_state,
            current_step_id=step_id,
        )
        verified = False
        reason = "step_not_verified"
        normalized_retries = max(0, int(max_retries))
        total_attempts = normalized_retries + 1
        for attempt in range(1, total_attempts + 1):
            step_meta = {
                "step_id": step_id,
                "step_title": step_title,
                "attempt": attempt,
                "retry_count": max(0, attempt - 1),
            }
            dispatch_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="coordinator",
                receiver="leader",
                performative="step_dispatch",
                content=f"[{step_id}] {step_title}",
                parent_span_id=current_parent_span,
                metadata=step_meta,
                on_event=on_event,
            )
            current_parent_span = str(dispatch_event.get("span_id") or current_parent_span)
            step_result = _execute_plan_step(
                prompt=prompt,
                hinted_prompt=hinted_prompt,
                plan_text=plan_text,
                step=step,
                runtime_state=current_state,
                leader_agent=leader_agent,
                leader_runtime_config=leader_runtime_config,
            )
            observed_tool_names.update(step_result.leader_tool_names)
            ask_human_requests.extend(step_result.ask_human_requests)
            step_result_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="leader",
                receiver="coordinator",
                performative="step_result",
                content=step_result.answer,
                parent_span_id=current_parent_span,
                metadata=step_meta,
                on_event=on_event,
            )
            current_parent_span = str(step_result_event.get("span_id") or current_parent_span)
            current_parent_span = _emit_leader_runtime_events(
                trace_payload=trace_payload,
                trace_context=trace_context,
                parent_span_id=current_parent_span,
                tool_trace_events=step_result.tool_trace_events,
                skill_activation_events=step_result.skill_activation_events,
                tool_activation_events=step_result.tool_activation_events,
                on_event=on_event,
                metadata=step_meta,
            )
            verification = _verify_step_execution(step=step, step_result=step_result)
            reason = verification.reason
            verify_meta = {
                **step_meta,
                "verification_status": "passed" if verification.passed else "failed",
            }
            verify_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="verifier",
                receiver="coordinator",
                performative="step_verify",
                content=reason,
                parent_span_id=current_parent_span,
                metadata=verify_meta,
                on_event=on_event,
            )
            current_parent_span = str(verify_event.get("span_id") or current_parent_span)
            if verification.passed:
                current_state = evolve_plan_runtime_state(
                    current_state,
                    completed_step_id=step_id,
                    artifact={
                        "type": "step_result",
                        "step_id": step_id,
                        "title": step_title,
                        "content": step_result.answer,
                    },
                    evidence_item=step_result.evidence_items[0]
                    if step_result.evidence_items
                    else None,
                    budget_usage_delta={"step_calls": 1},
                )
                complete_event = _emit_event(
                    trace_payload=trace_payload,
                    trace_context=trace_context,
                    sender="coordinator",
                    receiver="leader",
                    performative="step_complete",
                    content=f"[{step_id}] completed",
                    parent_span_id=current_parent_span,
                    metadata=verify_meta,
                    on_event=on_event,
                )
                current_parent_span = str(complete_event.get("span_id") or current_parent_span)
                verified = True
                break
            current_state = evolve_plan_runtime_state(
                current_state,
                current_step_id=step_id,
                error=f"{step_id}:{reason}",
                budget_usage_delta={"step_calls": 1},
            )
            if attempt <= normalized_retries:
                retry_event = _emit_event(
                    trace_payload=trace_payload,
                    trace_context=trace_context,
                    sender="coordinator",
                    receiver="leader",
                    performative="step_retry",
                    content=f"[{step_id}] {reason}",
                    parent_span_id=current_parent_span,
                    metadata=verify_meta,
                    on_event=on_event,
                )
                current_parent_span = str(retry_event.get("span_id") or current_parent_span)
        if not verified:
            failure_payload = {
                "step_id": step_id,
                "step_title": step_title,
                "reason": reason,
            }
            break
    return (
        current_state,
        sorted(observed_tool_names),
        ask_human_requests,
        current_parent_span,
        failure_payload,
    )


def _run_plan_runtime(
    *,
    prompt: str,
    hinted_prompt: str,
    plan: ExecutionPlan,
    plan_text: str,
    runtime_state: PlanRuntimeState,
    leader_agent: Any,
    leader_runtime_config: dict[str, Any] | None,
    llm: Any | None,
    trace_payload: TracePayload,
    trace_context: TraceContext,
    parent_span_id: str,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> tuple[PlanRuntimeState, set[str], list[dict[str, str]], str, ExecutionPlan, str]:
    current_parent_span = str(parent_span_id or "")
    current_plan = plan
    current_plan_text = plan_text
    current_state = runtime_state
    observed_leader_tool_names: set[str] = set()
    aggregated_ask_human_requests: list[dict[str, str]] = []
    plan_cycle_index = 0
    while True:
        (
            current_state,
            step_tool_names,
            step_ask_human_requests,
            current_parent_span,
            step_failure,
        ) = _run_single_agent_plan_steps(
            prompt=prompt,
            hinted_prompt=hinted_prompt,
            plan=current_plan,
            plan_text=current_plan_text,
            runtime_state=current_state,
            leader_agent=leader_agent,
            leader_runtime_config=leader_runtime_config,
            trace_payload=trace_payload,
            trace_context=trace_context,
            parent_span_id=current_parent_span,
            on_event=on_event,
        )
        observed_leader_tool_names.update(step_tool_names)
        aggregated_ask_human_requests.extend(step_ask_human_requests)
        if step_failure is None:
            break
        failure_reason = str(step_failure.get("reason") or "").strip()
        if not failure_needs_revision(failure_reason):
            current_state = evolve_plan_runtime_state(
                current_state,
                error=f"replan_skipped:{failure_reason or 'unknown_failure'}",
            )
            skip_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="planner",
                receiver="leader",
                performative="fallback",
                content=f"replan_skipped:{failure_reason or 'unknown_failure'}",
                parent_span_id=current_parent_span,
                metadata={
                    "failed_step_id": str(step_failure.get("step_id") or "").strip(),
                    "failure_reason": failure_reason,
                },
                on_event=on_event,
            )
            current_parent_span = str(skip_event.get("span_id") or current_parent_span)
            break
        plan_cycle_index += 1
        if not _can_start_next_plan_cycle(
            plan_cycle_index,
            max_cycles=MAX_SINGLE_AGENT_PLAN_CYCLES,
        ):
            current_state = evolve_plan_runtime_state(
                current_state,
                error="plan_cycle_guard_triggered",
            )
            guard_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="planner",
                receiver="leader",
                performative="fallback",
                content="plan_cycle_guard_triggered",
                parent_span_id=current_parent_span,
                metadata={
                    "failed_step_id": str(step_failure.get("step_id") or "").strip(),
                    "failure_reason": failure_reason,
                },
                on_event=on_event,
            )
            current_parent_span = str(guard_event.get("span_id") or current_parent_span)
            break
        revised_plan = revise_execution_plan(
            prompt=prompt,
            current_plan=current_plan,
            failed_step_id=str(step_failure.get("step_id") or "").strip(),
            failure_reason=failure_reason,
            llm=llm,
        )
        current_plan = revised_plan
        current_plan_text = render_execution_plan(revised_plan)
        current_state = evolve_plan_runtime_state(
            current_state,
            current_plan=revised_plan,
            current_step_id=None,
            budget_usage_delta={"replan_calls": 1},
        )
        replan_event = _emit_event(
            trace_payload=trace_payload,
            trace_context=trace_context,
            sender="planner",
            receiver="leader",
            performative="replan",
            content=current_plan_text,
            parent_span_id=current_parent_span,
            metadata={
                "failed_step_id": str(step_failure.get("step_id") or "").strip(),
                "failure_reason": failure_reason,
            },
            on_event=on_event,
        )
        current_parent_span = str(replan_event.get("span_id") or current_parent_span)
    return (
        current_state,
        observed_leader_tool_names,
        aggregated_ask_human_requests,
        current_parent_span,
        current_plan,
        current_plan_text,
    )


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
        settings.agent_team_max_members if max_team_members is None else int(max_team_members)
    )
    resolved_team_rounds = (
        settings.agent_team_max_rounds if max_team_rounds is None else int(max_team_rounds)
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

    advisory_decision = intercept_policy(
        RequestContext(
            prompt=prompt,
            context_digest=routing_context,
        ),
        llm=policy_router_llm,
    )
    policy_event = _emit_event(
        trace_payload=trace_payload,
        trace_context=trace_context,
        sender="policy_engine",
        receiver="leader",
        performative="policy",
        content=(
            f"plan={advisory_decision.plan_enabled}, "
            f"team={advisory_decision.team_enabled}, "
            f"reason={advisory_decision.reason}"
        ),
        parent_span_id=current_parent_span,
        metadata={"advisory_only": True},
        on_event=on_event,
    )
    current_parent_span = str(policy_event.get("span_id") or current_parent_span)
    policy_decision = _build_leader_first_default_decision(advisory_decision)
    workflow_mode = run_policy_route_node(advisory_decision)
    pending_plan_from_policy = workflow_mode in {WORKFLOW_PLAN_ACT, WORKFLOW_TEAM}
    pending_team_from_policy = workflow_mode == WORKFLOW_TEAM

    plan: ExecutionPlan | None = None
    plan_text = ""
    runtime_state: PlanRuntimeState | None = None

    policy_context = routing_context
    if runtime_state is None:
        runtime_state = create_plan_runtime_state(
            user_goal=prompt,
            current_plan=plan,
            context_summary=policy_context or routing_context,
        )

    team_execution = TeamExecution(enabled=False)

    pre_final_context = policy_context
    if team_execution.summary:
        pre_final_context = f"{pre_final_context}\n\n[团队摘要]\n{team_execution.summary[:1200]}"

    observed_leader_tool_names: set[str] = set()
    aggregated_ask_human_requests: list[dict[str, str]] = []

    answer = ""
    final_leader_passes = 0
    while True:
        final_leader_passes += 1
        leader_prompt = _build_final_leader_prompt(
            hinted_prompt=hinted_prompt,
            plan_text=plan_text,
            plan=plan,
            runtime_state=runtime_state,
            team_execution=team_execution,
        )
        result = leader_agent.invoke(
            {"messages": [{"role": "user", "content": leader_prompt}]},
            config=leader_runtime_config or {},
        )
        (
            answer,
            leader_tool_names,
            ask_human_requests,
            tool_trace_events,
            skill_activation_events,
            tool_activation_events,
        ) = _extract_leader_result_payload(result)
        observed_leader_tool_names.update(leader_tool_names)
        aggregated_ask_human_requests.extend(ask_human_requests)
        mode_activation_events = (
            extract_mode_activation_events_from_result(result) if isinstance(result, dict) else []
        )
        current_parent_span = _emit_leader_runtime_events(
            trace_payload=trace_payload,
            trace_context=trace_context,
            parent_span_id=current_parent_span,
            tool_trace_events=tool_trace_events,
            skill_activation_events=skill_activation_events,
            tool_activation_events=tool_activation_events,
            on_event=on_event,
        )
        for item in mode_activation_events:
            mode_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender=str(item.get("sender") or "leader"),
                receiver=str(item.get("receiver") or "mode"),
                performative="mode_activate",
                content=str(item.get("content") or "activate mode"),
                parent_span_id=current_parent_span,
                on_event=on_event,
            )
            current_parent_span = str(mode_event.get("span_id") or current_parent_span)
        requested_modes = _requested_modes_from_events(mode_activation_events)
        requested_plan = "plan" in requested_modes and plan is None
        requested_team = "team" in requested_modes and not team_execution.enabled
        should_run_plan = pending_plan_from_policy or requested_plan
        should_run_team = pending_team_from_policy or requested_team
        pending_plan_from_policy = False
        pending_team_from_policy = False
        if not should_run_plan and not should_run_team:
            break
        if should_run_plan:
            plan = build_execution_plan(prompt, llm=llm)
            plan_text = render_execution_plan(plan)
            runtime_state = create_plan_runtime_state(
                user_goal=prompt,
                current_plan=plan,
                context_summary=routing_context,
            )
            plan_event = _emit_event(
                trace_payload=trace_payload,
                trace_context=trace_context,
                sender="planner",
                receiver="leader",
                performative="plan",
                content=plan_text,
                parent_span_id=current_parent_span,
                metadata={"stage": "policy_route_node" if not requested_plan else "leader_tool"},
                on_event=on_event,
            )
            current_parent_span = str(plan_event.get("span_id") or current_parent_span)
            runtime_state = evolve_plan_runtime_state(
                runtime_state,
                budget_usage_delta={"planner_calls": 1},
            )
            (
                runtime_state,
                plan_tool_names,
                plan_ask_human_requests,
                current_parent_span,
                plan,
                plan_text,
            ) = _run_plan_runtime(
                prompt=prompt,
                hinted_prompt=hinted_prompt,
                plan=plan,
                plan_text=plan_text,
                runtime_state=runtime_state,
                leader_agent=leader_agent,
                leader_runtime_config=leader_runtime_config,
                llm=llm,
                trace_payload=trace_payload,
                trace_context=trace_context,
                parent_span_id=current_parent_span,
                on_event=on_event,
            )
            observed_leader_tool_names.update(plan_tool_names)
            aggregated_ask_human_requests.extend(plan_ask_human_requests)
        if should_run_team:
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
                policy_checkpoint_fn=None,
            )
            if team_execution.trace_events:
                current_parent_span = str(
                    team_execution.trace_events[-1].get("span_id") or current_parent_span
                )
            try:
                synced_todo_path = _persist_team_todo_records(team_execution)
                if synced_todo_path:
                    logger.info("team todo synced: %s", synced_todo_path)
            except Exception as exc:
                logger.warning("team todo sync failed: %s", exc)
        if requested_plan or requested_team:
            policy_decision = PolicyDecision(
                plan_enabled=plan is not None,
                team_enabled=team_execution.enabled,
                reason="leader requested mode runtime via tool call",
                confidence=policy_decision.confidence,
                source="leader_tool",
            )
        if final_leader_passes >= 2:
            break
    runtime_state = evolve_plan_runtime_state(
        runtime_state,
        artifact={"type": "final_answer", "content": answer},
        context_summary=pre_final_context,
        budget_usage_delta={"leader_calls": 1},
    )

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
        1 for e in trace_payload if isinstance(e, dict) and e.get("performative") == "replan"
    )
    _log_routing_decision(prompt, policy_decision, team_execution, replan_rounds)

    return OrchestratedTurn(
        answer=answer,
        policy_decision=policy_decision,
        team_execution=team_execution,
        trace_payload=trace_payload,
        plan=plan,
        plan_text=plan_text,
        runtime_state=runtime_state,
        leader_tool_names=sorted(observed_leader_tool_names),
        ask_human_requests=aggregated_ask_human_requests,
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
