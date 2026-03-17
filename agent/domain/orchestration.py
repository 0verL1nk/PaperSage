from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from a2a.types import Message, Part, Role, TextPart

TraceEvent = dict[str, Any]
TracePayload = list[TraceEvent]


DEFAULT_TRACE_CHANNEL = "internal.a2a"


@dataclass
class TraceContext:
    run_id: str
    task_id: str
    channel: str = DEFAULT_TRACE_CHANNEL
    _sequence: int = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


def _new_trace_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_trace_context(
    *,
    run_id: str | None = None,
    task_id: str | None = None,
    channel: str = DEFAULT_TRACE_CHANNEL,
) -> TraceContext:
    return TraceContext(
        run_id=str(run_id or _new_trace_id("run")),
        task_id=str(task_id or _new_trace_id("task")),
        channel=str(channel or DEFAULT_TRACE_CHANNEL),
    )


def is_valid_trace_route(sender: str, receiver: str, performative: str) -> bool:
    src = str(sender or "").strip()
    dst = str(receiver or "").strip()
    act = str(performative or "").strip()
    if not src or not dst or not act:
        return False
    if act == "request":
        return src == "user" and dst != "user"
    if act in {"policy", "fallback", "policy_switch"}:
        return dst in {"leader", "coordinator"}
    if act in {"plan", "replan"}:
        return src == "planner" and src != dst
    if act == "plan_todo":
        # leader 自我规划 todo 列表
        return src in {"leader", "coordinator"} and dst in {"leader", "coordinator"}
    if act == "plan_todo_reject":
        # 程序检测到依赖环，反馈给 leader 让其修正
        return src == "system" and dst in {"leader", "coordinator"}
    if act == "dispatch":
        return src in {"leader", "coordinator"} and dst not in {"user", src}
    if act == "member_output":
        return src != "user" and dst != "user" and src != dst
    if act == "final":
        return src != "user" and dst == "user"
    return True


def ensure_valid_trace_route(sender: str, receiver: str, performative: str) -> None:
    if is_valid_trace_route(sender, receiver, performative):
        return
    raise ValueError(
        "Invalid trace route: "
        f"sender={sender!r}, receiver={receiver!r}, performative={performative!r}"
    )


def build_trace_event(
    *,
    context: TraceContext,
    sender: str,
    receiver: str,
    performative: str,
    content: str,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceEvent:
    ensure_valid_trace_route(sender, receiver, performative)
    sequence = context.next_sequence()
    span_id = f"{context.run_id}:span:{sequence}"
    event_timestamp = now_iso()
    sdk_metadata: dict[str, Any] = {
        "sender": str(sender),
        "receiver": str(receiver),
        "performative": str(performative),
        "channel": context.channel,
        "sequence": sequence,
        "timestamp": event_timestamp,
    }
    if parent_span_id:
        sdk_metadata["parentSpanId"] = str(parent_span_id)
    if isinstance(metadata, dict) and metadata:
        sdk_metadata["traceMeta"] = metadata
    sdk_message = Message(
        role=Role.user if str(sender) == "user" else Role.agent,
        parts=[Part(root=TextPart(text=str(content)))],
        message_id=span_id,
        task_id=context.task_id,
        context_id=context.run_id,
        metadata=sdk_metadata,
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    payload: TraceEvent = {
        "sender": str(sender),
        "receiver": str(receiver),
        "performative": str(performative),
        "content": str(content),
        "timestamp": event_timestamp,
        "run_id": context.run_id,
        "task_id": context.task_id,
        "span_id": span_id,
        "parent_span_id": str(parent_span_id or ""),
        "sequence": sequence,
        "channel": context.channel,
        "a2aMessage": sdk_message,
    }
    if isinstance(metadata, dict) and metadata:
        payload["meta"] = metadata
    return payload


@dataclass(frozen=True)
class PolicyDecision:
    plan_enabled: bool
    team_enabled: bool
    reason: str
    confidence: float | None = None
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_enabled": self.plan_enabled,
            "team_enabled": self.team_enabled,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class TeamRole:
    name: str
    goal: str


@dataclass(frozen=True)
class TeamExecution:
    enabled: bool
    roles: list[str] = field(default_factory=list)
    member_count: int = 0
    rounds: int = 0
    summary: str = ""
    fallback_reason: str | None = None
    trace_events: TracePayload = field(default_factory=list)
    todo_records: list[dict[str, Any]] = field(default_factory=list)
    todo_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "roles": self.roles,
            "member_count": self.member_count,
            "rounds": self.rounds,
            "summary": self.summary,
            "fallback_reason": self.fallback_reason,
            "todo_records": self.todo_records,
            "todo_stats": self.todo_stats,
        }


@dataclass(frozen=True)
class PlanStep:
    id: str
    title: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    tool_hints: list[str] = field(default_factory=list)
    done_when: str = ""
    status: str = "todo"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "depends_on": list(self.depends_on),
            "tool_hints": list(self.tool_hints),
            "done_when": self.done_when,
            "status": self.status,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    goal: str
    constraints: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    tool_hints: list[str] = field(default_factory=list)
    done_when: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "constraints": list(self.constraints),
            "steps": [item.to_dict() for item in self.steps],
            "tool_hints": list(self.tool_hints),
            "done_when": self.done_when,
        }


@dataclass(frozen=True)
class PlanRuntimeState:
    user_goal: str
    constraints: list[str] = field(default_factory=list)
    current_plan: ExecutionPlan | None = None
    current_step_id: str = ""
    completed_step_ids: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    budget_usage: dict[str, int] = field(default_factory=dict)
    context_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_goal": self.user_goal,
            "constraints": list(self.constraints),
            "current_plan": self.current_plan.to_dict() if self.current_plan is not None else None,
            "current_step_id": self.current_step_id,
            "completed_step_ids": list(self.completed_step_ids),
            "artifacts": [dict(item) for item in self.artifacts],
            "evidence": [dict(item) for item in self.evidence],
            "errors": list(self.errors),
            "budget_usage": {str(key): int(value) for key, value in self.budget_usage.items()},
            "context_summary": self.context_summary,
        }


def render_execution_plan(plan: ExecutionPlan) -> str:
    lines = [f"目标：{plan.goal.strip() or '完成证据充分的回答'}"]
    for idx, step in enumerate(plan.steps, start=1):
        title = step.title.strip() or f"步骤 {idx}"
        lines.append(f"{idx}. {title}")
    return "\n".join(lines)


def _normalized_completed_step_ids(completed_step_ids: list[str]) -> set[str]:
    return {str(item).strip() for item in completed_step_ids if str(item).strip()}


def _plan_step_index(plan: ExecutionPlan | None) -> dict[str, PlanStep]:
    if plan is None:
        return {}
    records: dict[str, PlanStep] = {}
    for step in plan.steps:
        step_id = str(step.id or "").strip()
        if not step_id:
            continue
        records[step_id] = step
    return records


def _step_is_ready(
    step: PlanStep,
    *,
    completed: set[str],
    indexed_steps: dict[str, PlanStep],
) -> bool:
    dependencies = [str(item).strip() for item in step.depends_on if str(item).strip()]
    if not dependencies:
        return True
    # Unknown dependency should be treated as unresolved to avoid false execution.
    for dep_id in dependencies:
        if dep_id not in indexed_steps:
            return False
        if dep_id not in completed:
            return False
    return True


def next_ready_plan_step(
    plan: ExecutionPlan | None,
    completed_step_ids: list[str],
) -> PlanStep | None:
    if plan is None:
        return None
    completed = _normalized_completed_step_ids(completed_step_ids)
    indexed_steps = _plan_step_index(plan)
    for step in plan.steps:
        step_id = str(step.id or "").strip()
        if not step_id or step_id in completed:
            continue
        if _step_is_ready(step, completed=completed, indexed_steps=indexed_steps):
            return step
    return None


def list_unready_step_ids(
    plan: ExecutionPlan | None,
    completed_step_ids: list[str],
) -> list[str]:
    if plan is None:
        return []
    completed = _normalized_completed_step_ids(completed_step_ids)
    indexed_steps = _plan_step_index(plan)
    unresolved: list[str] = []
    for step in plan.steps:
        step_id = str(step.id or "").strip()
        if not step_id or step_id in completed:
            continue
        if not _step_is_ready(step, completed=completed, indexed_steps=indexed_steps):
            unresolved.append(step_id)
    return unresolved


def _next_pending_step_id(plan: ExecutionPlan | None, completed_step_ids: list[str]) -> str:
    step = next_ready_plan_step(plan, completed_step_ids)
    if step is not None:
        return str(step.id or "").strip()
    if plan is None:
        return ""
    completed = _normalized_completed_step_ids(completed_step_ids)
    for candidate in plan.steps:
        step_id = str(candidate.id or "").strip()
        if step_id and step_id not in completed:
            return step_id
    return ""


def create_plan_runtime_state(
    *,
    user_goal: str,
    current_plan: ExecutionPlan | None = None,
    constraints: list[str] | None = None,
    context_summary: str = "",
) -> PlanRuntimeState:
    normalized_constraints = (
        list(constraints)
        if isinstance(constraints, list)
        else list(getattr(current_plan, "constraints", []) or [])
    )
    return PlanRuntimeState(
        user_goal=str(user_goal or "").strip(),
        constraints=normalized_constraints,
        current_plan=current_plan,
        current_step_id=_next_pending_step_id(current_plan, []),
        context_summary=str(context_summary or "").strip(),
    )


def evolve_plan_runtime_state(
    state: PlanRuntimeState,
    *,
    current_plan: ExecutionPlan | None = None,
    current_step_id: str | None = None,
    completed_step_id: str | None = None,
    artifact: dict[str, Any] | None = None,
    evidence_item: dict[str, Any] | None = None,
    error: str | None = None,
    context_summary: str | None = None,
    budget_usage_delta: dict[str, int] | None = None,
) -> PlanRuntimeState:
    completed_step_ids = list(state.completed_step_ids)
    if completed_step_id:
        normalized_completed = str(completed_step_id).strip()
        if normalized_completed and normalized_completed not in completed_step_ids:
            completed_step_ids.append(normalized_completed)
    artifacts = list(state.artifacts)
    if isinstance(artifact, dict) and artifact:
        artifacts.append(dict(artifact))
    evidence = list(state.evidence)
    if isinstance(evidence_item, dict) and evidence_item:
        evidence.append(dict(evidence_item))
    errors = list(state.errors)
    if error:
        normalized_error = str(error).strip()
        if normalized_error:
            errors.append(normalized_error)
    budget_usage = dict(state.budget_usage)
    if isinstance(budget_usage_delta, dict):
        for key, value in budget_usage_delta.items():
            if isinstance(value, int):
                budget_usage[str(key)] = int(budget_usage.get(str(key), 0)) + value
    next_plan = current_plan if current_plan is not None else state.current_plan
    next_step_id = current_step_id
    if next_step_id is None:
        next_step_id = _next_pending_step_id(next_plan, completed_step_ids)
    return replace(
        state,
        current_plan=next_plan,
        current_step_id=str(next_step_id or "").strip(),
        completed_step_ids=completed_step_ids,
        artifacts=artifacts,
        evidence=evidence,
        errors=errors,
        budget_usage=budget_usage,
        context_summary=(
            str(context_summary).strip()
            if context_summary is not None
            else state.context_summary
        ),
    )


@dataclass(frozen=True)
class OrchestratedTurn:
    answer: str
    policy_decision: PolicyDecision
    team_execution: TeamExecution
    trace_payload: TracePayload
    plan: ExecutionPlan | None = None
    plan_text: str = ""
    runtime_state: PlanRuntimeState | None = None
    leader_tool_names: list[str] = field(default_factory=list)
    ask_human_requests: list[dict[str, str]] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)
    agent_plan: dict[str, str] | None = None
