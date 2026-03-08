from dataclasses import dataclass, field
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
    if act in {"policy", "fallback"}:
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
    if act in {"draft", "review"}:
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
class OrchestratedTurn:
    answer: str
    policy_decision: PolicyDecision
    team_execution: TeamExecution
    trace_payload: TracePayload
    plan_text: str = ""
    leader_tool_names: list[str] = field(default_factory=list)
    ask_human_requests: list[dict[str, str]] = field(default_factory=list)
