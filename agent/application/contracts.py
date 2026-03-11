from collections.abc import Callable
from typing import Any, TypedDict

from typing_extensions import NotRequired

from ..domain.orchestration import TraceEvent

EventCallback = Callable[[TraceEvent], None]
SearchDocumentFn = Callable[[str], str]


class TurnCoreResult(TypedDict):
    answer: str
    policy_decision: dict[str, Any]
    team_execution: dict[str, Any]
    trace_payload: list[TraceEvent]
    evidence_items: list[dict[str, Any]]
    mindmap_data: dict[str, Any] | None
    method_compare_data: dict[str, Any] | None
    run_latency_ms: float
    team_rounds: int
    phase_path: str
    used_document_rag: bool
    ask_human_requests: list[dict[str, str]]
    plan: NotRequired[dict[str, Any] | None]
    runtime_state: NotRequired[dict[str, Any] | None]
