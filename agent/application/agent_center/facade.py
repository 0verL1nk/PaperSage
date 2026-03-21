from dataclasses import dataclass
from typing import Any

from ..contracts import EventCallback, TurnCoreResult
from ..ports import EvidenceRetriever
from ..turn_engine import execute_turn_core


@dataclass(frozen=True)
class AgentCenterTurnRequest:
    prompt: str
    turn_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentCenterRuntimeDeps:
    leader_agent: Any
    leader_runtime_config: dict[str, Any]
    leader_llm: Any | None
    policy_llm: Any | None = None
    search_document_evidence_fn: EvidenceRetriever | None = None
    leader_tool_specs: list[dict[str, Any]] | None = None


def execute_agent_center_turn(
    *,
    request: AgentCenterTurnRequest,
    deps: AgentCenterRuntimeDeps,
    on_event: EventCallback | None = None,
) -> TurnCoreResult:
    return execute_turn_core(
        prompt=request.prompt,
        turn_context=request.turn_context,
        leader_agent=deps.leader_agent,
        leader_runtime_config=deps.leader_runtime_config,
        leader_llm=deps.leader_llm,
        policy_llm=deps.policy_llm,
        search_document_evidence_fn=deps.search_document_evidence_fn,
        leader_tool_specs=deps.leader_tool_specs,
        on_event=on_event,
    )
