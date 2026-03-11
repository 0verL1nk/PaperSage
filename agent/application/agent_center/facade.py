from dataclasses import dataclass
from typing import Any

from ..contracts import EventCallback, TurnCoreResult
from ..ports import EvidenceRetriever
from ..turn_engine import execute_turn_core


@dataclass(frozen=True)
class AgentCenterTurnRequest:
    prompt: str
    hinted_prompt: str
    force_plan: bool | None = None
    force_team: bool | None = None
    routing_context: str = ""
    emit_tool_load_event: bool = True


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
        hinted_prompt=request.hinted_prompt,
        leader_agent=deps.leader_agent,
        leader_runtime_config=deps.leader_runtime_config,
        leader_llm=deps.leader_llm,
        policy_llm=deps.policy_llm,
        search_document_evidence_fn=deps.search_document_evidence_fn,
        leader_tool_specs=deps.leader_tool_specs,
        emit_tool_load_event=request.emit_tool_load_event,
        force_plan=request.force_plan,
        force_team=request.force_team,
        routing_context=request.routing_context,
        on_event=on_event,
    )
