from dataclasses import dataclass
from typing import Any

from ..turn_engine import execute_turn_core


@dataclass(frozen=True)
class TurnRuntimeInputs:
    leader_agent: Any
    leader_runtime_config: dict[str, Any]
    leader_llm: Any | None
    policy_llm: Any | None = None
    search_document_evidence_fn: Any | None = None
    leader_tool_specs: list[dict[str, Any]] | None = None


def resolve_turn_runtime_inputs(session_state: dict[str, Any]) -> TurnRuntimeInputs:
    leader_agent = session_state.get("paper_agent")
    if leader_agent is None:
        raise ValueError("Leader agent is not initialized")

    leader_runtime_config = session_state.get("paper_agent_runtime_config")
    leader_llm = session_state.get("paper_leader_llm")
    policy_llm = session_state.get("paper_policy_router_llm") or leader_llm
    search_document_evidence_fn = session_state.get("paper_evidence_retriever")
    leader_tool_specs = session_state.get("paper_current_tool_specs")
    return TurnRuntimeInputs(
        leader_agent=leader_agent,
        leader_runtime_config=(
            leader_runtime_config if isinstance(leader_runtime_config, dict) else {}
        ),
        leader_llm=leader_llm,
        policy_llm=policy_llm,
        search_document_evidence_fn=(
            search_document_evidence_fn
            if callable(search_document_evidence_fn)
            else None
        ),
        leader_tool_specs=(
            leader_tool_specs if isinstance(leader_tool_specs, list) else []
        ),
    )


def execute_turn_with_runtime(
    *,
    prompt: str,
    hinted_prompt: str,
    runtime_inputs: TurnRuntimeInputs,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    routing_context: str = "",
    on_event=None,
) -> dict[str, Any]:
    return execute_turn_core(
        prompt=prompt,
        hinted_prompt=hinted_prompt,
        leader_agent=runtime_inputs.leader_agent,
        leader_runtime_config=runtime_inputs.leader_runtime_config,
        leader_llm=runtime_inputs.leader_llm,
        policy_llm=runtime_inputs.policy_llm,
        search_document_evidence_fn=runtime_inputs.search_document_evidence_fn,
        leader_tool_specs=runtime_inputs.leader_tool_specs,
        force_plan=force_plan,
        force_team=force_team,
        routing_context=routing_context,
        on_event=on_event,
    )
