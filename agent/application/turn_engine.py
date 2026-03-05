import json
import time
from typing import Any

from ..domain.orchestration import TraceEvent
from ..domain.trace import phase_label_from_performative, phase_summary
from ..method_compare_parser import extract_json_string, parse_method_compare_payload
from ..orchestration.orchestrator import execute_orchestrated_turn
from ..output_cleaner import replace_evidence_placeholders
from .contracts import EventCallback, SearchDocumentFn, TurnCoreResult
from .ports import AgentInvoker, EvidenceRetriever, OrchestratedTurnExecutor


def normalize_evidence_items(raw_payload: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_payload, dict):
        return []
    evidences = raw_payload.get("evidences")
    if not isinstance(evidences, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in evidences:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        normalized.append(item)
    return normalized


def build_search_document_fn(
    search_document_evidence_fn: EvidenceRetriever | None,
) -> SearchDocumentFn:
    if not callable(search_document_evidence_fn):
        return lambda _query: ""

    def _search(query: str) -> str:
        payload = search_document_evidence_fn(query)
        evidence_items = normalize_evidence_items(payload)
        return "\n".join(str(item.get("text", "")) for item in evidence_items)

    return _search


def try_parse_mindmap(answer: str) -> dict[str, Any] | None:
    if not isinstance(answer, str) or not answer.strip():
        return None
    try:
        json_block = extract_json_string(answer)
    except Exception:
        return None
    try:
        payload = json.loads(json_block)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if "name" not in payload:
        return None
    children = payload.get("children")
    if children is not None and not isinstance(children, list):
        return None
    return payload


def execute_turn_core(
    *,
    prompt: str,
    hinted_prompt: str,
    leader_agent: AgentInvoker,
    leader_runtime_config: dict[str, Any] | None,
    leader_llm: Any | None = None,
    search_document_evidence_fn: EvidenceRetriever | None = None,
    force_plan: bool | None = None,
    force_team: bool | None = None,
    routing_context: str = "",
    on_event: EventCallback | None = None,
    orchestrated_turn_executor: OrchestratedTurnExecutor | None = None,
) -> TurnCoreResult:
    if leader_agent is None:
        raise ValueError("Leader agent is not initialized")

    event_logs: list[TraceEvent] = []
    phase_labels: list[str] = []

    def _collect_event(item: TraceEvent) -> None:
        phase = phase_label_from_performative(str(item.get("performative", "")))
        phase_labels.append(phase)
        event: TraceEvent = dict(item)
        event["sender"] = str(item.get("sender", "unknown"))
        event["receiver"] = str(item.get("receiver", "unknown"))
        event["performative"] = str(item.get("performative", "message"))
        event["content"] = str(item.get("content", ""))
        event["phase"] = phase
        event_logs.append(event)
        if on_event is not None:
            on_event(event)

    run_started = time.perf_counter()
    search_document_fn = build_search_document_fn(search_document_evidence_fn)
    orchestrator = (
        execute_orchestrated_turn
        if orchestrated_turn_executor is None
        else orchestrated_turn_executor
    )
    orchestrated = orchestrator(
        prompt=prompt,
        hinted_prompt=hinted_prompt,
        leader_agent=leader_agent,
        leader_runtime_config=(
            leader_runtime_config if isinstance(leader_runtime_config, dict) else {}
        ),
        llm=leader_llm,
        search_document_fn=search_document_fn,
        search_document_evidence_fn=(
            search_document_evidence_fn if callable(search_document_evidence_fn) else None
        ),
        force_plan=force_plan,
        force_team=force_team,
        routing_context=routing_context,
        on_event=_collect_event,
    )

    answer = orchestrated.answer or "抱歉，我暂时没有生成有效回复。"
    policy_decision = orchestrated.policy_decision.to_dict()
    team_execution = orchestrated.team_execution.to_dict()
    trace_payload = event_logs if event_logs else orchestrated.trace_payload

    leader_tool_names = {
        str(name).strip()
        for name in getattr(orchestrated, "leader_tool_names", [])
        if str(name).strip()
    }
    used_document_rag = "search_document" in leader_tool_names
    evidence_items: list[dict[str, Any]] = []
    if used_document_rag and callable(search_document_evidence_fn):
        try:
            evidence_payload = search_document_evidence_fn(prompt)
            evidence_items = normalize_evidence_items(evidence_payload)
        except Exception:
            evidence_items = []

    answer = replace_evidence_placeholders(answer, evidence_items)
    method_compare_data = parse_method_compare_payload(answer)
    mindmap_data = try_parse_mindmap(answer)
    run_latency_ms = (time.perf_counter() - run_started) * 1000.0
    phase_path = phase_summary(phase_labels)

    return {
        "answer": answer,
        "policy_decision": policy_decision,
        "team_execution": team_execution,
        "trace_payload": trace_payload,
        "evidence_items": evidence_items,
        "mindmap_data": mindmap_data,
        "method_compare_data": method_compare_data,
        "run_latency_ms": run_latency_ms,
        "team_rounds": int(team_execution.get("rounds", 0)),
        "phase_path": phase_path,
        "used_document_rag": used_document_rag,
    }
