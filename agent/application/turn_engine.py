import json
import logging
import time
from typing import Any

from ..domain.orchestration import (
    OrchestratedTurn,
    PolicyDecision,
    TeamExecution,
    TraceEvent,
)
from ..domain.trace import phase_label_from_performative, phase_summary
from ..method_compare_parser import extract_json_string, parse_method_compare_payload
from ..output_cleaner import replace_evidence_placeholders
from .contracts import EventCallback, SearchDocumentFn, TurnCoreResult
from .ports import AgentInvoker, EvidenceRetriever, OrchestratedTurnExecutor

logger = logging.getLogger(__name__)


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

    # Check if mindmap tag exists
    if "<mindmap>" not in answer.lower():
        return None

    try:
        json_block = extract_json_string(answer)
    except Exception as e:
        logger.warning("Mindmap extraction failed: extract_json_string error: %s", e)
        return None

    try:
        payload = json.loads(json_block)
    except Exception as e:
        logger.warning("Mindmap parse failed: JSON decode error: %s, json_block=%s", e, json_block[:200])
        return None

    if not isinstance(payload, dict):
        logger.warning("Mindmap parse failed: payload is not dict, type=%s", type(payload))
        return None

    if "name" not in payload:
        logger.warning("Mindmap parse failed: missing 'name' field, keys=%s", list(payload.keys()))
        return None

    children = payload.get("children")
    if children is not None and not isinstance(children, list):
        logger.warning("Mindmap parse failed: 'children' is not list, type=%s", type(children))
        return None

    logger.info("Mindmap parsed successfully: name=%s, children_count=%s", payload.get("name"), len(children) if children else 0)
    return payload


def _maybe_to_dict(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    to_dict = getattr(payload, "to_dict", None)
    if not callable(to_dict):
        return None
    result = to_dict()
    if not isinstance(result, dict):
        return None
    return result


def execute_turn_core(
    *,
    prompt: str,
    hinted_prompt: str,
    leader_agent: AgentInvoker,
    leader_runtime_config: dict[str, Any] | None,
    leader_llm: Any | None = None,
    policy_llm: Any | None = None,
    search_document_evidence_fn: EvidenceRetriever | None = None,
    leader_tool_specs: list[dict[str, Any]] | None = None,
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

    registered_tool_names: list[str] = []
    if isinstance(leader_tool_specs, list):
        for item in leader_tool_specs:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            registered_tool_names.append(name)

    run_started = time.perf_counter()
    search_document_fn = build_search_document_fn(search_document_evidence_fn)

    # 直接调用 leader_agent 或使用提供的 executor
    if orchestrated_turn_executor is not None:
        orchestrated = orchestrated_turn_executor(
            prompt=prompt,
            hinted_prompt=hinted_prompt,
            leader_agent=leader_agent,
            leader_runtime_config=(
                leader_runtime_config if isinstance(leader_runtime_config, dict) else {}
            ),
            llm=leader_llm,
            policy_llm=policy_llm,
            search_document_fn=search_document_fn,
            search_document_evidence_fn=(
                search_document_evidence_fn if callable(search_document_evidence_fn) else None
            ),
            routing_context=routing_context,
            on_event=_collect_event,
        )
    else:
        # 直接调用 leader_agent
        result = leader_agent.invoke(
            {"messages": [{"role": "user", "content": hinted_prompt}]},
            config=leader_runtime_config if isinstance(leader_runtime_config, dict) else {},
        )

        # 提取 answer
        answer = ""
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    answer = str(last_msg.content)
                elif isinstance(last_msg, dict):
                    answer = str(last_msg.get("content", ""))

        # 构造简化的 OrchestratedTurn
        orchestrated = OrchestratedTurn(
            answer=answer,
            policy_decision=PolicyDecision(decision="direct", reason="simplified"),
            team_execution=TeamExecution(rounds=0, members=[]),
            trace_payload=[],
        )

    answer = orchestrated.answer
    if not answer:
        logger.warning(
            "Empty answer from orchestrated turn. orchestrated_type=%s, has_answer=%s",
            type(orchestrated).__name__,
            hasattr(orchestrated, "answer"),
        )
        answer = "抱歉，我暂时没有生成有效回复。"
    policy_decision = orchestrated.policy_decision.to_dict()
    team_execution = orchestrated.team_execution.to_dict()
    trace_payload = event_logs if event_logs else orchestrated.trace_payload
    plan_payload = getattr(orchestrated, "plan", None)
    runtime_state_payload = getattr(orchestrated, "runtime_state", None)

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
        "plan": _maybe_to_dict(plan_payload),
        "runtime_state": _maybe_to_dict(runtime_state_payload),
        "evidence_items": evidence_items,
        "mindmap_data": mindmap_data,
        "method_compare_data": method_compare_data,
        "run_latency_ms": run_latency_ms,
        "team_rounds": int(team_execution.get("rounds", 0)),
        "phase_path": phase_path,
        "used_document_rag": used_document_rag,
        "ask_human_requests": list(getattr(orchestrated, "ask_human_requests", []) or []),
        "todos": list(getattr(orchestrated, "todos", []) or []),
        "agent_plan": getattr(orchestrated, "agent_plan", None),
    }
