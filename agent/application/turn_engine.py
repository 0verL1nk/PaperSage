import json
import logging
import time
from typing import Any

from ..domain.orchestration import TraceEvent
from ..domain.trace import phase_label_from_performative, phase_summary
from ..method_compare_parser import extract_json_string, parse_method_compare_payload
from .contracts import EventCallback, SearchDocumentFn, TurnCoreResult
from .ports import AgentInvoker, EvidenceRetriever

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


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "").strip()
    return str(getattr(message, "name", "") or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").strip().lower()
    return str(getattr(message, "type", getattr(message, "role", "")) or "").strip().lower()


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    else:
        tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def _parse_tool_json_payload(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_search_document_evidence_items(
    messages: list[Any],
    *,
    referenced_chunk_ids: list[str],
    referenced_doc_uids: list[str],
) -> list[dict[str, Any]]:
    referenced_chunks = {item.strip() for item in referenced_chunk_ids if item.strip()}
    referenced_docs = {item.strip() for item in referenced_doc_uids if item.strip()}
    if not referenced_chunks and not referenced_docs:
        return []

    def _matches_reference(item: dict[str, Any]) -> bool:
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id and chunk_id in referenced_chunks:
            return True
        doc_uid = str(item.get("doc_uid", "")).strip()
        if doc_uid and doc_uid in referenced_docs:
            return True
        if chunk_id and ":chunk_" in chunk_id:
            chunk_doc_uid = chunk_id.split(":chunk_", 1)[0].strip()
            if chunk_doc_uid and chunk_doc_uid in referenced_docs:
                return True
        return False

    collected: list[dict[str, Any]] = []
    for message in messages:
        if _message_role(message) != "tool" or _message_name(message) != "search_document":
            continue
        payload = _parse_tool_json_payload(_message_content(message))
        if payload is None:
            continue
        collected.extend(normalize_evidence_items(payload))

    matched: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for item in collected:
        if not _matches_reference(item):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        matched.append(item)
    return matched


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
    if isinstance(payload, dict):
        return dict(payload)
    to_dict = getattr(payload, "to_dict", None)
    if not callable(to_dict):
        return None
    result = to_dict()
    if not isinstance(result, dict):
        return None
    return result


def _stable_phase_path(*, phase_labels: list[str], answer: str, messages: list[Any]) -> str:
    normalized_labels = list(phase_labels)
    if (answer.strip() or messages) and (
        not normalized_labels or normalized_labels[-1] != "输出最终答案"
    ):
        normalized_labels.append("输出最终答案")
    return phase_summary(normalized_labels)


def extract_evidence_chunk_ids(answer: str) -> list[str]:
    """从 answer 中提取所有 <evidence> 标签中的 chunk_id。

    格式: <evidence>chunk_id|p页码|o起止偏移</evidence>
    返回: ['chunk_id1', 'chunk_id2', ...]
    """
    if not isinstance(answer, str):
        return []
    import re
    pattern = re.compile(
        r"<evidence>([^<|]+)(?:\|[^<]*)?</evidence>",
        flags=re.IGNORECASE,
    )
    matches = pattern.findall(answer)
    return [chunk_id.strip() for chunk_id in matches if chunk_id.strip()]



def extract_evidence_doc_uids(answer: str) -> list[str]:
    """从 answer 中提取文档级引用，如 arxiv:2310.11511|p1|o0-10。"""
    if not isinstance(answer, str):
        return []
    import re

    references: set[str] = set()
    inline_pattern = re.compile(
        r"(?<![\w/])([A-Za-z0-9_.-]+:[^|\s<>\]]+)(?=\|p\d+)",
        flags=re.IGNORECASE,
    )
    for raw_ref in inline_pattern.findall(answer):
        ref = raw_ref.strip()
        if not ref:
            continue
        if ":chunk_" in ref:
            ref = ref.split(":chunk_", 1)[0].strip()
        if ref:
            references.add(ref)

    for chunk_id in extract_evidence_chunk_ids(answer):
        ref = chunk_id.split(":chunk_", 1)[0].strip() if ":chunk_" in chunk_id else chunk_id.strip()
        if ref:
            references.add(ref)
    return sorted(references)


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
) -> TurnCoreResult:
    if leader_agent is None:
        raise ValueError("Leader agent is not initialized")

    event_logs: list[TraceEvent] = []
    phase_labels: list[str] = []

    def _collect_event(item: TraceEvent) -> None:
        logger.info(f"_collect_event called: performative={item.get('performative')}, content={item.get('content')}")
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

    # 构建 config,传递必要参数给 middleware
    config = leader_runtime_config if isinstance(leader_runtime_config, dict) else {}
    if "configurable" not in config:
        config["configurable"] = {}

    # 传递 on_event 回调给 middleware
    config["configurable"]["on_event"] = _collect_event
    # 传递 llm 给 OrchestrationMiddleware
    if leader_llm is not None:
        config["configurable"]["llm"] = leader_llm
    # 确保有 thread_id 用于 session 隔离
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = "default"

    # 直接调用 leader_agent
    result = leader_agent.invoke(
        {"messages": [{"role": "user", "content": hinted_prompt}]},
        config=config,
    )

    # 提取 answer
    answer = ""
    messages: list[Any] = []
    if isinstance(result, dict):
        raw_messages = result.get("messages", [])
        messages = raw_messages if isinstance(raw_messages, list) else []
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                answer = str(last_msg.content)
            elif isinstance(last_msg, dict):
                answer = str(last_msg.get("content", ""))

    if not answer:
        logger.warning("Empty answer from agent execution")
        answer = "抱歉，我暂时没有生成有效回复。"

    # 从result中提取信息（如果有的话）
    trace_payload = event_logs
    plan_payload = result.get("plan") if isinstance(result, dict) else None
    runtime_state_payload = result.get("runtime_state") if isinstance(result, dict) else None

    # 检测是否使用了document RAG
    used_document_rag = False
    for msg in messages:
        for call in _message_tool_calls(msg):
            if call.get("name") == "search_document":
                used_document_rag = True
                break
        if used_document_rag:
            break

    # 从 answer 中提取 agent 引用的 chunk_id
    referenced_chunk_ids = extract_evidence_chunk_ids(answer)
    referenced_doc_uids = extract_evidence_doc_uids(answer)

    evidence_items = _extract_search_document_evidence_items(
        messages,
        referenced_chunk_ids=referenced_chunk_ids,
        referenced_doc_uids=referenced_doc_uids,
    )
    if (
        not evidence_items
        and (referenced_chunk_ids or referenced_doc_uids)
        and used_document_rag
        and callable(search_document_evidence_fn)
    ):
        try:
            # 获取所有相关证据
            evidence_payload = search_document_evidence_fn(prompt)
            all_evidence = normalize_evidence_items(evidence_payload)

            # 筛选出 agent 实际引用的证据
            referenced_chunk_set = {item.strip() for item in referenced_chunk_ids if item.strip()}
            referenced_doc_set = {item.strip() for item in referenced_doc_uids if item.strip()}
            for item in all_evidence:
                chunk_id = str(item.get("chunk_id", "")).strip()
                doc_uid = str(item.get("doc_uid", "")).strip()
                chunk_doc_uid = chunk_id.split(":chunk_", 1)[0].strip() if ":chunk_" in chunk_id else ""
                if (
                    chunk_id in referenced_chunk_set
                    or (doc_uid and doc_uid in referenced_doc_set)
                    or (chunk_doc_uid and chunk_doc_uid in referenced_doc_set)
                ):
                    evidence_items.append(item)
        except Exception:
            evidence_items = []
    method_compare_data = parse_method_compare_payload(answer)
    mindmap_data = try_parse_mindmap(answer)
    run_latency_ms = (time.perf_counter() - run_started) * 1000.0
    phase_path = _stable_phase_path(phase_labels=phase_labels, answer=answer, messages=messages)

    # 从 result 中提取 middleware 添加的 state
    todos = result.get("todos", []) if isinstance(result, dict) else []
    agent_plan = result.get("agent_plan") if isinstance(result, dict) else None

    return {
        "answer": answer,
        "policy_decision": {"plan_enabled": False, "team_enabled": False, "reason": "middleware-based"},
        "team_execution": {"enabled": False, "rounds": 0},
        "trace_payload": trace_payload,
        "plan": _maybe_to_dict(plan_payload),
        "runtime_state": _maybe_to_dict(runtime_state_payload),
        "evidence_items": evidence_items,
        "mindmap_data": mindmap_data,
        "method_compare_data": method_compare_data,
        "run_latency_ms": run_latency_ms,
        "team_rounds": 0,
        "phase_path": phase_path,
        "used_document_rag": used_document_rag,
        "ask_human_requests": [],
        "todos": todos,
        "agent_plan": agent_plan,
        "leader_tool_names": registered_tool_names,
        "output_messages": messages if isinstance(messages, list) else [],
    }
