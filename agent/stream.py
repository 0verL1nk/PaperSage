import json
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
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
    return ""


def _message_attr(message: Any, key: str, default: Any = "") -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _message_text(message: Any) -> str:
    return _content_to_text(_message_attr(message, "content", ""))


def extract_result_text(result: dict[str, Any]) -> str:
    raw_messages = result.get("messages", [])
    if not isinstance(raw_messages, list):
        return "抱歉，我暂时没有生成有效回复。"

    for message in reversed(raw_messages):
        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type not in {"ai", "assistant"} and role != "assistant":
            continue
        text = _message_text(message)
        if text:
            return text
    return "抱歉，我暂时没有生成有效回复。"


def extract_tool_names_from_result(result: Any) -> set[str]:
    raw_messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(raw_messages, list):
        return set()

    tool_names: set[str] = set()
    for message in raw_messages:
        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    name = str(call.get("name") or "").strip()
                else:
                    name = str(getattr(call, "name", "") or "").strip()
                if name:
                    tool_names.add(name)

        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type == "tool" or role == "tool":
            name = str(_message_attr(message, "name", "") or "").strip()
            if name:
                tool_names.add(name)

    return tool_names


def _iter_result_messages(result: Any) -> list[Any]:
    raw_messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(raw_messages, list):
        return []
    return raw_messages


def _extract_skill_name_from_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed
        except Exception:
            return ""
    if not isinstance(args, dict):
        return ""
    for key in ("skill_name", "name", "skill"):
        value = str(args.get(key) or "").strip()
        if value:
            return value
    return ""


def _extract_tool_name_from_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed
        except Exception:
            return ""
    if not isinstance(args, dict):
        return ""
    value = str(args.get("tool_name") or args.get("name") or "").strip()
    return value


def _extract_mode_name_from_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            args = parsed
        except Exception:
            return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("mode") or "").strip().lower()


def extract_tool_trace_events_from_result(result: Any) -> list[dict[str, str]]:
    messages = _iter_result_messages(result)
    events: list[dict[str, str]] = []

    for message in messages:
        content = _message_text(message)
        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    call_name = str(call.get("name") or "tool").strip() or "tool"
                    args = call.get("args", {})
                else:
                    call_name = str(getattr(call, "name", "tool") or "tool").strip() or "tool"
                    args = getattr(call, "args", {})
                events.append(
                    {
                        "sender": "leader",
                        "receiver": call_name,
                        "performative": "tool_call",
                        "content": str(args) if args else content or "(tool call)",
                    }
                )

        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type == "tool" or role == "tool":
            tool_name = str(_message_attr(message, "name", "tool") or "tool").strip() or "tool"
            events.append(
                {
                    "sender": tool_name,
                    "receiver": "leader",
                    "performative": "tool_result",
                    "content": content or "(tool result)",
                }
            )
    return events


def extract_skill_activation_events_from_result(result: Any) -> list[dict[str, str]]:
    messages = _iter_result_messages(result)
    events: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _append(skill_name: str, content: str) -> None:
        normalized_name = str(skill_name or "").strip()
        if not normalized_name:
            return
        normalized_content = str(content or "").strip()
        dedupe_key = (normalized_name, normalized_content)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        events.append(
            {
                "sender": "leader",
                "receiver": f"skill:{normalized_name}",
                "performative": "skill_activate",
                "content": normalized_content or f"activate {normalized_name}",
            }
        )

    for message in messages:
        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    call_name = str(call.get("name") or "").strip()
                    args = call.get("args", {})
                else:
                    call_name = str(getattr(call, "name", "") or "").strip()
                    args = getattr(call, "args", {})
                if call_name != "use_skill":
                    continue
                skill_name = _extract_skill_name_from_args(args)
                _append(skill_name, str(args) if args else "")

        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type != "tool" and role != "tool":
            continue
        tool_name = str(_message_attr(message, "name", "") or "").strip()
        if tool_name != "use_skill":
            continue
        content = _message_text(message).strip()
        if not content:
            continue
        first_line = content.splitlines()[0].strip()
        if first_line.lower().startswith("skill:"):
            skill_name = first_line.split(":", 1)[1].strip()
            _append(skill_name, first_line)

    return events


def extract_tool_activation_events_from_result(result: Any) -> list[dict[str, str]]:
    messages = _iter_result_messages(result)
    events: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _append(tool_name: str, content: str) -> None:
        normalized_name = str(tool_name or "").strip()
        if not normalized_name:
            return
        normalized_content = str(content or "").strip()
        dedupe_key = (normalized_name, normalized_content)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        events.append(
            {
                "sender": "leader",
                "receiver": f"tool:{normalized_name}",
                "performative": "tool_activate",
                "content": normalized_content or f"activate {normalized_name}",
            }
        )

    for message in messages:
        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    call_name = str(call.get("name") or "").strip()
                    args = call.get("args", {})
                else:
                    call_name = str(getattr(call, "name", "") or "").strip()
                    args = getattr(call, "args", {})
                if call_name != "activate_tool":
                    continue
                tool_name = _extract_tool_name_from_args(args)
                _append(tool_name, str(args) if args else "")

        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type != "tool" and role != "tool":
            continue
        tool_name = str(_message_attr(message, "name", "") or "").strip()
        if tool_name != "activate_tool":
            continue
        content = _message_text(message).strip()
        if not content:
            continue
        try:
            payload = json.loads(content)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            target_name = str(payload.get("tool_name") or "").strip()
            _append(target_name, content)
            continue

    return events


def extract_mode_activation_events_from_result(result: Any) -> list[dict[str, str]]:
    """提取模式激活事件（已废弃：start_plan 和 start_team 工具已被移除）"""
    return []


def extract_ask_human_requests_from_result(result: Any) -> list[dict[str, str]]:
    raw_messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(raw_messages, list):
        return []

    requests: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def _append_request(question: str, context: str, urgency: str) -> None:
        q = str(question or "").strip()
        if not q:
            return
        c = str(context or "").strip()
        u = str(urgency or "normal").strip().lower()
        if u not in {"low", "normal", "high"}:
            u = "normal"
        key = (q, c, u)
        if key in seen:
            return
        seen.add(key)
        requests.append({"question": q, "context": c, "urgency": u})

    for message in raw_messages:
        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                if name != "ask_human":
                    continue
                args = call.get("args")
                if isinstance(args, dict):
                    _append_request(
                        str(args.get("question") or ""),
                        str(args.get("context") or ""),
                        str(args.get("urgency") or "normal"),
                    )

        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type != "tool" and role != "tool":
            continue
        name = str(_message_attr(message, "name", "") or "").strip()
        if name != "ask_human":
            continue
        content = _message_text(message).strip()
        if not content:
            continue
        try:
            payload = json.loads(content)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("type") or "").strip() not in {"ask_human", ""}:
            continue
        _append_request(
            str(payload.get("question") or ""),
            str(payload.get("context") or ""),
            str(payload.get("urgency") or "normal"),
        )

    return requests


def extract_stream_text(chunk: Any) -> str:
    candidate = chunk
    if isinstance(chunk, tuple) and len(chunk) > 0:
        candidate = chunk[0]

    msg_type = str(_message_attr(candidate, "type", "") or "").lower()
    role = str(_message_attr(candidate, "role", "") or "").lower()
    if msg_type == "tool" or role == "tool":
        return ""

    if _message_attr(candidate, "content", None) is not None:
        return _message_text(candidate)

    if isinstance(candidate, dict):
        messages = candidate.get("messages")
        if isinstance(messages, list) and messages:
            latest = messages[-1]
            latest_type = str(_message_attr(latest, "type", "") or "").lower()
            latest_role = str(_message_attr(latest, "role", "") or "").lower()
            if latest_type == "tool" or latest_role == "tool":
                return ""
            return _message_text(latest)

    return ""


def extract_trace_events_from_update(update: Any) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    message_candidates: list[Any] = []

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            maybe_messages = obj.get("messages")
            if isinstance(maybe_messages, list):
                message_candidates.extend(maybe_messages)
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    _collect(value)
            return
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _collect(item)

    _collect(update)
    for message in message_candidates:
        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        name = str(_message_attr(message, "name", "tool") or "tool")
        content = _message_text(message)

        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_name = str(call.get("name") or "tool")
                args = call.get("args", {})
                events.append(
                    {
                        "sender": "react_agent",
                        "receiver": call_name,
                        "performative": "tool_call",
                        "content": str(args) if args else content or "(tool call)",
                    }
                )

        if msg_type == "tool" or role == "tool":
            events.append(
                {
                    "sender": name,
                    "receiver": "react_agent",
                    "performative": "tool_result",
                    "content": content or "(tool result)",
                }
            )
            continue

        if msg_type in {"human", "user"} or role == "user":
            events.append(
                {
                    "sender": "user",
                    "receiver": "react_agent",
                    "performative": "request",
                    "content": content,
                }
            )
            continue

        if msg_type in {"ai", "assistant"} or role == "assistant":
            if content.strip():
                events.append(
                    {
                        "sender": "react_agent",
                        "receiver": "user",
                        "performative": "final",
                        "content": content,
                    }
                )
    return events


def iter_agent_response_deltas(
    agent: Any,
    messages: list[dict[str, str]],
    config: dict[str, Any] | None = None,
) -> Iterator[str]:
    logger.debug("Agent stream start: message_count=%s", len(messages))
    has_stream_output = False
    chunk_count = 0
    for chunk in agent.stream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        chunk_count += 1
        delta = extract_stream_text(chunk)
        if not delta:
            continue
        has_stream_output = True
        yield delta

    if has_stream_output:
        logger.debug("Agent stream finished with streamed chunks: raw_chunks=%s", chunk_count)
        return

    logger.debug("Agent stream yielded no text, fallback to invoke")
    result = agent.invoke({"messages": messages}, config=config)
    if isinstance(result, dict):
        yield extract_result_text(result)
        logger.debug("Agent invoke fallback returned dict result")
        return
    logger.warning("Agent invoke fallback returned unexpected type: %s", type(result))
    yield "抱歉，我暂时没有生成有效回复。"
