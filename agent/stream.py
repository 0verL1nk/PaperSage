from collections.abc import Iterator
from typing import Any


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


def extract_result_text(result: dict[str, Any]) -> str:
    raw_messages = result.get("messages", [])
    if not isinstance(raw_messages, list):
        return "抱歉，我暂时没有生成有效回复。"

    for message in reversed(raw_messages):
        if getattr(message, "type", "") != "ai":
            continue
        text = _content_to_text(getattr(message, "content", ""))
        if text:
            return text
    return "抱歉，我暂时没有生成有效回复。"


def extract_stream_text(chunk: Any) -> str:
    candidate = chunk
    if isinstance(chunk, tuple) and len(chunk) > 0:
        candidate = chunk[0]

    if hasattr(candidate, "type") and getattr(candidate, "type", "") == "tool":
        return ""

    if hasattr(candidate, "content"):
        return _content_to_text(getattr(candidate, "content", ""))

    if isinstance(candidate, dict):
        messages = candidate.get("messages")
        if isinstance(messages, list) and messages:
            latest = messages[-1]
            if hasattr(latest, "content"):
                if hasattr(latest, "type") and getattr(latest, "type", "") == "tool":
                    return ""
                return _content_to_text(getattr(latest, "content", ""))
            if isinstance(latest, dict):
                msg_type = latest.get("type") or latest.get("role")
                if msg_type == "tool":
                    return ""
                return _content_to_text(latest.get("content", ""))

    return ""


def iter_agent_response_deltas(
    agent: Any,
    messages: list[dict[str, str]],
    config: dict[str, Any] | None = None,
) -> Iterator[str]:
    has_stream_output = False
    for chunk in agent.stream(
        {"messages": messages},
        config=config,
        stream_mode="messages",
    ):
        delta = extract_stream_text(chunk)
        if not delta:
            continue
        has_stream_output = True
        yield delta

    if has_stream_output:
        return

    result = agent.invoke({"messages": messages}, config=config)
    if isinstance(result, dict):
        yield extract_result_text(result)
        return
    yield "抱歉，我暂时没有生成有效回复。"
