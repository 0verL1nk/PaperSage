"""Progressive tool disclosure middleware."""

import json
import os
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse


_TOOL_VISIBILITY_ATTR = "_progressive_tool_visibility"


def _env_flag(name: str, default: bool = False) -> bool:
    """Check if environment variable is set to true."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _message_attr(message: Any, key: str, default: Any = "") -> Any:
    """Get attribute from message (dict or object)."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _content_to_text(content: Any) -> str:
    """Convert message content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_tool_names_from_search_result(content: str) -> list[str]:
    """Extract tool names from search_tools result."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("type") == "tool_search_result":
            return [t["tool_name"] for t in data.get("tools", []) if isinstance(t, dict)]
    except Exception:
        pass
    return []


def _extract_activated_tool_names(messages: list[Any]) -> set[str]:
    """Extract activated tool names from message history."""
    activated: set[str] = set()
    for message in messages:
        msg_type = str(_message_attr(message, "type", "") or "").lower()
        role = str(_message_attr(message, "role", "") or "").lower()
        if msg_type != "tool" and role != "tool":
            continue
        tool_name = str(_message_attr(message, "name", "") or "").strip()
        if tool_name != "search_tools":
            continue
        raw_content = _content_to_text(_message_attr(message, "content", ""))
        if not raw_content.strip():
            continue
        tool_names = _extract_tool_names_from_search_result(raw_content)
        for t in tool_names:
            activated.add(t)
    return activated


class ProgressiveToolDisclosureMiddleware(AgentMiddleware):
    """Rebuild visible tools per-model-call based on activation history."""

    def __init__(self, *, fixed_tool_names: set[str], lazy_tool_names: set[str]) -> None:
        self.fixed_tool_names = set(fixed_tool_names)
        self.lazy_tool_names = set(lazy_tool_names)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Filter tools based on activation history."""
        all_tools = list(request.tools or [])
        if not all_tools:
            return handler(request)

        activated_names = _extract_activated_tool_names(list(request.messages or []))
        visible_names = (
            set(self.fixed_tool_names) | {"search_tools"} | (activated_names & self.lazy_tool_names)
        )

        filtered_tools: list[Any] = []
        for item in all_tools:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = str(getattr(item, "name", "") or "").strip()
            if not name:
                continue
            if name in visible_names:
                filtered_tools.append(item)

        if not filtered_tools:
            return handler(request)
        if len(filtered_tools) == len(all_tools):
            return handler(request)

        request_with_filtered_tools = request.override(tools=filtered_tools)
        return handler(request_with_filtered_tools)


def build_progressive_tool_middleware(tools: list[Any]) -> list[AgentMiddleware]:
    """Build progressive tool disclosure middleware from tool list."""
    progressive_enabled = _env_flag("AGENT_PROGRESSIVE_TOOL_DISCLOSURE", default=True)
    if not progressive_enabled:
        return []

    fixed_tool_names: set[str] = set()
    lazy_tool_names: set[str] = set()
    has_activation_tool = False

    for item in tools:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            visibility = str(item.get(_TOOL_VISIBILITY_ATTR) or "").strip().lower()
        else:
            name = str(getattr(item, "name", "") or "").strip()
            visibility = str(getattr(item, _TOOL_VISIBILITY_ATTR, "") or "").strip().lower()

        if not name:
            continue
        if name == "search_tools":
            has_activation_tool = True
            continue
        if visibility == "lazy":
            lazy_tool_names.add(name)
        else:
            fixed_tool_names.add(name)

    if not has_activation_tool or not lazy_tool_names:
        return []

    return [
        ProgressiveToolDisclosureMiddleware(
            fixed_tool_names=fixed_tool_names,
            lazy_tool_names=lazy_tool_names,
        )
    ]
