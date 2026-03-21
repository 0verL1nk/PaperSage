"""Mindmap output format guard middleware."""

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_MAX_REWRITE_ATTEMPTS = 1
_MAX_BAD_OUTPUT_PREVIEW = 600
_RETRY_PROMPT = """你刚才的思维导图输出格式不符合要求，必须重写。

错误要求说明：
- 必须只输出一个 `<mindmap>...</mindmap>` 包裹的 JSON 对象
- 禁止输出 Mermaid
- 禁止输出 Markdown 代码块
- 禁止输出标题、说明文字、前后缀或任何额外文本

正确格式示例：
<mindmap>
{{"name":"主题","children":[{{"name":"子主题","children":[]}}]}}
</mindmap>

你刚才的错误输出片段：
{bad_output}

请基于同一内容立即重写，并且只输出合法的 `<mindmap>...</mindmap>` 内容。"""

_FINAL_FAILURE_MESSAGE = (
    "思维导图输出格式校验失败：必须只输出 `<mindmap>...</mindmap>` 包裹的单个 JSON 对象，"
    "不能输出 Mermaid、Markdown 代码块或额外说明文字。请重试。"
)


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


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def _extract_use_skill_name(tool_call: dict[str, Any]) -> str:
    name = str(tool_call.get("name") or "").strip()
    if name != "use_skill":
        return ""
    args = tool_call.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("skill_name") or "").strip().lower()


def _expects_mindmap_response(messages: list[Any]) -> bool:
    for message in messages:
        role = str(getattr(message, "type", getattr(message, "role", "")) or "").strip().lower()
        content = _content_to_text(getattr(message, "content", ""))
        if isinstance(message, dict):
            role = str(message.get("type") or message.get("role") or "").strip().lower()
            content = _content_to_text(message.get("content", ""))
        for tool_call in _message_tool_calls(message):
            if _extract_use_skill_name(tool_call) == "mindmap":
                return True
        if role == "tool":
            name = str(getattr(message, "name", "") or "")
            if isinstance(message, dict):
                name = str(message.get("name") or "")
            if name == "use_skill" and "Skill: mindmap" in content:
                return True
    return False


def _extract_last_ai_message(messages: list[Any]) -> AIMessage | None:
    for item in reversed(messages):
        if isinstance(item, AIMessage):
            return item
    return None


def _parse_strict_mindmap_payload(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value.lower().startswith("<mindmap>") or not value.lower().endswith("</mindmap>"):
        return None
    prefix_len = len("<mindmap>")
    suffix_len = len("</mindmap>")
    inner = value[prefix_len:-suffix_len].strip()
    if not inner:
        return None
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(inner)
    except json.JSONDecodeError:
        return None
    if inner[end:].strip():
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("name"), str) or not str(payload.get("name")).strip():
        return None
    children = payload.get("children")
    if children is not None and not isinstance(children, list):
        return None
    return payload


def _looks_like_mindmap_output(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False
    if "<mindmap" in value:
        return True
    if "```mermaid" in value:
        return True
    if "```" in value and "mindmap" in value:
        return True
    if value.startswith("mindmap"):
        return True
    if "\nmindmap" in value:
        return True
    if "root((" in value or "root ((" in value:
        return True
    if ("思维导图" in value or "脑图" in value or "概念图" in value) and "```" in value:
        return True
    return False


def _preview_bad_output(text: str) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= _MAX_BAD_OUTPUT_PREVIEW:
        return collapsed
    return f"{collapsed[:_MAX_BAD_OUTPUT_PREVIEW]}..."


class MindmapFormatMiddleware(AgentMiddleware):
    """Retry mindmap responses until they satisfy the strict tagged JSON contract."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        current_request = request
        for attempt in range(_MAX_REWRITE_ATTEMPTS + 1):
            response = handler(current_request)
            ai_message = _extract_last_ai_message(list(response.result or []))
            if ai_message is None:
                return response
            if getattr(ai_message, "tool_calls", None):
                return response

            answer = _content_to_text(ai_message.content)
            if _parse_strict_mindmap_payload(answer) is not None:
                return response
            if not (
                _looks_like_mindmap_output(answer)
                or _expects_mindmap_response(list(current_request.messages or []))
            ):
                return response

            if attempt >= _MAX_REWRITE_ATTEMPTS:
                logger.warning("mindmap format validation failed after retry: %s", _preview_bad_output(answer))
                return ModelResponse(result=[AIMessage(content=_FINAL_FAILURE_MESSAGE)])

            logger.info("mindmap format validation failed, requesting rewrite: %s", _preview_bad_output(answer))
            rewrite_prompt = HumanMessage(
                content=_RETRY_PROMPT.format(bad_output=_preview_bad_output(answer) or "(empty)")
            )
            current_request = current_request.override(
                messages=[*list(current_request.messages or []), ai_message, rewrite_prompt]
            )

        return handler(current_request)


mindmap_format_middleware = MindmapFormatMiddleware()

__all__ = [
    "MindmapFormatMiddleware",
    "mindmap_format_middleware",
]
