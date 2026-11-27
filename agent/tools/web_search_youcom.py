"""You.com MCP you-search 提供者（可选，独立模块拆分）"""
import json
import logging
from typing import Any

import httpx

from .utils import (
    DEFAULT_WEB_MAX_RESULTS,
    DEFAULT_WEB_TIMEOUT_SECONDS,
    WIKIPEDIA_COMPLIANT_USER_AGENT,
    _env_flag,
    _env_value,
    _format_web_results,
    _load_secret,
)

logger = logging.getLogger(__name__)

# You.com MCP endpoints. The free profile works keyless (rate-limited); an
# API key switches the provider to the authenticated endpoint.
YOUCOM_KEYLESS_MCP_URL = "https://api.you.com/mcp?profile=free"
YOUCOM_AUTHENTICATED_MCP_URL = "https://api.you.com/mcp"

_YOUCOM_REQUEST_ID = 1


def _parse_youcom_mcp_payload(response_text: str) -> dict[str, Any]:
    """Return the JSON-RPC result object from an MCP HTTP response body.

    The endpoint answers either as an SSE stream (``event:``/``data:`` frames,
    notification frames first) or as a single JSON-RPC object. Frames that
    carry notifications instead of a result are skipped; a JSON-RPC error
    frame raises.
    """
    frames: list[Any] = []
    stripped = response_text.strip()
    if stripped.startswith("event:") or stripped.startswith("data:"):
        # LF/CRLF frame split only: str.splitlines() would shred JSON lines
        # containing U+0085/U+2028 separators inside result payloads.
        for raw_line in stripped.split("\n"):
            line = raw_line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            candidate = line[len("data:"):].strip()
            if not candidate:
                continue
            try:
                frames.append(json.loads(candidate))
            except json.JSONDecodeError:
                # Skip frames that are not JSON-RPC objects (keep-alives);
                # the result frame itself must parse below.
                continue
    else:
        try:
            frames.append(json.loads(stripped))
        except json.JSONDecodeError:
            raise RuntimeError("youcom invalid response body") from None

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        error = frame.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "unknown error")
            raise RuntimeError(f"youcom jsonrpc error: {message}")
        result = frame.get("result")
        if isinstance(result, dict):
            return result
    raise RuntimeError("youcom invalid payload")


def _extract_youcom_text(result: dict[str, Any]) -> str:
    """Return the first text block of a tools/call result payload."""
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    raise RuntimeError("youcom invalid payload")


def _normalize_youcom_web_items(text: str) -> list[dict[str, str]]:
    """Map a you-search JSON payload to the house title/url/description shape.

    Items missing ``description`` fall back to their first
    ``contents.highlights`` entry so evidence snippets stay usable.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError("youcom invalid payload") from None
    container = payload.get("results") if isinstance(payload, dict) else None
    items = container.get("web") if isinstance(container, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("youcom invalid payload")
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        description = str(item.get("description") or "").strip()
        if not description:
            contents = item.get("contents")
            highlights = contents.get("highlights") if isinstance(contents, dict) else None
            if isinstance(highlights, list) and highlights:
                description = str(highlights[0] or "").strip()
        normalized.append({"title": title, "url": url, "description": description})
    return normalized


def _build_youcom_web_search_client():
    """You.com you-search provider.

    Opt-in only: enable with AGENT_WEB_YOUCOM_ENABLED=1. Works keyless on the
    free profile (rate-limited); set YOUCOM_API_KEY to use an authenticated
    account. Override the endpoint with AGENT_YOUCOM_MCP_URL.
    """
    if not _env_flag("AGENT_WEB_YOUCOM_ENABLED", default=False):
        return None
    api_key = _load_secret("YOUCOM_API_KEY")
    default_url = YOUCOM_AUTHENTICATED_MCP_URL if api_key else YOUCOM_KEYLESS_MCP_URL
    search_url = _env_value("AGENT_YOUCOM_MCP_URL", default=default_url)

    class _YoucomWebSearch:
        def __init__(self, key: str, url: str) -> None:
            self.api_key = key
            self.search_url = url

        def run(self, query: str) -> str:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "User-Agent": WIKIPEDIA_COMPLIANT_USER_AGENT,
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = httpx.post(
                self.search_url,
                json={
                    "jsonrpc": "2.0",
                    "id": _YOUCOM_REQUEST_ID,
                    "method": "tools/call",
                    "params": {
                        "name": "you-search",
                        "arguments": {
                            "query": query,
                            "count": DEFAULT_WEB_MAX_RESULTS,
                        },
                    },
                },
                headers=headers,
                timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"youcom status={response.status_code}")
            result = _parse_youcom_mcp_payload(response.text)
            text = _extract_youcom_text(result)
            if result.get("isError") is True:
                raise RuntimeError(f"youcom tool error: {text[:200]}")
            items = _normalize_youcom_web_items(text)
            return _format_web_results(
                items,
                title_key="title",
                url_key="url",
                snippet_key="description",
            )

    return _YoucomWebSearch(api_key, search_url)

