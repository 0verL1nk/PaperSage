import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .scholarly_search import (
    ScholarlySearchError,
    format_search_papers_results,
    search_semantic_scholar,
)
from .skills.loader import build_skill_runtime_payload, discover_available_skills
from .tools import LOCAL_OPS_TOOL_METADATA, ToolMetadata, build_local_ops_tools

logger = logging.getLogger(__name__)


class SearchDocumentInput(BaseModel):
    query: str = Field(
        description="Specific paper question or keyword used to retrieve evidence snippets."
    )


class ListDocumentInput(BaseModel):
    verbose: bool = Field(
        default=False,
        description="If True, include character length for each document.",
    )


class ReadDocumentInput(BaseModel):
    offset: int = Field(
        default=0,
        description="Character offset to start reading from (0 means start from beginning).",
    )
    limit: int = Field(
        default=2000,
        description="Maximum number of characters to read (recommended 1000-3000 for context).",
    )
    include_rag: bool = Field(
        default=False,
        description="Whether to use RAG to retrieve relevant context around the reading position.",
    )


class SearchWebInput(BaseModel):
    query: str = Field(
        description="Supplemental public-web query used only when document evidence is insufficient."
    )


class SearchPapersInput(BaseModel):
    query: str = Field(
        description="Academic search query for finding relevant papers from scholarly sources."
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of papers to return (1-20).",
    )


class SkillInput(BaseModel):
    skill_name: str = Field(
        description=(
            "Skill template name. Available skills include: summary, critical_reading, "
            "method_compare, translation, mindmap, agentic_search."
        )
    )
    task: str = Field(
        description="Current user task where the selected skill guidance should be applied."
    )


class StartPlanInput(BaseModel):
    goal: str = Field(description="Planning goal or sub-goal that requires a structured step plan.")
    reason: str = Field(
        default="",
        description="Optional short reason explaining why plan mode is needed.",
    )


class StartTeamInput(BaseModel):
    goal: str = Field(
        description="Collaboration goal that should be delegated to a sub-agent team."
    )
    reason: str = Field(
        default="",
        description="Optional short reason explaining why team collaboration is needed.",
    )
    roles_hint: str = Field(
        default="",
        description="Optional comma-separated role hints for the team.",
    )


class SearchToolsInput(BaseModel):
    query: str = Field(
        description="Search query describing the needed capability (e.g., 'read pdf', 'web search')."
    )
    reason: str = Field(
        default="",
        description="Optional short reason for activation.",
    )


DEFAULT_MAX_QUERY_CHARS = 1200
DEFAULT_WEB_MAX_RESULTS = 5
DEFAULT_WEB_TIMEOUT_SECONDS = 8.0
DEFAULT_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_SEARXNG_INSTANCES = (
    "https://searx.be",
    "https://search.inetol.net",
    "https://opnxng.com",
)
_DANGEROUS_QUERY_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"reveal\s+.*(api\s*key|token|password)",
    r"(api\s*key|access\s*token|password)\s*[:=]",
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bssh\b",
]
_DEFAULT_FIXED_TOOL_NAMES = {
    "search_document",
    "ask_human",
}
_TOOL_VISIBILITY_ATTR = "_progressive_tool_visibility"
_TOOL_METADATA = (
    ToolMetadata(
        name="search_document",
        description="Search uploaded paper content for relevant evidence snippets using RAG.",
    ),
    ToolMetadata(
        name="list_document",
        description="List all documents loaded in the current project scope with their names and identifiers.",
    ),
    ToolMetadata(
        name="read_document",
        description=(
            "Read a specific portion of the uploaded paper with pagination. "
            "Use offset to skip to a position, limit to control chunk size. "
            "Set include_rag=True to get relevant context around the position."
        ),
    ),
    *LOCAL_OPS_TOOL_METADATA,
    ToolMetadata(
        name="search_web",
        description="Search public web content for supplemental context.",
    ),
    ToolMetadata(
        name="search_papers",
        description="Search academic papers from scholarly providers when document evidence is insufficient.",
    ),
    ToolMetadata(
        name="use_skill",
        description="Apply a named skill template to the current task and return operational guidance.",
    ),
    ToolMetadata(
        name="start_plan",
        description="Request structured planning runtime for a goal when a direct answer is insufficient.",
    ),
    ToolMetadata(
        name="start_team",
        description="Request sub-agent team runtime for parallel analysis or cross-checking.",
    ),
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return default


def _read_from_dotenv(name: str, dotenv_path: str = ".env") -> str:
    try:
        with open(dotenv_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                if key.strip() != name:
                    continue
                value = raw_value.strip()
                if not value:
                    return ""
                if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                    value = value[1:-1]
                return value.strip()
    except Exception:
        return ""
    return ""


def _load_secret(name: str) -> str:
    value = _env_value(name, default="")
    if value:
        return value
    return _read_from_dotenv(name)


def _sanitize_query(query: str) -> str:
    value = query.strip()
    if len(value) > DEFAULT_MAX_QUERY_CHARS:
        value = value[:DEFAULT_MAX_QUERY_CHARS]
    return value


def _preview(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


def _is_dangerous_query(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in _DANGEROUS_QUERY_PATTERNS)


def _tool_enabled(tool_name: str, allowed_tools: set[str] | None = None) -> bool:
    normalized = tool_name.strip().lower()
    if allowed_tools is not None and normalized not in allowed_tools:
        return False
    env_key = f"AGENT_DISABLE_{normalized.upper()}"
    return not _env_flag(env_key, default=False)


def _parse_tool_name_set(raw_value: str) -> set[str]:
    items = [part.strip().lower() for part in str(raw_value or "").split(",")]
    return {item for item in items if item}


def _resolve_fixed_tool_names(*, enabled_tool_names: set[str]) -> set[str]:
    configured = _env_value("AGENT_FIXED_TOOLS", default="")
    if configured:
        candidates = _parse_tool_name_set(configured)
    else:
        candidates = set(_DEFAULT_FIXED_TOOL_NAMES)
    return {name for name in candidates if name in enabled_tool_names}


def _schema_manifest_for_tool(tool_obj: Any) -> dict[str, Any]:
    args_schema = getattr(tool_obj, "args_schema", None)
    if args_schema is None or not hasattr(args_schema, "model_json_schema"):
        return {"type": "object", "fields": [], "required": []}
    try:
        schema_obj = args_schema.model_json_schema()
    except Exception:
        return {"type": "object", "fields": [], "required": []}
    if not isinstance(schema_obj, dict):
        return {"type": "object", "fields": [], "required": []}
    properties = schema_obj.get("properties")
    fields: list[str] = []
    if isinstance(properties, dict):
        fields = sorted(str(key).strip() for key in properties.keys() if str(key).strip())
    required = schema_obj.get("required")
    required_fields: list[str] = []
    if isinstance(required, list):
        required_fields = sorted(str(item).strip() for item in required if str(item).strip())
    return {
        "type": str(schema_obj.get("type") or "object"),
        "fields": fields,
        "required": required_fields,
    }


def _message_attr(message: Any, key: str, default: Any = "") -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


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
            elif hasattr(item, "text"):
                text = getattr(item, "text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_tool_names_from_search_result(content: str) -> list[str]:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("type") == "tool_search_result":
            return [
                t["tool_name"]
                for t in data.get("tools", [])
                if isinstance(t, dict) and "tool_name" in t
            ]
    except Exception:
        pass
    return []


def _extract_activated_tool_names(messages: list[Any]) -> set[str]:
    activated: set[str] = set()
    for message in messages:
        tool_calls = _message_attr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    _call_name = str(call.get("name") or "").strip()
                    _args = call.get("args", {})
                else:
                    _call_name = str(getattr(call, "name", "") or "").strip()
                    _args = getattr(call, "args", {})

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


def discover_available_tools(
    *, read_document_enabled: bool = True, list_document_enabled: bool = True
) -> list[ToolMetadata]:
    """发现可用工具（元信息阶段，不创建运行时实例）"""
    discovered: list[ToolMetadata] = []
    for item in _TOOL_METADATA:
        if item.name == "read_document" and not read_document_enabled:
            continue
        if item.name == "list_document" and not list_document_enabled:
            continue
        discovered.append(item)
    return discovered


def _get_skill_options() -> str:
    """获取可用的 skill 列表"""
    skills = discover_available_skills()
    if skills:
        return ", ".join(sorted(s.name for s in skills))
    # 回退到旧的行为
    return "summary, critical_reading, method_compare, translation, mindmap"


def _build_native_web_search_client():
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return None

    class _NativeDuckDuckGoSearch:
        def run(self, query: str) -> str:
            with DDGS(timeout=int(DEFAULT_WEB_TIMEOUT_SECONDS)) as client:
                results = client.text(query, max_results=DEFAULT_WEB_MAX_RESULTS)
            return _format_web_results(
                results, title_key="title", url_key="href", snippet_key="body"
            )

    return _NativeDuckDuckGoSearch()


def _parse_searxng_instances() -> list[str]:
    configured = str(os.getenv("AGENT_SEARXNG_BASE_URLS", "") or "").strip()
    if configured:
        items = [item.strip().rstrip("/") for item in configured.split(",")]
        return [item for item in items if item]
    return [item.rstrip("/") for item in DEFAULT_SEARXNG_INSTANCES]


def _format_web_results(
    results: Any,
    *,
    title_key: str,
    url_key: str,
    snippet_key: str,
) -> str:
    if not isinstance(results, list) or not results:
        return "No web search results found."
    lines: list[str] = []
    for idx, item in enumerate(results[:DEFAULT_WEB_MAX_RESULTS], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get(title_key) or "").strip()
        href = str(item.get(url_key) or "").strip()
        body = str(item.get(snippet_key) or "").strip()
        snippet = _preview(body, limit=180) if body else ""
        lines.append(
            f"{idx}. {title or 'Untitled'}\nURL: {href or 'n/a'}\nSnippet: {snippet or '-'}"
        )
    return "\n\n".join(lines) if lines else "No web search results found."


def _build_searxng_web_search_client():
    instances = _parse_searxng_instances()
    if not instances:
        return None

    class _SearxngWebSearch:
        def __init__(self, base_urls: list[str]):
            self.base_urls = base_urls

        def run(self, query: str) -> str:
            last_error = "unknown error"
            for base_url in self.base_urls:
                try:
                    response = httpx.get(
                        f"{base_url}/search",
                        params={
                            "q": query,
                            "format": "json",
                            "safesearch": "1",
                        },
                        headers={"User-Agent": "llm-app/1.0 (+search_web)"},
                        timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
                    )
                    if response.status_code >= 400:
                        last_error = f"{base_url} status={response.status_code}"
                        continue
                    payload = response.json()
                    if not isinstance(payload, dict):
                        last_error = f"{base_url} invalid json payload"
                        continue
                    results = payload.get("results")
                    rendered = _format_web_results(
                        results,
                        title_key="title",
                        url_key="url",
                        snippet_key="content",
                    )
                    if rendered != "No web search results found.":
                        return rendered
                    last_error = f"{base_url} no results"
                except Exception as exc:
                    last_error = f"{base_url} {exc}"
                    continue
            raise RuntimeError(f"SearXNG unavailable: {last_error}")

    return _SearxngWebSearch(instances)


def _build_wikipedia_web_search_client():
    class _WikipediaWebSearch:
        def run(self, query: str) -> str:
            response = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": DEFAULT_WEB_MAX_RESULTS,
                },
                headers={"User-Agent": "llm-app/1.0 (+search_web)"},
                timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"wikipedia status={response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("wikipedia invalid payload")
            query_block = payload.get("query")
            search_items = query_block.get("search") if isinstance(query_block, dict) else None
            if not isinstance(search_items, list):
                return "No web search results found."
            normalized: list[dict[str, str]] = []
            for item in search_items[:DEFAULT_WEB_MAX_RESULTS]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                snippet = re.sub(r"<[^>]+>", "", snippet)
                pageid = item.get("pageid")
                if isinstance(pageid, int):
                    url = f"https://en.wikipedia.org/?curid={pageid}"
                else:
                    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                normalized.append(
                    {
                        "title": title or "Untitled",
                        "url": url,
                        "content": snippet,
                    }
                )
            return _format_web_results(
                normalized,
                title_key="title",
                url_key="url",
                snippet_key="content",
            )

    return _WikipediaWebSearch()


def _build_brave_web_search_client():
    api_key = _load_secret("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return None
    search_url = _env_value("BRAVE_SEARCH_URL", default=DEFAULT_BRAVE_SEARCH_URL)

    class _BraveWebSearch:
        def __init__(self, key: str, url: str):
            self.api_key = key
            self.search_url = url

        def run(self, query: str) -> str:
            response = httpx.get(
                self.search_url,
                params={
                    "q": query,
                    "count": DEFAULT_WEB_MAX_RESULTS,
                },
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                    "User-Agent": "llm-app/1.0 (+search_web)",
                },
                timeout=DEFAULT_WEB_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"brave status={response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("brave invalid payload")
            web_block = payload.get("web")
            results = web_block.get("results") if isinstance(web_block, dict) else None
            return _format_web_results(
                results,
                title_key="title",
                url_key="url",
                snippet_key="description",
            )

    return _BraveWebSearch(api_key, search_url)


def build_agent_tools(
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
    list_documents_fn: Callable[[], list[dict[str, Any]]] | None = None,
    allowed_tools: set[str] | None = None,
):
    """构建 Agent 工具集

    Args:
        search_document_fn: RAG 检索函数
        read_document_fn: 分块读取函数，接收 (offset, limit)，返回 (content, total_length)
        list_documents_fn: 文档列表函数，返回 list[{doc_uid, doc_name, char_length}]
    """
    if allowed_tools is None:
        normalized_allowlist = None
    else:
        normalized_allowlist = {name.strip().lower() for name in allowed_tools if name.strip()}

    discovered_tools = discover_available_tools(
        read_document_enabled=read_document_fn is not None,
        list_document_enabled=list_documents_fn is not None,
    )
    enabled_tool_names = {
        item.name for item in discovered_tools if _tool_enabled(item.name, normalized_allowlist)
    }

    # 渐进式加载：web provider 在首次调用 web search 时再初始化。
    web_search_clients: list[tuple[str, Any]] | None = None

    def _ensure_web_search_clients() -> list[tuple[str, Any]]:
        nonlocal web_search_clients
        if web_search_clients is not None:
            return web_search_clients

        clients: list[tuple[str, Any]] = []

        brave_client = _build_brave_web_search_client()
        if brave_client is not None:
            clients.append(("brave_search_api", brave_client))
            logger.info("tool.search_web provider initialized: brave_search_api")

        searxng_client = _build_searxng_web_search_client()
        if searxng_client is not None:
            clients.append(("searxng_public_pool", searxng_client))
            logger.info("tool.search_web provider initialized: searxng_public_pool")

        wikipedia_client = _build_wikipedia_web_search_client()
        if wikipedia_client is not None:
            clients.append(("wikipedia_api", wikipedia_client))
            logger.info("tool.search_web provider initialized: wikipedia_api")

        if not clients:
            logger.warning("tool.search_web no primary provider initialized")

        if not clients:
            allow_ddg_fallback = _env_flag("AGENT_WEB_ENABLE_DDG_FALLBACK", default=False)
            if allow_ddg_fallback:
                fallback_client = _build_native_web_search_client()
                if fallback_client is not None:
                    clients.append(("native_duckduckgo_search", fallback_client))
                    logger.info(
                        "tool.search_web provider fallback initialized: native_duckduckgo_search"
                    )
                else:
                    try:
                        fallback_client = DuckDuckGoSearchRun(name="search_web")
                        clients.append(("langchain_duckduckgo", fallback_client))
                        logger.info(
                            "tool.search_web provider fallback initialized: langchain_duckduckgo"
                        )
                    except Exception as exc:
                        logger.warning("tool.search_web ddg fallback unavailable: %s", exc)
            else:
                logger.warning(
                    "tool.search_web provider unavailable: searxng disabled/unreachable and ddg fallback disabled"
                )

        web_search_clients = clients
        return web_search_clients

    def _run_web_search_internal(query: str) -> tuple[str | None, str | None, str | None]:
        clients = _ensure_web_search_clients()
        if not clients:
            return (
                None,
                None,
                (
                    "Web search is unavailable in current environment. "
                    "Set BRAVE_SEARCH_API_KEY in .env, or configure AGENT_SEARXNG_BASE_URLS, "
                    "or enable AGENT_WEB_ENABLE_DDG_FALLBACK=1."
                ),
            )
        errors: list[str] = []
        for provider_name, provider in clients:
            try:
                response = provider.run(query)
                response_text = response if isinstance(response, str) else str(response)
                if not response_text or response_text.strip() == "No web search results found.":
                    errors.append(f"{provider_name}: no results")
                    continue
                return provider_name, response_text, None
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                logger.info("tool.search_web provider failed: %s (%s)", provider_name, exc)
        return None, None, f"Web search failed: {' | '.join(errors)}"

    def _run_paper_search_internal(
        query: str, limit: int
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            papers = search_semantic_scholar(query=query, limit=limit)
        except ScholarlySearchError as exc:
            logger.warning("tool.search_papers failed: %s", exc)
            return [], f"Academic search failed: {exc}"
        return papers, None

    runtime_tools: list[Any] = []
    runtime_tool_map: dict[str, Any] = {}

    def _register_tool(tool_obj: Any) -> None:
        tool_name = str(getattr(tool_obj, "name", "") or "").strip()
        if not tool_name:
            return
        runtime_tools.append(tool_obj)
        runtime_tool_map[tool_name] = tool_obj

    if "search_document" in enabled_tool_names:

        @tool(
            "search_document",
            description="Search uploaded paper content for relevant evidence snippets using RAG.",
            args_schema=SearchDocumentInput,
        )
        def search_document(query: str) -> str:
            safe_query = _sanitize_query(query)
            logger.info(
                "tool.search_document called: query_len=%s query_preview=%s",
                len(safe_query),
                _preview(safe_query),
            )
            if not safe_query:
                logger.warning("tool.search_document blocked: empty query after sanitization")
                return "Document search query is empty after sanitization."
            if _is_dangerous_query(safe_query):
                logger.warning("tool.search_document blocked by policy")
                return "Blocked by tool policy: query appears unsafe for document search."
            if search_document_evidence_fn is not None:
                try:
                    evidence_payload = search_document_evidence_fn(safe_query)
                    evidence_count = (
                        len(evidence_payload.get("evidences", []))
                        if isinstance(evidence_payload, dict)
                        else 0
                    )
                    logger.info(
                        "tool.search_document success: mode=evidence_json evidences=%s",
                        evidence_count,
                    )
                    return json.dumps(evidence_payload, ensure_ascii=False)
                except Exception:
                    logger.exception("tool.search_document evidence function failed, fallback=text")
                    # 回退到文本输出，避免工具失败影响主流程
                    return search_document_fn(safe_query)
            logger.info("tool.search_document success: mode=text")
            return search_document_fn(safe_query)

        _register_tool(search_document)

    if "list_document" in enabled_tool_names and list_documents_fn is not None:

        @tool(
            "list_document",
            description="List all documents loaded in the current project scope with their names and identifiers.",
            args_schema=ListDocumentInput,
        )
        def list_document(verbose: bool = False) -> str:
            logger.info("tool.list_document called: verbose=%s", verbose)
            try:
                docs = list_documents_fn()
            except Exception as exc:
                logger.exception("tool.list_document failed: %s", exc)
                return f"Failed to list documents: {exc}"
            if not isinstance(docs, list) or not docs:
                logger.info("tool.list_document: no documents found")
                return "No documents loaded in current project scope."
            items: list[dict[str, Any]] = []
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                entry: dict[str, Any] = {
                    "doc_uid": str(doc.get("doc_uid") or doc.get("uid") or ""),
                    "doc_name": str(doc.get("doc_name") or doc.get("file_name") or ""),
                }
                if verbose:
                    text = doc.get("text") or ""
                    entry["char_length"] = len(str(text))
                items.append(entry)
            logger.info("tool.list_document success: count=%s", len(items))
            return json.dumps({"count": len(items), "documents": items}, ensure_ascii=False)

        _register_tool(list_document)

    for local_tool in build_local_ops_tools(enabled_tool_names=enabled_tool_names):
        _register_tool(local_tool)

    if "read_document" in enabled_tool_names and read_document_fn is not None:

        @tool(
            "read_document",
            description="Read a specific portion of the uploaded paper with pagination. "
            "Use offset to skip to a position, limit to control chunk size. "
            "Set include_rag=True to get relevant context around the position.",
            args_schema=ReadDocumentInput,
        )
        def read_document(offset: int = 0, limit: int = 2000, include_rag: bool = False) -> str:
            logger.info(
                "tool.read_document called: offset=%s limit=%s include_rag=%s",
                offset,
                limit,
                include_rag,
            )
            content, total = read_document_fn(offset, limit)

            # 如果启用 RAG，获取相关上下文
            rag_context = ""
            if include_rag:
                # 基于位置获取相关段落
                # 计算当前位置的大致位置（用于检索）
                query = f"position_{offset}"
                rag_context = search_document_fn(query)

            result = f"=== 文档阅读 (字符位置 {offset} - {offset + limit}) ===\n"
            result += f"总长度: {total} 字符\n"
            result += f"当前 chunk: {len(content)} 字符\n\n"
            if rag_context:
                result += f"=== 相关上下文 (RAG) ===\n{rag_context}\n\n"
            result += f"=== 内容 ===\n{content}"
            logger.info(
                "tool.read_document success: chunk_len=%s total_len=%s", len(content), total
            )
            return result

        _register_tool(read_document)

    if "search_web" in enabled_tool_names:

        @tool(
            "search_web",
            description="Search public web content for supplemental context.",
            args_schema=SearchWebInput,
        )
        def search_web(query: str) -> str:
            safe_query = _sanitize_query(query)
            logger.info(
                "tool.search_web called: query_len=%s query_preview=%s",
                len(safe_query),
                _preview(safe_query),
            )
            if not safe_query:
                logger.warning("tool.search_web blocked: empty query after sanitization")
                return "Web search query is empty after sanitization."
            if _is_dangerous_query(safe_query):
                logger.warning("tool.search_web blocked by policy")
                return "Blocked by tool policy: query appears unsafe for web search."
            provider_name, response_text, error_text = _run_web_search_internal(safe_query)
            if response_text is None:
                logger.warning("tool.search_web unavailable: %s", error_text)
                return str(error_text or "Web search failed.")
            logger.info(
                "tool.search_web success: provider=%s response_len=%s",
                provider_name,
                len(response_text),
            )
            return response_text

        _register_tool(search_web)

    if "search_papers" in enabled_tool_names:

        @tool(
            "search_papers",
            description="Search academic papers from scholarly providers when document evidence is insufficient.",
            args_schema=SearchPapersInput,
        )
        def search_papers(query: str, limit: int = 5) -> str:
            safe_query = _sanitize_query(query)
            logger.info(
                "tool.search_papers called: limit=%s query_len=%s query_preview=%s",
                limit,
                len(safe_query),
                _preview(safe_query),
            )
            if not safe_query:
                logger.warning("tool.search_papers blocked: empty query after sanitization")
                return "Academic search query is empty after sanitization."
            if _is_dangerous_query(safe_query):
                logger.warning("tool.search_papers blocked by policy")
                return "Blocked by tool policy: query appears unsafe for academic search."
            papers, error_text = _run_paper_search_internal(safe_query, limit)
            if error_text is not None:
                return f"{error_text} Try narrowing the topic or use search_web."
            logger.info("tool.search_papers success: results=%s", len(papers))
            return format_search_papers_results(papers)

        _register_tool(search_papers)

    if "use_skill" in enabled_tool_names:

        @tool(
            "use_skill",
            description="Apply a named skill template to the current task and return operational guidance.",
            args_schema=SkillInput,
        )
        def use_skill(skill_name: str, task: str) -> str:
            normalized_name = skill_name.strip().lower()
            logger.info("tool.use_skill called: skill=%s task_len=%s", normalized_name, len(task))
            if _is_dangerous_query(task):
                logger.warning("tool.use_skill blocked by policy")
                return "Blocked by tool policy: task appears unsafe."

            runtime_payload = build_skill_runtime_payload(
                normalized_name,
                task=task,
                max_references=2,
                reference_char_limit=1800,
            )
            if runtime_payload is not None:
                logger.info("tool.use_skill success: skill=%s", normalized_name)
                parts: list[str] = [
                    f"Skill: {runtime_payload['name']}",
                    f"Description: {runtime_payload['description']}",
                    "",
                    "Instructions:",
                    str(runtime_payload["instructions"]),
                ]

                references = runtime_payload.get("references", [])
                if isinstance(references, list) and references:
                    parts.extend(["", "Selected references:"])
                    for item in references:
                        if not isinstance(item, dict):
                            continue
                        path_value = str(item.get("path") or "").strip()
                        content_value = str(item.get("content") or "").strip()
                        if not path_value or not content_value:
                            continue
                        parts.append(f"- {path_value}")
                        parts.append(content_value)

                scripts = runtime_payload.get("scripts", [])
                if isinstance(scripts, list) and scripts:
                    parts.extend(["", "Available scripts:"])
                    parts.extend(f"- {str(item)}" for item in scripts if str(item).strip())

                agent_metadata = runtime_payload.get("agent_metadata")
                if isinstance(agent_metadata, str) and agent_metadata.strip():
                    parts.extend(["", f"Agent metadata: {agent_metadata}"])

                parts.extend(["", f"Task: {task}"])
                return "\n".join(parts)

            # 如果没找到，返回错误信息
            options = _get_skill_options()
            logger.warning("tool.use_skill unknown skill: %s", normalized_name)
            return f"Unknown skill '{skill_name}'. Available skills: {options}."

        _register_tool(use_skill)

    if "start_plan" in enabled_tool_names:

        @tool(
            "start_plan",
            description="Request structured planning runtime for a goal when direct answering is insufficient.",
            args_schema=StartPlanInput,
        )
        def start_plan(goal: str, reason: str = "") -> str:
            normalized_goal = str(goal or "").strip()
            if not normalized_goal:
                return "goal is required to start plan mode."
            payload = {
                "type": "mode_activate",
                "mode": "plan",
                "goal": normalized_goal,
                "reason": str(reason or "").strip(),
            }
            logger.info(
                "tool.start_plan called: goal_len=%s reason=%s",
                len(normalized_goal),
                _preview(str(reason or "")),
            )
            return json.dumps(payload, ensure_ascii=False)

        _register_tool(start_plan)

    if "start_team" in enabled_tool_names:

        @tool(
            "start_team",
            description="Request sub-agent team runtime for parallel analysis or cross-checking.",
            args_schema=StartTeamInput,
        )
        def start_team(goal: str, reason: str = "", roles_hint: str = "") -> str:
            normalized_goal = str(goal or "").strip()
            if not normalized_goal:
                return "goal is required to start team mode."
            payload = {
                "type": "mode_activate",
                "mode": "team",
                "goal": normalized_goal,
                "reason": str(reason or "").strip(),
                "roles_hint": str(roles_hint or "").strip(),
            }
            logger.info(
                "tool.start_team called: goal_len=%s reason=%s roles_hint=%s",
                len(normalized_goal),
                _preview(str(reason or "")),
                _preview(str(roles_hint or "")),
            )
            return json.dumps(payload, ensure_ascii=False)

        _register_tool(start_team)

    progressive_enabled = _env_flag("AGENT_PROGRESSIVE_TOOL_DISCLOSURE", default=True)
    if not progressive_enabled:
        for tool_obj in runtime_tools:
            setattr(tool_obj, _TOOL_VISIBILITY_ATTR, "fixed")
        logger.info(
            "Agent tools prepared (legacy): discovered=%s enabled=%s names=%s",
            len(discovered_tools),
            len(runtime_tools),
            ",".join(tool_item.name for tool_item in runtime_tools),
        )
        return runtime_tools

    fixed_tool_names = _resolve_fixed_tool_names(enabled_tool_names=set(runtime_tool_map.keys()))
    lazy_tool_names = sorted(
        name for name in runtime_tool_map.keys() if name not in fixed_tool_names
    )

    if not lazy_tool_names:
        exposed = [tool_obj for tool_obj in runtime_tools if tool_obj.name in fixed_tool_names]
        for tool_obj in exposed:
            setattr(tool_obj, _TOOL_VISIBILITY_ATTR, "fixed")
        logger.info(
            "Agent tools prepared (progressive,no-lazy): discovered=%s fixed=%s names=%s",
            len(discovered_tools),
            len(exposed),
            ",".join(tool_item.name for tool_item in exposed),
        )
        return exposed

    metadata_by_name = {item.name: item.description for item in discovered_tools}

    # 将 Skill 也直接作为伪工具元数据注入到 Registry（这样模型直接查 "翻译"，就能找到 use_skill，且知道传入啥 skill）
    # 或者直接把 skill 的 keyword 附加给 use_skill 的 description 中
    skills = discover_available_skills()
    if skills:
        skill_docs = []
        for s in skills:
            skill_docs.append(
                f"[{s.name}]: {s.description} (keywords: {getattr(s, 'keywords', '')})"
            )

        # 增强 use_skill 的可搜索性
        if "use_skill" in metadata_by_name:
            metadata_by_name["use_skill"] += "\nAvailable skills to use:\n" + "\n".join(skill_docs)

        # Also update the tool object description so ToolRegistry can index it
        if "use_skill" in runtime_tool_map:
            runtime_tool_map["use_skill"].description = metadata_by_name["use_skill"]

    from .tools.registry import ToolRegistry

    registry = ToolRegistry()
    for name, obj in runtime_tool_map.items():
        if name in lazy_tool_names:
            registry.register(name, obj)

    @tool(
        "search_tools",
        description="Search for available lazy tools by query and return their compact manifests.",
        args_schema=SearchToolsInput,
    )
    def search_tools(query: str, reason: str = "") -> str:
        safe_query = str(query or "").strip()
        if not safe_query:
            return "query is required."

        results = registry.search(safe_query, top_k=3)
        if not results:
            return json.dumps({"type": "tool_search_result", "query": safe_query, "tools": []})

        tools_payload = []
        for t in results:
            name = str(getattr(t, "name", ""))
            tools_payload.append(
                {
                    "tool_name": name,
                    "description": metadata_by_name.get(name, ""),
                    "manifest": _schema_manifest_for_tool(t),
                }
            )

        return json.dumps(
            {
                "type": "tool_search_result",
                "query": safe_query,
                "reason": reason,
                "tools": tools_payload,
            },
            ensure_ascii=False,
        )

    for tool_name, tool_obj in runtime_tool_map.items():
        visibility = "fixed" if tool_name in fixed_tool_names else "lazy"
        setattr(tool_obj, _TOOL_VISIBILITY_ATTR, visibility)
    setattr(search_tools, _TOOL_VISIBILITY_ATTR, "fixed")

    all_tools = list(runtime_tools)
    all_tools.append(search_tools)

    logger.info(
        "Agent tools prepared (progressive): discovered=%s fixed=%s lazy=%s registered=%s names=%s",
        len(discovered_tools),
        len(fixed_tool_names),
        len(lazy_tool_names),
        len(all_tools),
        ",".join(tool_item.name for tool_item in all_tools),
    )
    return all_tools
