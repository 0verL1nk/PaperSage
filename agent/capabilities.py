import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .scholarly_search import (
    ScholarlySearchError,
    format_search_papers_results,
    search_semantic_scholar,
)
from .skills.loader import discover_available_skills, get_skill

logger = logging.getLogger(__name__)


class SearchDocumentInput(BaseModel):
    query: str = Field(
        description="Specific paper question or keyword used to retrieve evidence snippets."
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
        description="Skill template name. Available skills: summary, critical_reading, method_compare, translation, mindmap."
    )
    task: str = Field(
        description="Current user task where the selected skill guidance should be applied."
    )


DEFAULT_MAX_QUERY_CHARS = 1200
DEFAULT_WEB_MAX_RESULTS = 5
DEFAULT_WEB_TIMEOUT_SECONDS = 8.0
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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
            return _format_web_results(results, title_key="title", url_key="href", snippet_key="body")

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


def build_agent_tools(
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
    allowed_tools: set[str] | None = None,
):
    """构建 Agent 工具集

    Args:
        search_document_fn: RAG 检索函数
        read_document_fn: 分块读取函数，接收 (offset, limit)，返回 (content, total_length)
    """
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
                logger.info("tool.search_document success: mode=evidence_json evidences=%s", evidence_count)
                return json.dumps(evidence_payload, ensure_ascii=False)
            except Exception:
                logger.exception("tool.search_document evidence function failed, fallback=text")
                # 回退到文本输出，避免工具失败影响主流程
                return search_document_fn(safe_query)
        logger.info("tool.search_document success: mode=text")
        return search_document_fn(safe_query)

    if read_document_fn is not None:
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
            logger.info("tool.read_document success: chunk_len=%s total_len=%s", len(content), total)
            return result
    else:
        read_document = None

    web_search_clients: list[tuple[str, Any]] = []

    searxng_client = _build_searxng_web_search_client()
    if searxng_client is not None:
        web_search_clients.append(("searxng_public_pool", searxng_client))
        logger.info("tool.search_web provider initialized: searxng_public_pool")

    wikipedia_client = _build_wikipedia_web_search_client()
    if wikipedia_client is not None:
        web_search_clients.append(("wikipedia_api", wikipedia_client))
        logger.info("tool.search_web provider initialized: wikipedia_api")

    if not web_search_clients:
        logger.warning("tool.search_web no primary provider initialized")

    if not web_search_clients:
        allow_ddg_fallback = _env_flag("AGENT_WEB_ENABLE_DDG_FALLBACK", default=False)
        if allow_ddg_fallback:
            web_search_client = _build_native_web_search_client()
            if web_search_client is not None:
                web_search_clients.append(("native_duckduckgo_search", web_search_client))
                logger.info("tool.search_web provider fallback initialized: native_duckduckgo_search")
            else:
                try:
                    web_search_client = DuckDuckGoSearchRun(name="search_web")
                    web_search_clients.append(("langchain_duckduckgo", web_search_client))
                    logger.info("tool.search_web provider fallback initialized: langchain_duckduckgo")
                except Exception as exc:
                    logger.warning("tool.search_web ddg fallback unavailable: %s", exc)
        else:
            logger.warning(
                "tool.search_web provider unavailable: searxng disabled/unreachable and ddg fallback disabled"
            )

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
        if not web_search_clients:
            logger.warning("tool.search_web unavailable: no provider")
            return (
                "Web search is unavailable in current environment. "
                "Configure AGENT_SEARXNG_BASE_URLS or enable AGENT_WEB_ENABLE_DDG_FALLBACK=1."
            )
        errors: list[str] = []
        for provider_name, provider in web_search_clients:
            try:
                response = provider.run(safe_query)
                response_text = response if isinstance(response, str) else str(response)
                if not response_text or response_text.strip() == "No web search results found.":
                    errors.append(f"{provider_name}: no results")
                    continue
                logger.info(
                    "tool.search_web success: provider=%s response_len=%s",
                    provider_name,
                    len(response_text),
                )
                return response_text
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                logger.info("tool.search_web provider failed: %s (%s)", provider_name, exc)
        logger.error("tool.search_web failed: providers=%s", " | ".join(errors))
        return f"Web search failed: {' | '.join(errors)}"

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
        try:
            papers = search_semantic_scholar(query=safe_query, limit=limit)
        except ScholarlySearchError as exc:
            logger.warning("tool.search_papers failed: %s", exc)
            return f"Academic search failed: {exc} Try narrowing the topic or use search_web."
        logger.info("tool.search_papers success: results=%s", len(papers))
        return format_search_papers_results(papers)

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

        # 尝试从文件加载 skill
        skill = get_skill(normalized_name)
        if skill:
            # 找到 skill，返回完整指令
            logger.info("tool.use_skill success: skill=%s", normalized_name)
            return f"Skill: {skill.name}\nDescription: {skill.description}\n\nInstructions:\n{skill.instructions}\n\nTask: {task}"

        # 如果没找到，返回错误信息
        options = _get_skill_options()
        logger.warning("tool.use_skill unknown skill: %s", normalized_name)
        return f"Unknown skill '{skill_name}'. Available skills: {options}."

    tools = [search_document, search_web, search_papers, use_skill]
    if read_document is not None:
        tools.insert(1, read_document)  # 在 search_document 后插入 read_document

    if allowed_tools is None:
        normalized_allowlist = None
    else:
        normalized_allowlist = {name.strip().lower() for name in allowed_tools if name.strip()}

    filtered_tools = [
        tool_item for tool_item in tools if _tool_enabled(tool_item.name, normalized_allowlist)
    ]
    logger.info(
        "Agent tools prepared: total=%s enabled=%s names=%s",
        len(tools),
        len(filtered_tools),
        ",".join(tool_item.name for tool_item in filtered_tools),
    )
    return filtered_tools
