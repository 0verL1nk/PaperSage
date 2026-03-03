import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

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
            with DDGS() as client:
                results = client.text(query, max_results=5)
            if not isinstance(results, list) or not results:
                return "No web search results found."
            lines: list[str] = []
            for idx, item in enumerate(results[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                href = str(item.get("href") or "").strip()
                body = str(item.get("body") or "").strip()
                snippet = _preview(body, limit=180) if body else ""
                lines.append(
                    f"{idx}. {title or 'Untitled'}\nURL: {href or 'n/a'}\nSnippet: {snippet or '-'}"
                )
            return "\n\n".join(lines) if lines else "No web search results found."

    return _NativeDuckDuckGoSearch()


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

    web_search_client = _build_native_web_search_client()
    if web_search_client is not None:
        logger.info("tool.search_web provider initialized: native_duckduckgo_search")
    else:
        try:
            web_search_client = DuckDuckGoSearchRun(name="search_web")
            logger.info("tool.search_web provider initialized: langchain_duckduckgo")
        except Exception as exc:
            logger.warning("tool.search_web langchain provider unavailable: %s", exc)

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
        if web_search_client is None:
            logger.warning("tool.search_web unavailable: no provider")
            return (
                "Web search is unavailable in current environment. "
                "Install 'ddgs' or 'duckduckgo-search' to enable it."
            )
        try:
            response = web_search_client.run(safe_query)
            response_text = response if isinstance(response, str) else str(response)
            logger.info("tool.search_web success: response_len=%s", len(response_text))
            return response_text
        except Exception as exc:
            logger.exception("tool.search_web failed")
            return f"Web search failed: {exc}"

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
