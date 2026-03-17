"""论文搜索工具模块"""
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..scholarly_search import (
    ScholarlySearchError,
    format_search_papers_results,
    search_semantic_scholar,
)
from .utils import _is_dangerous_query, _preview, _sanitize_query

logger = logging.getLogger(__name__)


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


def build_paper_search_tool() -> Any:
    """构建论文搜索工具"""

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

    return search_papers
