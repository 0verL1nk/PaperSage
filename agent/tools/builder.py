"""工具构建器模块"""
import logging
from collections.abc import Callable
from typing import Any

from .document import build_list_document_tool, build_read_document_tool, build_search_document_tool
from .paper_search import search_papers
from .plan_tools import read_plan, write_plan
from .skill import use_skill
from .types import ToolMetadata
from .web_search import search_web

logger = logging.getLogger(__name__)


def discover_available_tools(
    *, read_document_enabled: bool = True, list_document_enabled: bool = True
) -> list[ToolMetadata]:
    """发现可用工具（元信息阶段，不创建运行时实例）"""
    tools = [
        ToolMetadata(name=write_plan.name, description=write_plan.description.split("\n")[0]),
        ToolMetadata(name=read_plan.name, description=read_plan.description.split("\n")[0]),
        ToolMetadata(name=search_web.name, description=search_web.description.split("\n")[0]),
        ToolMetadata(name=search_papers.name, description=search_papers.description.split("\n")[0]),
        ToolMetadata(name=use_skill.name, description=use_skill.description.split("\n")[0]),
    ]

    tools.append(ToolMetadata(name="search_document", description="Search uploaded paper content for relevant evidence snippets using RAG."))

    if list_document_enabled:
        tools.append(ToolMetadata(name="list_document", description="List all documents loaded in the current project scope."))
    if read_document_enabled:
        tools.append(ToolMetadata(name="read_document", description="Read a specific portion of the uploaded paper with pagination."))

    return tools


def build_agent_tools(
    search_document_fn: Callable[[str], str] | None = None,
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
    read_document_fn: Callable[[int, int], tuple[str, int]] | None = None,
    list_documents_fn: Callable[[], list[dict[str, Any]]] | None = None,
):
    """构建 Agent 工具集"""
    runtime_tools: list[Any] = []

    if callable(search_document_fn):
        runtime_tools.append(build_search_document_tool(search_document_fn, search_document_evidence_fn))
        if list_documents_fn:
            runtime_tools.append(build_list_document_tool(list_documents_fn))
        if read_document_fn:
            runtime_tools.append(build_read_document_tool(read_document_fn, search_document_fn))

    runtime_tools.extend([search_web, search_papers, use_skill, write_plan, read_plan])

    logger.info("Agent tools prepared: count=%s", len(runtime_tools))
    return runtime_tools
