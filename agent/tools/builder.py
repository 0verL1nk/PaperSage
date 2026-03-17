"""工具构建器模块"""
import logging
from collections.abc import Callable
from typing import Any

from ..tools import ToolMetadata
from .document import build_list_document_tool, build_read_document_tool, build_search_document_tool
from .paper_search import build_paper_search_tool
from .skill import build_skill_tool
from .utils import _env_flag
from .web_search import build_web_search_tool

logger = logging.getLogger(__name__)

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
)


def _tool_enabled(tool_name: str, allowed_tools: set[str] | None = None) -> bool:
    normalized = tool_name.strip().lower()
    if allowed_tools is not None and normalized not in allowed_tools:
        return False
    env_key = f"AGENT_DISABLE_{normalized.upper()}"
    return not _env_flag(env_key, default=False)


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
        search_document_evidence_fn: RAG 检索函数（返回结构化证据）
        read_document_fn: 分块读取函数，接收 (offset, limit)，返回 (content, total_length)
        list_documents_fn: 文档列表函数，返回 list[{doc_uid, doc_name, char_length}]
        allowed_tools: 允许的工具名称集合
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

    runtime_tools: list[Any] = []

    if "search_document" in enabled_tool_names:
        tool = build_search_document_tool(search_document_fn, search_document_evidence_fn)
        runtime_tools.append(tool)

    if "list_document" in enabled_tool_names and list_documents_fn is not None:
        tool = build_list_document_tool(list_documents_fn)
        runtime_tools.append(tool)

    if "read_document" in enabled_tool_names and read_document_fn is not None:
        tool = build_read_document_tool(read_document_fn, search_document_fn)
        runtime_tools.append(tool)

    if "search_web" in enabled_tool_names:
        tool = build_web_search_tool()
        runtime_tools.append(tool)

    if "search_papers" in enabled_tool_names:
        tool = build_paper_search_tool()
        runtime_tools.append(tool)

    if "use_skill" in enabled_tool_names:
        tool = build_skill_tool()
        runtime_tools.append(tool)

    logger.info(
        "Agent tools prepared: discovered=%s enabled=%s names=%s",
        len(discovered_tools),
        len(runtime_tools),
        ",".join(tool.name for tool in runtime_tools),
    )
    return runtime_tools
