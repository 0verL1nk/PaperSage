"""文档工具模块"""
import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .utils import _is_dangerous_query, _preview, _sanitize_query

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


def build_search_document_tool(
    search_document_fn: Callable[[str], str],
    search_document_evidence_fn: Callable[[str], dict[str, Any]] | None = None,
) -> Any:
    """构建文档搜索工具"""

    @tool(
        "search_document",
        description="""Search uploaded paper content for relevant evidence snippets using RAG.

Returns: JSON object with structure:
{
  "evidences": [
    {
      "chunk_id": "unique_chunk_identifier",
      "text": "evidence text content",
      "score": 0.95,
      "page_no": 5,
      "offset_start": 100,
      "offset_end": 200,
      "doc_name": "document.pdf"
    }
  ]
}

IMPORTANT: When citing evidence in your answer, use the format:
<evidence>chunk_id|p{page_no}|o{offset_start}-{offset_end}</evidence>

Example: Based on the research<evidence>chunk_abc123|p5|o100-200</evidence>, the method is effective.""",
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
                return search_document_fn(safe_query)
        logger.info("tool.search_document success: mode=text")
        return search_document_fn(safe_query)

    return search_document


def build_list_document_tool(
    list_documents_fn: Callable[[], list[dict[str, Any]]],
) -> Any:
    """构建文档列表工具"""

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

    return list_document


def build_read_document_tool(
    read_document_fn: Callable[[int, int], tuple[str, int]],
    search_document_fn: Callable[[str], str],
) -> Any:
    """构建文档阅读工具"""

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

        rag_context = ""
        if include_rag:
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

    return read_document
