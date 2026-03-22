"""文档工具模块"""

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .utils import (
    _is_dangerous_query,
    _is_low_information_query,
    _normalize_query_cache_key,
    _preview,
    _query_overlap_score,
    _sanitize_query,
)

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
    doc_id: str = Field(
        default="",
        description="Document identifier to read. Use doc_id from list_document output. If empty, reads the only available document.",
    )
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
    cached_text_results: dict[str, str] = {}
    cached_evidence_payloads: dict[str, dict[str, Any]] = {}
    cached_queries: dict[str, str] = {}
    query_family_threshold = 0.75

    def _build_dedupe_payload(
        payload: dict[str, Any],
        *,
        query: str,
        reason: str,
        matched_query: str | None = None,
    ) -> str:
        dedupe_payload = dict(payload)
        meta = dedupe_payload.get("meta")
        meta_payload = dict(meta) if isinstance(meta, dict) else {}
        base_message = "Identical query reused cached result; refine query instead of repeating it."
        if reason == "same_query_family":
            base_message = (
                "A closely related query already returned evidence. Do not call search_document again "
                "for this query family; synthesize from the existing results or move to the final answer."
            )
        meta_payload["dedupe"] = {
            "reused_cached_result": True,
            "query": query,
            "reason": reason,
            "matched_query": matched_query or query,
            "message": base_message,
            "should_stop": True,
        }
        dedupe_payload["meta"] = meta_payload
        return json.dumps(dedupe_payload, ensure_ascii=False)

    def _build_policy_block_payload(query: str, *, reason: str, message: str) -> str:
        payload = {
            "evidences": [],
            "meta": {
                "query_policy": {
                    "blocked": True,
                    "query": query,
                    "reason": reason,
                    "message": message,
                }
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _find_similar_cache_key(query: str, *, exact_key: str) -> tuple[str, str] | None:
        best_match: tuple[str, str] | None = None
        best_score = 0.0
        for cached_key, cached_query in cached_queries.items():
            if cached_key == exact_key:
                continue
            score = _query_overlap_score(query, cached_query)
            if score < query_family_threshold or score <= best_score:
                continue
            best_score = score
            best_match = (cached_key, cached_query)
        return best_match

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
        cache_key = _normalize_query_cache_key(safe_query)
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
        if _is_low_information_query(safe_query):
            logger.warning(
                "tool.search_document blocked: low_information_query query_preview=%s",
                _preview(safe_query),
            )
            if search_document_evidence_fn is not None:
                return _build_policy_block_payload(
                    safe_query,
                    reason="low_information_query",
                    message=(
                        "Query is too generic to produce new evidence. Reuse the existing evidence or ask "
                        "a more specific document question instead of repeating broad terms like page/table/result."
                    ),
                )
            return (
                "Blocked by tool policy: query is too generic to produce new evidence. "
                "Use a more specific document question."
            )
        if search_document_evidence_fn is not None:
            cached_payload = cached_evidence_payloads.get(cache_key)
            if isinstance(cached_payload, dict):
                logger.info(
                    "tool.search_document dedupe hit: mode=evidence_json query_preview=%s",
                    _preview(safe_query),
                )
                return _build_dedupe_payload(
                    cached_payload,
                    query=safe_query,
                    reason="identical_or_normalized_query",
                )
            similar_match = _find_similar_cache_key(safe_query, exact_key=cache_key)
            if similar_match is not None:
                matched_key, matched_query = similar_match
                similar_payload = cached_evidence_payloads.get(matched_key)
                if isinstance(similar_payload, dict):
                    logger.info(
                        "tool.search_document family dedupe hit: mode=evidence_json query_preview=%s matched_query_preview=%s",
                        _preview(safe_query),
                        _preview(matched_query),
                    )
                    return _build_dedupe_payload(
                        similar_payload,
                        query=safe_query,
                        reason="same_query_family",
                        matched_query=matched_query,
                    )
            try:
                evidence_payload = search_document_evidence_fn(safe_query)
                evidence_count = (
                    len(evidence_payload.get("evidences", []))
                    if isinstance(evidence_payload, dict)
                    else 0
                )
                if isinstance(evidence_payload, dict):
                    cached_evidence_payloads[cache_key] = dict(evidence_payload)
                    cached_queries.setdefault(cache_key, safe_query)
                logger.info(
                    "tool.search_document success: mode=evidence_json evidences=%s",
                    evidence_count,
                )
                return json.dumps(evidence_payload, ensure_ascii=False)
            except Exception:
                logger.exception("tool.search_document evidence function failed, fallback=text")
                if cache_key in cached_text_results:
                    logger.info(
                        "tool.search_document dedupe hit: mode=text query_preview=%s",
                        _preview(safe_query),
                    )
                    return cached_text_results[cache_key]
                similar_match = _find_similar_cache_key(safe_query, exact_key=cache_key)
                if similar_match is not None:
                    matched_key, matched_query = similar_match
                    cached_result = cached_text_results.get(matched_key)
                    if isinstance(cached_result, str):
                        logger.info(
                            "tool.search_document family dedupe hit: mode=text query_preview=%s matched_query_preview=%s",
                            _preview(safe_query),
                            _preview(matched_query),
                        )
                        return cached_result
                result = search_document_fn(safe_query)
                cached_text_results[cache_key] = result
                cached_queries.setdefault(cache_key, safe_query)
                return result
        if cache_key in cached_text_results:
            logger.info(
                "tool.search_document dedupe hit: mode=text query_preview=%s",
                _preview(safe_query),
            )
            return cached_text_results[cache_key]
        similar_match = _find_similar_cache_key(safe_query, exact_key=cache_key)
        if similar_match is not None:
            matched_key, matched_query = similar_match
            cached_result = cached_text_results.get(matched_key)
            if isinstance(cached_result, str):
                logger.info(
                    "tool.search_document family dedupe hit: mode=text query_preview=%s matched_query_preview=%s",
                    _preview(safe_query),
                    _preview(matched_query),
                )
                return cached_result
        logger.info("tool.search_document success: mode=text")
        result = search_document_fn(safe_query)
        cached_text_results[cache_key] = result
        cached_queries.setdefault(cache_key, safe_query)
        return result

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
    read_document_fn: Callable[[int, int], tuple[str, int]] | None,
    search_document_fn: Callable[[str], str],
    doc_id_to_text: dict[str, str] | None = None,
    default_doc_id: str = "",
) -> Any:
    """构建文档阅读工具

    Args:
        read_document_fn: 单文档读取函数，接收 (offset, limit)，返回 (content, total_len)
        search_document_fn: RAG 检索函数
        doc_id_to_text: 多文档映射表 {doc_id: document_text}，非空时启用多文档模式
        default_doc_id: 默认文档 ID（多文档模式下且 doc_id 为空时使用）
    """

    @tool(
        "read_document",
        description="Read a specific portion of a document by character offset and limit. "
        "Use doc_id to select which document (from list_document output). "
        "Use offset to skip to a position, limit to control chunk size. "
        "Set include_rag=True to get relevant context around the reading position.",
        args_schema=ReadDocumentInput,
    )
    def read_document(
        doc_id: str = "", offset: int = 0, limit: int = 2000, include_rag: bool = False
    ) -> str:
        logger.info(
            "tool.read_document called: doc_id=%s offset=%s limit=%s include_rag=%s",
            doc_id,
            offset,
            limit,
            include_rag,
        )

        # 多文档模式
        if doc_id_to_text is not None:
            target_doc_id = doc_id.strip() or default_doc_id
            if not target_doc_id:
                available = list(doc_id_to_text.keys())
                if len(available) == 1:
                    target_doc_id = available[0]
                else:
                    return (
                        "read_document requires doc_id when multiple documents are in scope. "
                        "Use list_document(verbose=True) to see available doc_ids. "
                        f"Available: {available}"
                    )
            if target_doc_id not in doc_id_to_text:
                return f"Document '{target_doc_id}' not found. Use list_document(verbose=True) to see available doc_ids."
            text = doc_id_to_text[target_doc_id]
            total = len(text)
            content = text[offset : offset + limit]
            doc_label = target_doc_id
        else:
            # 单文档模式（向后兼容）
            if callable(read_document_fn):
                content, total = read_document_fn(offset, limit)
            else:
                content, total = "", 0
            doc_label = "文档"

        rag_context = ""
        if include_rag:
            query = f"position_{offset}"
            rag_context = search_document_fn(query)

        result = f"=== {doc_label} (字符位置 {offset} - {offset + limit}) ===\n"
        result += f"总长度: {total} 字符\n"
        result += f"当前 chunk: {len(content)} 字符\n\n"
        if rag_context:
            result += f"=== 相关上下文 (RAG) ===\n{rag_context}\n\n"
        result += f"=== 内容 ===\n{content}"
        logger.info(
            "tool.read_document success: doc_id=%s chunk_len=%s total_len=%s",
            doc_id or "default",
            len(content),
            total,
        )
        return result

    return read_document
