import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryWriteItemInput(BaseModel):
    memory_type: str = Field(description="`user_memory` or `knowledge_memory`.")
    content: str = Field(
        description="Declarative long-term memory fragment statement. Never provide Q/A transcripts."
    )
    canonical_text: str = Field(
        default="",
        description="Canonical fragment statement used for dedup. Prefer the shortest stable form.",
    )
    title: str = Field(default="", description="Short label for the memory item.")
    dedup_key: str = Field(
        default="",
        description="Optional stable slot key for updates, e.g. `user:response_style`.",
    )
    action: str = Field(default="ADD", description="ADD, UPDATE, DELETE, NONE, or SUPERSEDE.")
    confidence: float = Field(default=0.9, description="Confidence between 0 and 1.")


class WriteMemoryInput(BaseModel):
    items: list[MemoryWriteItemInput] = Field(
        description="Memory items to write via the canonical reconcile pipeline."
    )


class QueryMemoryInput(BaseModel):
    query: str = Field(
        description="Semantic query used to find possibly related long-term memories before writing or answering."
    )
    memory_type: str = Field(
        default="",
        description="Optional memory type filter, e.g. `user_memory` or `knowledge_memory`.",
    )
    limit: int = Field(default=5, description="Maximum number of memory items to return.")


def build_query_memory_tool(
    query_memory_fn: Callable[[str, str | None, int], list[dict[str, Any]]],
) -> Any:
    @tool(
        "query_memory",
        description=(
            "Query existing long-term memory for the current user/project/session scope. "
            "Use this before write_memory when you suspect a duplicate, update, or contradiction."
        ),
        args_schema=QueryMemoryInput,
    )
    def query_memory(query: str, memory_type: str = "", limit: int = 5) -> str:
        normalized_query = str(query or "").strip()
        normalized_memory_type = str(memory_type or "").strip() or None
        normalized_limit = max(1, int(limit))
        logger.info(
            "tool.query_memory called: query_len=%s memory_type=%s limit=%s",
            len(normalized_query),
            normalized_memory_type or "-",
            normalized_limit,
        )
        result = query_memory_fn(normalized_query, normalized_memory_type, normalized_limit)
        logger.info("tool.query_memory success: results=%s", len(result))
        return json.dumps({"results": result}, ensure_ascii=False)

    return query_memory


def build_write_memory_tool(
    write_memory_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> Any:
    @tool(
        "write_memory",
        description=(
            "Persist durable long-term memory fragments for the current user/project/session. "
            "Use this when you identify a stable user preference, project fact, or durable knowledge. "
            "Each item must be a declarative fragment statement, not Q/A dialogue."
        ),
        args_schema=WriteMemoryInput,
    )
    def write_memory(items: list[MemoryWriteItemInput]) -> str:
        normalized_items = [item.model_dump() for item in items]
        logger.info("tool.write_memory called: items=%s", len(normalized_items))
        result = write_memory_fn(normalized_items)
        logger.info("tool.write_memory success: results=%s", len(result))
        return json.dumps({"results": result}, ensure_ascii=False)

    return write_memory
