from typing import Any

from ..tools.document import (
    build_list_document_tool,
    build_read_document_tool,
    build_search_document_tool,
)


def build_document_tools(deps: Any) -> list[Any]:
    tools: list[Any] = []
    tools.append(
        build_search_document_tool(
            deps.search_document_fn,
            getattr(deps, "search_document_evidence_fn", None),
        )
    )
    list_documents_fn = getattr(deps, "list_documents_fn", None)
    if callable(list_documents_fn):
        tools.append(build_list_document_tool(list_documents_fn))
    read_document_fn = getattr(deps, "read_document_fn", None)
    doc_id_to_text = getattr(deps, "doc_id_to_text", None)
    default_id = getattr(deps, "doc_id_default", "")
    if callable(read_document_fn) or doc_id_to_text:
        tools.append(
            build_read_document_tool(
                read_document_fn, deps.search_document_fn, doc_id_to_text, default_id
            )
        )
    return tools
