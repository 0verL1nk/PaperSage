from typing import Any

from ..paper_agent import create_paper_agent_session


def create_leader_session(
    *,
    llm: Any,
    search_document_fn,
    search_document_evidence_fn=None,
    read_document_fn=None,
    list_documents_fn=None,
    document_name: str = "",
    project_name: str = "",
    scope_summary: str = "",
):
    return create_paper_agent_session(
        llm=llm,
        search_document_fn=search_document_fn,
        search_document_evidence_fn=search_document_evidence_fn,
        read_document_fn=read_document_fn,
        list_documents_fn=list_documents_fn,
        document_name=document_name,
        project_name=project_name,
        scope_summary=scope_summary,
    )
