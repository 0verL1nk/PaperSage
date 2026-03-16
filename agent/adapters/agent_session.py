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
    project_uid: str | None = None,
    session_uid: str | None = None,
    user_uuid: str | None = None,
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
        project_uid=project_uid,
        session_uid=session_uid,
        user_uuid=user_uuid,
    )
