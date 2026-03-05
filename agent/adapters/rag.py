from collections.abc import Callable
from typing import Any

from ..rag.hybrid import (
    build_local_evidence_retriever_with_settings,
    build_project_evidence_retriever_with_settings,
)

EvidenceRetriever = Callable[[str], dict[str, Any]]


def create_project_evidence_retriever(
    *,
    documents: list[dict[str, str]],
    project_uid: str,
) -> EvidenceRetriever:
    if len(documents) == 1:
        only = documents[0]
        return build_local_evidence_retriever_with_settings(
            document_text=only["text"],
            doc_uid=only["doc_uid"],
            doc_name=only["doc_name"],
            project_uid=project_uid,
        )
    return build_project_evidence_retriever_with_settings(
        documents=documents,
        project_uid=project_uid,
    )
